!pip install torch torchvision kagglehub -q

import torch, torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)
from torch.utils.data import DataLoader
from google.colab import files
from PIL import Image
import os, torchvision, zipfile, shutil, glob, random

os.makedirs("/content/binary/leaf",     exist_ok=True)
os.makedirs("/content/binary/not_leaf", exist_ok=True)
print("✅ Folders ready!")

# ── Download PlantVillage leaf images via kagglehub ──────────────────────────
print("\n⬇️ Downloading PlantVillage dataset via kagglehub...")
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"📂 Downloaded to: {path}")

# Find train folder inside the downloaded path
train_folder = None
for root, dirs, files_list in os.walk(path):
    if os.path.basename(root) == "train":
        train_folder = root
        break

if train_folder is None:
    # fallback: search for any folder with class subfolders
    for root, dirs, files_list in os.walk(path):
        if len(dirs) > 10:   # PlantVillage has 38 classes
            train_folder = root
            break

print(f"📂 Using train folder: {train_folder}")

# Copy leaf images — take up to 100 per class (38 classes = ~3800 images)
leaf_count = 0
class_dirs = [
    d for d in os.listdir(train_folder)
    if os.path.isdir(os.path.join(train_folder, d))
]
print(f"🌿 Found {len(class_dirs)} plant classes")

for cls in class_dirs:
    cls_path = os.path.join(train_folder, cls)
    imgs = [
        f for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    random.shuffle(imgs)
    for fname in imgs[:100]:   # max 100 per class
        src = os.path.join(cls_path, fname)
        dst = f"/content/binary/leaf/leaf_{leaf_count}_{fname}"
        shutil.copy(src, dst)
        leaf_count += 1

print(f"✅ Total leaf images copied: {leaf_count}")

# ── Upload not-leaf ZIP (your random images) ─────────────────────────────────
print("\n📁 Upload your NOT-LEAF zip file (random images: cars, people, objects, balls etc.)...")
uploaded = files.upload()
notleaf_zip = list(uploaded.keys())[0]
print(f"✅ Got: {notleaf_zip}")

with zipfile.ZipFile(notleaf_zip, 'r') as z:
    z.extractall("/content/notleaf_raw")
print("✅ Not-leaf zip extracted!")

count2 = 0
for root, dirs, fnames in os.walk("/content/notleaf_raw"):
    for fname in fnames:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            src = os.path.join(root, fname)
            dst = f"/content/binary/not_leaf/notleaf_{count2}_{fname}"
            shutil.copy(src, dst)
            count2 += 1
print(f"✅ Total uploaded not-leaf images: {count2}")

# ── Auto download CIFAR-10 + STL-10 as extra negatives ───────────────────────
print("\n⬇️ Auto downloading CIFAR-10...")
cifar = torchvision.datasets.CIFAR10(
    root="./cifar", train=True, download=True
)
for i in range(500):
    img, _ = cifar[i]
    img.save(f"/content/binary/not_leaf/cifar_{i}.jpg")
print("✅ CIFAR done!")

print("⬇️ Auto downloading STL-10...")
stl = torchvision.datasets.STL10(
    root="./stl", split="train", download=True
)
for i in range(300):
    img, _ = stl[i]
    img.save(f"/content/binary/not_leaf/stl_{i}.jpg")
print("✅ STL done!")

print(f"\n📊 Final counts:")
print(f"Leaf    : {len(os.listdir('/content/binary/leaf'))}")
print(f"Not-leaf: {len(os.listdir('/content/binary/not_leaf'))}")

# ── Prepare dataset ───────────────────────────────────────────────────────────
tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

full_ds = datasets.ImageFolder(
    "/content/binary", transform=tf_train
)
print(f"\nClasses found: {full_ds.classes}")
print(f"Class mapping: {full_ds.class_to_idx}")
# NOTE: 'leaf' < 'not_leaf' alphabetically → leaf=0, not_leaf=1
LEAF_IDX = full_ds.class_to_idx['leaf']
print(f"Leaf index: {LEAF_IDX}  ← app uses prob[{LEAF_IDX}]")

n = len(full_ds)
n_train = int(0.8 * n)
n_val   = n - n_train
train_ds, val_ds = torch.utils.data.random_split(
    full_ds, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
val_ds.dataset.transform = tf_val

train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)
print(f"Train: {n_train} | Val: {n_val}")

# ── Build MobileNetV3 ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

validator = mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
)
validator.classifier[3] = nn.Linear(1024, 2)
validator = validator.to(device)

optimizer = torch.optim.AdamW(
    validator.parameters(), lr=3e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15
)
criterion = nn.CrossEntropyLoss()

# ── Train ─────────────────────────────────────────────────────────────────────
best_val_acc = 0.0
EPOCHS = 15
print("\n🚀 Training started...")
print("="*50)

for epoch in range(EPOCHS):
    validator.train()
    tr_c = tr_t = 0
    for imgs, labels in train_ld:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out  = validator(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        tr_c += (out.argmax(1) == labels).sum().item()
        tr_t += len(labels)

    validator.eval()
    vl_c = vl_t = 0
    with torch.no_grad():
        for imgs, labels in val_ld:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            out    = validator(imgs)
            vl_c  += (out.argmax(1) == labels).sum().item()
            vl_t  += len(labels)

    tr_acc = tr_c / tr_t * 100
    vl_acc = vl_c / vl_t * 100
    scheduler.step()

    status = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(
            validator.state_dict(),
            "/content/leaf_validator.pt"
        )
        status = " ✅ BEST SAVED"

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {tr_acc:.1f}% | "
          f"Val: {vl_acc:.1f}%{status}")

print(f"\n🎉 Best val accuracy: {best_val_acc:.1f}%")

# ── Test on sample images ─────────────────────────────────────────────────────
print("\n🧪 Quick test results:")
validator.load_state_dict(
    torch.load("/content/leaf_validator.pt", map_location=device)
)
validator.eval()

test_leaf    = glob.glob("/content/binary/leaf/*.jpg")[:5]
test_notleaf = glob.glob("/content/binary/not_leaf/notleaf_*.jpg")[:10]

print(f"\n--- Leaf images (should all say LEAF) ---")
for fpath in test_leaf:
    img = Image.open(fpath).convert("RGB")
    t   = tf_val(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = validator(t)
        prob = torch.softmax(out, dim=-1)[0]
    lp     = prob[LEAF_IDX].item() * 100
    result = "✅ LEAF" if lp > 55 else "❌ NOT LEAF"
    print(f"{os.path.basename(fpath)[:35]}: {result} ({lp:.1f}%)")

print(f"\n--- Your random images (should all say NOT LEAF) ---")
for fpath in test_notleaf:
    img = Image.open(fpath).convert("RGB")
    t   = tf_val(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out  = validator(t)
        prob = torch.softmax(out, dim=-1)[0]
    lp     = prob[LEAF_IDX].item() * 100
    result = "✅ LEAF" if lp > 55 else "❌ NOT LEAF"
    print(f"{os.path.basename(fpath)[:35]}: {result} ({lp:.1f}%)")

# ── Download ──────────────────────────────────────────────────────────────────
print("\n⬇️ Downloading leaf_validator.pt...")
files.download("/content/leaf_validator.pt")
print("✅ Drop this file into your project folder alongside soil_validator.pt !")
