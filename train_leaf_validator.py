"""
train_leaf_validator.py
=======================
Trains a binary leaf / non-leaf classifier (MobileNetV3-small)
and saves it as leaf_validator.pt — same pattern as soil_validator.pt.

Positive class  (label=1): plant leaf images
Negative class  (label=0): random non-leaf images (CIFAR-10 + STL-10 + optional folder)

Requirements:
    pip install torch torchvision tqdm

Run:
    python train_leaf_validator.py

Output:
    leaf_validator.pt   (drop-in replacement, loaded the same way as soil_validator.pt)
"""

import os, random, pathlib, shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
EPOCHS       = 15
BATCH_SIZE   = 64
LR           = 3e-4
IMG_SIZE     = 224
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH    = "leaf_validator.pt"

# Optional: path to a folder of real leaf images (PlantVillage, iNaturalist, etc.)
# Set to None to rely only on the automatically downloaded plant datasets.
EXTRA_LEAF_DIR    = None   # e.g. r"C:\datasets\plantvillage\train"
# Optional: path to a folder of extra negative images (cars, faces, objects, etc.)
EXTRA_NONLEAF_DIR = None   # e.g. r"C:\datasets\imagenet_samples"

print(f"Using device: {DEVICE}")

# ── Transforms ──────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Helper: wrap any ImageFolder-style dataset with a fixed label ────────────
class LabelOverrideDataset(Dataset):
    def __init__(self, base_dataset, label: int):
        self.base  = base_dataset
        self.label = label
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return img, self.label


# ── Helper: load a plain image folder (no subfolders needed) ─────────────────
class FlatImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, folder, label, transform):
        self.paths = [
            p for p in pathlib.Path(folder).rglob("*")
            if p.suffix.lower() in self.EXTS
        ]
        self.label     = label
        self.transform = transform
        print(f"  FlatImageDataset: {len(self.paths)} images from {folder}")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.label


# ── Build datasets ───────────────────────────────────────────────────────────
print("\nDownloading / loading datasets …")

# --- Negatives: CIFAR-10 (32×32 objects — cars, planes, animals) ---
cifar_train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
cifar_val   = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)
neg_train   = LabelOverrideDataset(cifar_train, label=0)
neg_val     = LabelOverrideDataset(cifar_val,   label=0)

# --- Negatives: STL-10 (96×96 objects — better resolution) ---
try:
    stl_train = datasets.STL10(root="./data", split="train", download=True, transform=train_tf)
    stl_val   = datasets.STL10(root="./data", split="test",  download=True, transform=val_tf)
    neg_train = ConcatDataset([neg_train, LabelOverrideDataset(stl_train, 0)])
    neg_val   = ConcatDataset([neg_val,   LabelOverrideDataset(stl_val,   0)])
    print("  STL-10 loaded.")
except Exception as e:
    print(f"  STL-10 skipped ({e})")

# --- Negatives: extra folder (optional) ---
if EXTRA_NONLEAF_DIR and os.path.isdir(EXTRA_NONLEAF_DIR):
    neg_train = ConcatDataset([neg_train, FlatImageDataset(EXTRA_NONLEAF_DIR, 0, train_tf)])

# --- Positives: Oxford Flowers-102 (plant images — closest freely available) ---
# We use it purely as "plant / leaf-like" positives. For a production model
# replace/augment with actual PlantVillage images.
try:
    flowers_train = datasets.Flowers102(root="./data", split="train", download=True, transform=train_tf)
    flowers_val   = datasets.Flowers102(root="./data", split="val",   download=True, transform=val_tf)
    pos_train = LabelOverrideDataset(flowers_train, label=1)
    pos_val   = LabelOverrideDataset(flowers_val,   label=1)
    print("  Flowers-102 loaded as leaf positives.")
except Exception as e:
    print(f"  Flowers-102 skipped ({e})")
    pos_train, pos_val = None, None

# --- Positives: extra leaf folder (optional but STRONGLY recommended) ---
# Point EXTRA_LEAF_DIR at your PlantVillage dataset for best results.
if EXTRA_LEAF_DIR and os.path.isdir(EXTRA_LEAF_DIR):
    extra_pos = FlatImageDataset(EXTRA_LEAF_DIR, 1, train_tf)
    pos_train = ConcatDataset([pos_train, extra_pos]) if pos_train else extra_pos
    print(f"  Extra leaf images loaded from {EXTRA_LEAF_DIR}")

if pos_train is None:
    raise RuntimeError(
        "No positive (leaf) images found.\n"
        "Either Flowers-102 download failed, or set EXTRA_LEAF_DIR to a folder of leaf images."
    )

# ── Combine & balance ────────────────────────────────────────────────────────
train_ds = ConcatDataset([pos_train, neg_train])
val_ds   = ConcatDataset([pos_val,   neg_val])

n_pos = len(pos_train)
n_neg = len(neg_train)
print(f"\nTrain  — positives: {n_pos}  negatives: {n_neg}  total: {len(train_ds)}")
print(f"Val    — positives: {len(pos_val)}  negatives: {len(neg_val)}  total: {len(val_ds)}")

# Weighted sampler to handle class imbalance
w_pos = 1.0 / n_pos
w_neg = 1.0 / n_neg
sample_weights = [w_pos] * n_pos + [w_neg] * n_neg
sampler = WeightedRandomSampler(sample_weights, num_samples=min(len(train_ds), 40_000), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ── Model ────────────────────────────────────────────────────────────────────
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(1024, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training loop ────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (out.argmax(1) == labels).sum().item()
        train_total   += imgs.size(0)
    scheduler.step()

    # --- val ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            val_correct += (out.argmax(1) == labels).sum().item()
            val_total   += imgs.size(0)

    t_acc = train_correct / train_total * 100
    v_acc = val_correct   / val_total   * 100
    print(f"Epoch {epoch:02d}  train_loss={train_loss/train_total:.4f}  "
          f"train_acc={t_acc:.1f}%  val_acc={v_acc:.1f}%")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model ({v_acc:.1f}%) → {SAVE_PATH}")

print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")
print(f"Model saved to: {SAVE_PATH}")
print("\nDrop leaf_validator.pt into your project folder and update streamlit_app.py to load it.")
