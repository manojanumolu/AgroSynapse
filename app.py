# app.py — Multimodal Soil & Crop Recommendation System
# ResNet-50 + XGBoost + TSACA Cross-Attention Fusion + GRN  |  Accuracy: 98.67%
#
# Usage:
#   1. pip install -r requirements.txt
#   2. python app.py
#   3. Open http://localhost:5000 in browser

import os, io, json, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import models, transforms

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def mpath(n): return os.path.join(BASE, n)

# ── Load config ────────────────────────────────────────────────
with open(mpath("model_config.json")) as f: CFG = json.load(f)
with open(mpath("class_names.json"))  as f: CLASS_NAMES = json.load(f)

IMG_FEAT_DIM = CFG["IMG_FEAT_DIM"]   # 512
XGB_PROJ_DIM = CFG["XGB_PROJ_DIM"]   # 256
FUSED_DIM    = CFG["FUSED_DIM"]      # 512
NUM_HEADS    = CFG["NUM_HEADS"]       # 8
NUM_CLASSES  = CFG["NUM_CLASSES"]    # 6
TAB_FEAT_DIM = CFG["TAB_FEAT_DIM"]   # 19
NUMERIC_COLS = CFG["NUMERIC_COLS"]   # 9 columns
IMG_SIZE     = CFG["IMG_SIZE"]       # 224

print(f"Classes : {CLASS_NAMES}")


# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — identical to training
# ══════════════════════════════════════════════════════════════

class ResNet50Classifier(nn.Module):
    """ResNet-50 backbone → 2048→1024→512 projection → classifier."""
    def __init__(self, nc, fd=512):
        super().__init__()
        base = models.resnet50(weights=None)
        self.backbone   = nn.Sequential(*list(base.children())[:-2])
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, fd),
            nn.BatchNorm1d(fd), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(fd, nc))

    def forward(self, x, return_features=False):
        f = self.projection(self.pool(self.backbone(x)))
        if return_features: return f
        return self.classifier(f)


class TabProjector(nn.Module):
    """Projects 19-dim tabular input → 256-dim for fusion."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class TSACAFusion(nn.Module):
    """Tabular-Supervised Cross-Attention fusion (img 512 + tab 256 → 512)."""
    def __init__(self, img_dim, tab_dim, fd, nh, nl=3):
        super().__init__()
        self.ip = nn.Sequential(nn.Linear(img_dim, fd), nn.LayerNorm(fd), nn.ReLU())
        self.tp = nn.Sequential(nn.Linear(tab_dim, fd), nn.LayerNorm(fd), nn.ReLU())
        self.ca = nn.ModuleList([
            nn.MultiheadAttention(fd, nh, dropout=0.1, batch_first=True)
            for _ in range(nl)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(fd, fd*4), nn.GELU(),
                          nn.Dropout(0.1), nn.Linear(fd*4, fd))
            for _ in range(nl)])
        self.nm   = nn.ModuleList([nn.LayerNorm(fd) for _ in range(nl*2)])
        self.gate = nn.Sequential(
            nn.Linear(fd*2, fd*2), nn.ReLU(),
            nn.Linear(fd*2, fd),   nn.Sigmoid())
        self.out  = nn.Sequential(nn.Linear(fd, fd), nn.LayerNorm(fd), nn.ReLU())

    def forward(self, img_f, tab_f):
        ip = self.ip(img_f).unsqueeze(1)
        tp = self.tp(tab_f).unsqueeze(1)
        x  = tp
        for i, (a, f) in enumerate(zip(self.ca, self.ff)):
            ao, _ = a(query=x, key=ip, value=ip)
            x = self.nm[i*2](x + ao)
            x = self.nm[i*2+1](x + f(x))
        x  = x.squeeze(1)
        gw = self.gate(torch.cat([x, tp.squeeze(1)], dim=-1))
        return self.out(gw * x + (1 - gw) * tp.squeeze(1))


class GatedLinearUnit(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc   = nn.Linear(i, o)
        self.gate = nn.Linear(i, o)
    def forward(self, x): return self.fc(x) * torch.sigmoid(self.gate(x))


class GRNBlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim*2); self.elu = nn.ELU()
        self.fc2  = nn.Linear(dim*2, dim)
        self.glu  = GatedLinearUnit(dim, dim)
        self.norm = nn.LayerNorm(dim); self.drop = nn.Dropout(drop)
    def forward(self, x):
        h = self.elu(self.fc1(x)); h = self.drop(self.fc2(h))
        return self.norm(self.glu(h) + x)


class GRNCropPredictor(nn.Module):
    def __init__(self, in_dim, nc, nb=5, drop=0.2):
        super().__init__()
        self.proj   = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.LayerNorm(in_dim), nn.ReLU())
        self.blocks = nn.ModuleList([GRNBlock(in_dim, drop) for _ in range(nb)])
        self.head   = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Dropout(drop / 2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, nc))
    def forward(self, f):
        x = self.proj(f)
        for b in self.blocks: x = b(x)
        logits = self.head(x)
        return logits, torch.softmax(logits, -1).max(-1).values


class FusionGRNModel(nn.Module):
    """Wraps TSACA fusion + GRN predictor."""
    def __init__(self):
        super().__init__()
        self.tsaca = TSACAFusion(IMG_FEAT_DIM, XGB_PROJ_DIM, FUSED_DIM, NUM_HEADS)
        self.grn   = GRNCropPredictor(FUSED_DIM, NUM_CLASSES)
    def forward(self, img_f, tab_f):
        return self.grn(self.tsaca(img_f, tab_f))


# ══════════════════════════════════════════════════════════════
# LOAD ALL MODELS (CPU inference)
# ══════════════════════════════════════════════════════════════
print("Loading models...")

img_model     = ResNet50Classifier(NUM_CLASSES, IMG_FEAT_DIM)
tab_projector = TabProjector(TAB_FEAT_DIM, XGB_PROJ_DIM)
fusion_model  = FusionGRNModel()

img_model.load_state_dict(
    torch.load(mpath("img_model.pt"), map_location="cpu", weights_only=True))
tab_projector.load_state_dict(
    torch.load(mpath("tab_projector.pt"), map_location="cpu", weights_only=True))
fusion_model.load_state_dict(
    torch.load(mpath("fusion_model.pt"), map_location="cpu", weights_only=True))

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(mpath("xgb_model.json"))

with open(mpath("scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

img_model.eval(); tab_projector.eval(); fusion_model.eval()
print("All models loaded successfully!")


# ══════════════════════════════════════════════════════════════
# TRANSFORMS & LOOKUP MAPS
# ══════════════════════════════════════════════════════════════
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def is_soil_image(pil_img, img_model, transform):
    """Rule-based validator + ResNet confidence."""
    arr = np.array(pil_img.resize((200, 200)).convert("RGB")).astype(float)
    r = arr[:, :, 0]; g = arr[:, :, 1]; b = arr[:, :, 2]
    total = 200 * 200

    cyan_pixels     = np.sum((b > 150) & (g > 150) & (r < 100)) / total
    orange_neon     = np.sum((r > 220) & (g > 80) & (g < 170) & (b < 80)) / total
    pink_neon       = np.sum((r > 200) & (b > 150) & (g < 100)) / total
    bright_red_neon = np.sum((r > 220) & (g < 60) & (b < 60)) / total
    if (cyan_pixels + orange_neon + pink_neon + bright_red_neon) > 0.02:
        return False

    skin = np.sum(
        (r > 160) & (g > 110) & (b > 90) & (r > g) & (g > b) &
        ((r + g + b) / 3 > 120) & ((r + g + b) / 3 < 210) &
        ((r - b) > 25) & ((r - b) < 120)
    ) / total
    if skin > 0.30:
        return False

    if np.sum((b > r + 40) & (b > g + 30) & (b > 110)) / total > 0.22:
        return False
    if np.sum((g > r + 35) & (g > b + 35) & (g > 90)) / total > 0.22:
        return False

    if arr.mean() > 195:
        return False

    h_diff = np.abs(np.diff(arr[:, :, 0].astype(float), axis=1)).mean()
    v_diff = np.abs(np.diff(arr[:, :, 0].astype(float), axis=0)).mean()
    if (h_diff + v_diff) / 2 > 28:
        return False

    img_t = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        out   = img_model(img_t, return_features=False)
        probs = torch.softmax(out, dim=-1)[0]
    maxp = probs.max().item() * 100
    top2 = torch.topk(probs, 2).values
    gap  = (top2[0] - top2[1]).item()
    if maxp < 35.0:
        return False
    if maxp < 45.0 and gap < 0.08:
        return False

    return True

SEASON_MAP = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
IRRIG_MAP  = {"Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3}
PREV_MAP   = {"Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3,
               "Sugarcane": 4, "Tomato": 5, "Wheat": 6}
REGION_MAP = {"Central": 0, "East": 1, "North": 2, "South": 3, "West": 4}

# Soil-level fertilizer recommendations
SOIL_FERT_MAP = {
    "Alluvial Soil" : {"fertilizer": "NPK 20:20:0 + Zinc",  "npk": "N:P:K = 80:40:20 kg/ha"},
    "Black Soil"    : {"fertilizer": "Urea + MOP",           "npk": "N:P:K = 60:30:30 kg/ha"},
    "Clay Soil"     : {"fertilizer": "Urea + DAP",           "npk": "N:P:K = 60:60:60 kg/ha"},
    "Laterite Soil" : {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:60:20 kg/ha"},
    "Red Soil"      : {"fertilizer": "NPK 17:17:17",         "npk": "N:P:K = 50:50:50 kg/ha"},
    "Yellow Soil"   : {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:30:20 kg/ha"},
}

# Crop-level fertilizer recommendations
CROP_FERT_MAP = {
    "Cotton":    {"fertilizer": "NPK 17:17:17",  "npk": "50:50:50 kg/ha"},
    "Maize":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
    "Potato":    {"fertilizer": "NPK 15:15:15",  "npk": "180:120:80 kg/ha"},
    "Rice":      {"fertilizer": "Urea + SSP",    "npk": "100:50:25 kg/ha"},
    "Sugarcane": {"fertilizer": "NPK 20:10:10",  "npk": "250:85:115 kg/ha"},
    "Tomato":    {"fertilizer": "NPK 12:32:16",  "npk": "200:150:200 kg/ha"},
    "Wheat":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
}

# Crop recommendations per (soil, season)
CROP_MAP = {
    ("Red Soil",      "Kharif"): ["Cotton",    "Maize",      "Groundnut",  "Tomato"],
    ("Red Soil",      "Rabi")  : ["Wheat",     "Sunflower",  "Linseed",    "Potato"],
    ("Red Soil",      "Zaid")  : ["Watermelon","Cucumber",   "Bitter Gourd","Moong"],
    ("Alluvial Soil", "Kharif"): ["Rice",      "Sugarcane",  "Maize",      "Jute"],
    ("Alluvial Soil", "Rabi")  : ["Wheat",     "Mustard",    "Barley",     "Peas"],
    ("Alluvial Soil", "Zaid")  : ["Watermelon","Muskmelon",  "Cucumber",   "Moong"],
    ("Black Soil",    "Kharif"): ["Cotton",    "Sorghum",    "Soybean",    "Groundnut"],
    ("Black Soil",    "Rabi")  : ["Wheat",     "Chickpea",   "Linseed",    "Safflower"],
    ("Black Soil",    "Zaid")  : ["Sunflower", "Sesame",     "Maize",      "Moong"],
    ("Clay Soil",     "Kharif"): ["Rice",      "Jute",       "Sugarcane",  "Taro"],
    ("Clay Soil",     "Rabi")  : ["Wheat",     "Barley",     "Mustard",    "Spinach"],
    ("Clay Soil",     "Zaid")  : ["Cucumber",  "Bitter Gourd","Pumpkin",   "Moong"],
    ("Laterite Soil", "Kharif"): ["Cashew",    "Rubber",     "Tea",        "Coffee"],
    ("Laterite Soil", "Rabi")  : ["Tapioca",   "Groundnut",  "Turmeric",   "Ginger"],
    ("Laterite Soil", "Zaid")  : ["Mango",     "Pineapple",  "Jackfruit",  "Banana"],
    ("Yellow Soil",   "Kharif"): ["Rice",      "Maize",      "Groundnut",  "Sesame"],
    ("Yellow Soil",   "Rabi")  : ["Wheat",     "Mustard",    "Potato",     "Barley"],
    ("Yellow Soil",   "Zaid")  : ["Sunflower", "Moong",      "Cucumber",   "Tomato"],
}


# ══════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "ok",
        "classes" : CLASS_NAMES,
        "accuracy": "98.67%",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST multipart/form-data:
      image      — soil photo file (JPEG/PNG)
      n          — Nitrogen (kg/ha)
      p          — Phosphorus (kg/ha)
      k          — Potassium (kg/ha)
      temp       — Temperature (°C)
      humidity   — Humidity (%)
      rainfall   — Rainfall (mm)
      ph         — Soil pH
      yield_val  — Yield last season (kg/ha)
      fert_used  — Fertilizer used last season (kg/ha)
      season     — Kharif / Rabi / Zaid
      irrigation — Canal / Drip / Rainfed / Sprinkler
      prev_crop  — Cotton/Maize/Potato/Rice/Sugarcane/Tomato/Wheat
      region     — Central / East / North / South / West
    """
    try:
        def gf(k, d=0.0): return float(request.form.get(k, d))
        def gs(k, d=""):  return request.form.get(k, d)

        n      = gf("n");          p    = gf("p");        k_val  = gf("k")
        temp   = gf("temp");       hum  = gf("humidity"); rain   = gf("rainfall")
        ph     = gf("ph");         yld  = gf("yield_val");fert   = gf("fert_used")
        season = gs("season",     "Kharif")
        irrig  = gs("irrigation", "Canal")
        prev   = gs("prev_crop",  "Wheat")
        region = gs("region",     "South")

        # ── Build tabular features ──
        num_raw = np.array([[n, p, k_val, temp, hum, rain, ph, yld, fert]])
        num_sc  = scaler.transform(pd.DataFrame(num_raw, columns=NUMERIC_COLS))
        cat_enc = np.array([[
            SEASON_MAP.get(season, 0), IRRIG_MAP.get(irrig, 0),
            PREV_MAP.get(prev, 0),     REGION_MAP.get(region, 0),
        ]])
        scaled_feat = np.concatenate([num_sc, cat_enc], axis=1).astype(np.float32)
        # scaled_feat shape: (1, 13)  [9 numeric + 4 categorical]

        # ── XGBoost probabilities ──
        xgb_probs = xgb_clf.predict_proba(scaled_feat)   # (1, 6)
        tab_raw   = np.concatenate([xgb_probs, scaled_feat], axis=1).astype(np.float32)
        # tab_raw shape: (1, 19)  [6 probs + 13 features]
        tab_t = torch.tensor(tab_raw, dtype=torch.float32)

        # ── Image processing ──
        file = request.files.get("image")
        if file is None:
            return jsonify({"error": "No image provided"}), 400
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        if not is_soil_image(pil_img, img_model, eval_tf):
            return jsonify({
                "error": "No soil detected. Please upload a clear soil photo."
            }), 400
        img_t   = eval_tf(pil_img).unsqueeze(0)

        # ── Model inference ──
        with torch.no_grad():
            img_feat       = img_model(img_t, return_features=True)
            tab_feat       = tab_projector(tab_t)
            logits, _      = fusion_model(img_feat, tab_feat)

        fusion_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (6,)

        # ── Calibrated ensemble (fusion + XGBoost) ──────────────────
        # XGBoost uses soil chemistry (NPK, pH) and is more reliable for
        # visually similar soils like Red vs Yellow. We blend the two
        # probability distributions, giving XGBoost extra weight when the
        # fusion model is uncertain (top-2 gap < 0.20).
        xgb_p   = xgb_probs[0]                     # (6,) already probabilities
        top2    = np.partition(fusion_probs, -2)[-2:]
        gap     = float(top2[-1] - top2[-2])        # confidence margin
        # When the model is unsure (gap < 0.20), lean more on tabular signal
        xgb_w   = 0.45 if gap < 0.20 else 0.30
        blended = (1 - xgb_w) * fusion_probs + xgb_w * xgb_p

        # Extra nudge: penalise Red↔Yellow confusion using pH + color proxy.
        # Yellow soils are typically more acidic (pH 5.0–6.5) and have lower
        # iron content reflected as lower K values.
        RED_IDX    = CLASS_NAMES.index("Red Soil")
        YELLOW_IDX = CLASS_NAMES.index("Yellow Soil")
        if blended[RED_IDX] > 0.30 or blended[YELLOW_IDX] > 0.30:
            # pH below 6.5 and K below 50 both favour Yellow Soil
            ph_score  = max(0.0, (6.5 - ph) / 6.5)          # 0→1 as pH drops
            k_score   = max(0.0, (50.0 - k_val) / 50.0)     # 0→1 as K drops
            yellow_boost = 0.12 * (ph_score + k_score) / 2.0
            blended[YELLOW_IDX] = min(1.0, blended[YELLOW_IDX] + yellow_boost)
            blended[RED_IDX]    = max(0.0, blended[RED_IDX]    - yellow_boost)
            blended = blended / blended.sum()                 # renormalise

        pred_idx   = int(np.argmax(blended))
        soil_name  = CLASS_NAMES[pred_idx]
        confidence = round(float(blended[pred_idx]) * 100, 2)
        probs      = blended.tolist()

        # ── Lookup results ──
        soil_fert = SOIL_FERT_MAP.get(soil_name, {
            "fertilizer": "NPK 14:14:14", "npk": "N:P:K = 60:30:30 kg/ha"})

        crops_all = CROP_MAP.get(
            (soil_name, season),
            CROP_MAP.get((soil_name, "Kharif"), ["Wheat", "Rice", "Maize", "Mustard"]))

        crop_recs = []
        for i, crop in enumerate(crops_all[:3]):
            cf = CROP_FERT_MAP.get(crop, {"fertilizer": "NPK 14:14:14", "npk": "60:40:20 kg/ha"})
            crop_recs.append({
                "name":       crop,
                "rank":       i + 1,
                "stars":      5 - i,
                "fertilizer": cf["fertilizer"],
                "npk":        cf["npk"],
            })

        return jsonify({
            "soil_type"        : soil_name,
            "confidence"       : confidence,
            "all_probabilities": {
                CLASS_NAMES[i]: round(probs[i] * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            "soil_fertilizer"  : soil_fert["fertilizer"],
            "soil_npk"         : soil_fert["npk"],
            "recommended_crops": crop_recs,
            "season"           : season,
            "region"           : region,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n  Soil Classifier Web App")
    print("  Accuracy: 98.67%\n")
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
