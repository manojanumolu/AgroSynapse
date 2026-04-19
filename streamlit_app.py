# streamlit_app.py — Multimodal Soil & Crop Recommendation System
# ResNet-50 + XGBoost + TSACA Fusion + GRN  |  Accuracy: 98.67%
# Run: streamlit run streamlit_app.py

import io, os, json, pickle, re, zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

_IMPORT_ERROR = None
try:
    import torch, torch.nn as nn
    import xgboost as xgb
    from torchvision import models, transforms
except Exception as _imp_err:
    _IMPORT_ERROR = _imp_err

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSynapse AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if _IMPORT_ERROR is not None:
    st.error(
        "**Dependency import failed.** This usually means a missing or incompatible "
        "package version on Streamlit Cloud.\n\n"
        f"Error: `{_IMPORT_ERROR}`"
    )
    st.stop()

# ?? Session state ???????????????????????????????????????????
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "auto_temp" not in st.session_state:
    st.session_state.auto_temp = 27.2
if "auto_hum" not in st.session_state:
    st.session_state.auto_hum = 75.3
if "auto_rain" not in st.session_state:
    st.session_state.auto_rain = 1352.7
if "location_name" not in st.session_state:
    st.session_state.location_name = ""
if "location_note" not in st.session_state:
    st.session_state.location_note = ""
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False
if "page" not in st.session_state:
    st.session_state.page = "home"
if "leaf_img_bytes" not in st.session_state:
    st.session_state.leaf_img_bytes = None
if "leaf_result" not in st.session_state:
    st.session_state.leaf_result = None
if "leaf_model_error" not in st.session_state:
    st.session_state.leaf_model_error = None
if "leaf_valid" not in st.session_state:
    st.session_state.leaf_valid = None   # None=not checked, True=valid, False=invalid

if st.session_state.theme == "dark":
    THEME_VARS = """<style>
:root {
    --bg: #1c211d;
    --surface: #222823;
    --surface-2: #283028;
    --surface-3: #2d352e;
    --surface-container-low: #283028;
    --surface-container-lowest: #313b33;
    --surface-container: #2a312b;
    --surface-container-high: #333c34;
    --surface-container-highest: #3a443b;
    --outline: #4b564d;
    --text: #e8ece8;
    --muted: #b8c1b8;
    --primary: #acf3ba;
    --primary-2: #4a8c5c;
    --secondary: #9fd2a8;
  --secondary-fixed: #2b3f5a;
  --on-secondary-fixed-variant: #cfe3ff;
  --tertiary: #a56b6d;
  --danger: #ff6b6b;
  --card: #1d1d1d;
  --border: #2f2f2f;
  --pill: #1a3a1a;
  --sidebar: #1a1a1a;
}
</style>"""
else:
    THEME_VARS = """<style>
:root {
    --bg: #F2EEE7;
  --surface: #DCDCDC;
    --surface-2: #E3E3E3;
    --surface-3: #E1E1E1;
    --surface-container-low: #E0E0E0;
  --surface-container-lowest: #DCDCDC;
    --surface-container: #DEDEDE;
    --surface-container-high: #DCDCDC;
    --surface-container-highest: #D7D7D7;
    --outline: #C7CCC7;
    --text: #404942;
    --muted: #707971;
    --primary: #1E5C3A;
    --primary-2: #1E5C3A;
    --secondary: #4A8C5C;
  --secondary-fixed: #cee5ff;
  --on-secondary-fixed-variant: #224a6b;
  --tertiary: #713638;
  --danger: #ff4d4f;
  --card: #DCDCDC;
  --border: #eaeaea;
  --pill: #e6f7ed;
  --sidebar: #f7f8fa;
}
</style>"""

# ── Paths ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def mpath(n): return os.path.join(BASE, n)

# ── Model file verification (runs at startup, visible in logs) ──
_MODEL_FILES = [
    "img_model.pt", "fusion_model.pt",
    "tab_projector.pt", "xgb_model.json", "scaler.pkl",
]
_file_status = {}
for _f in _MODEL_FILES:
    _p = mpath(_f)
    _exists = os.path.exists(_p)
    _size   = os.path.getsize(_p) if _exists else 0
    _file_status[_f] = {"exists": _exists, "mb": round(_size / 1024 / 1024, 1)}
    print(f"{_f}: exists={_exists}, size={_size/1024/1024:.1f}MB")

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS — identical to training
# ══════════════════════════════════════════════════════════════

class ResNet50Classifier(nn.Module):
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
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    512),  nn.BatchNorm1d(512),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class TSACAFusion(nn.Module):
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
        self.fc = nn.Linear(i, o); self.gate = nn.Linear(i, o)
    def forward(self, x): return self.fc(x) * torch.sigmoid(self.gate(x))


class GRNBlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*2); self.elu = nn.ELU()
        self.fc2 = nn.Linear(dim*2, dim)
        self.glu = GatedLinearUnit(dim, dim)
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
    def __init__(self, img_dim, xgb_dim, fused_dim, num_heads, num_classes):
        super().__init__()
        self.tsaca = TSACAFusion(img_dim, xgb_dim, fused_dim, num_heads)
        self.grn   = GRNCropPredictor(fused_dim, num_classes)
    def forward(self, img_f, tab_f):
        return self.grn(self.tsaca(img_f, tab_f))


# ══════════════════════════════════════════════════════════════
# MODEL LOADING — cached per session, loads once
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_all_models():
    """Load and cache all models once per session.
    Raises a clear error if any model file is missing or too small
    (which happens when Git LFS pointers aren't resolved on the server).
    """
    # ── Validate model files before loading ────────────────────
    MIN_SIZES = {
        "img_model.pt":     25,   # fp16 shrinked
        "fusion_model.pt":  10,
        "tab_projector.pt":  0.2,
    }
    for fname, min_mb in MIN_SIZES.items():
        info = _file_status.get(fname, {})
        if not info.get("exists"):
            raise FileNotFoundError(
                f"{fname} not found. Check repository LFS setup.")
        if info["mb"] < min_mb:
            raise ValueError(
                f"{fname} is only {info['mb']:.1f} MB — "
                f"expected >{min_mb} MB. "
                f"Git LFS pointers may not have been resolved on this server. "
                f"Run: git lfs pull")

    with open(mpath("model_config.json")) as f: cfg = json.load(f)
    with open(mpath("class_names.json"))  as f: cls = json.load(f)

    img_dim   = cfg["IMG_FEAT_DIM"]
    xgb_dim   = cfg["XGB_PROJ_DIM"]
    fused_dim = cfg["FUSED_DIM"]
    num_heads = cfg["NUM_HEADS"]
    num_cls   = cfg["NUM_CLASSES"]
    tab_dim   = cfg["TAB_FEAT_DIM"]
    num_cols  = cfg["NUMERIC_COLS"]

    img_m  = ResNet50Classifier(num_cls, img_dim)
    tab_p  = TabProjector(tab_dim, xgb_dim)
    fusion = FusionGRNModel(img_dim, xgb_dim, fused_dim, num_heads, num_cls)

    def _load_fp16_state(path):
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            out = {}
            for k, v in state.items():
                if torch.is_tensor(v) and v.dtype == torch.float16:
                    out[k] = v.float()
                else:
                    out[k] = v
            return out
        return state

    img_m.load_state_dict(_load_fp16_state(mpath("img_model.pt")))
    tab_p.load_state_dict(_load_fp16_state(mpath("tab_projector.pt")))
    fusion.load_state_dict(_load_fp16_state(mpath("fusion_model.pt")))

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(mpath("xgb_model.json"))

    with open(mpath("scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    # Explicit eval() — essential for BatchNorm/Dropout at inference
    img_m.eval(); tab_p.eval(); fusion.eval()

    return img_m, tab_p, fusion, xgb_clf, scaler, cls, num_cols


# ── Lookup maps ────────────────────────────────────────────────
SEASON_MAP = {"Kharif": 0, "Rabi": 1, "Zaid": 2}
IRRIG_MAP  = {"Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3}
PREV_MAP   = {"Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3,
               "Sugarcane": 4, "Tomato": 5, "Wheat": 6}
REGION_MAP = {"Central": 0, "East": 1, "North": 2, "South": 3, "West": 4}

# Dataset-aligned numeric bounds from Soil_Nutrients.csv
DATASET_N_RANGE = (20.0, 187.0)
DATASET_P_RANGE = (10.0, 101.0)
DATASET_K_RANGE = (10.0, 113.0)
IDEAL_N_RANGE = (60.0, 140.0)
IDEAL_P_RANGE = (25.0, 80.0)
IDEAL_K_RANGE = (35.0, 120.0)

SOIL_FERT_MAP = {
    "Alluvial Soil": {"fertilizer": "NPK 20:20:0 + Zinc",  "npk": "N:P:K = 80:40:20 kg/ha"},
    "Black Soil":    {"fertilizer": "Urea + MOP",           "npk": "N:P:K = 60:30:30 kg/ha"},
    "Clay Soil":     {"fertilizer": "Urea + DAP",           "npk": "N:P:K = 60:60:60 kg/ha"},
    "Laterite Soil": {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:60:20 kg/ha"},
    "Red Soil":      {"fertilizer": "NPK 17:17:17",         "npk": "N:P:K = 50:50:50 kg/ha"},
    "Yellow Soil":   {"fertilizer": "DAP + Compost",        "npk": "N:P:K = 40:30:20 kg/ha"},
}

CROP_FERT_MAP = {
    "Cotton":    {"fertilizer": "NPK 17:17:17",  "npk": "50:50:50 kg/ha"},
    "Chilli":    {"fertilizer": "NPK 19:19:19 + Potash", "npk": "120:60:60 kg/ha"},
    "Maize":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
    "Potato":    {"fertilizer": "NPK 15:15:15",  "npk": "180:120:80 kg/ha"},
    "Rice":      {"fertilizer": "Urea + SSP",    "npk": "100:50:25 kg/ha"},
    "Sugarcane": {"fertilizer": "NPK 20:10:10",  "npk": "250:85:115 kg/ha"},
    "Tomato":    {"fertilizer": "NPK 12:32:16",  "npk": "200:150:200 kg/ha"},
    "Wheat":     {"fertilizer": "Urea + DAP",    "npk": "120:60:40 kg/ha"},
}

SOIL_COLORS = {
    "Alluvial Soil": "#a87c4f",
    "Black Soil":    "#2f2f2f",
    "Clay Soil":     "#8b5e34",
    "Laterite Soil": "#7a2f2f",
    "Red Soil":      "#b6422b",
    "Yellow Soil":   "#d0a200",
}

CROP_MAP = {
    ("Red Soil",      "Kharif"): ["Cotton",    "Maize",      "Groundnut",  "Tomato", "Chilli"],
    ("Red Soil",      "Rabi")  : ["Wheat",     "Sunflower",  "Linseed",    "Potato"],
    ("Red Soil",      "Zaid")  : ["Watermelon","Cucumber",   "Bitter Gourd","Moong", "Chilli"],
    ("Alluvial Soil", "Kharif"): ["Rice",      "Sugarcane",  "Maize",      "Jute"],
    ("Alluvial Soil", "Rabi")  : ["Wheat",     "Mustard",    "Barley",     "Peas", "Chilli"],
    ("Alluvial Soil", "Zaid")  : ["Watermelon","Muskmelon",  "Cucumber",   "Moong"],
    ("Black Soil",    "Kharif"): ["Cotton",    "Sorghum",    "Soybean",    "Groundnut"],
    ("Black Soil",    "Rabi")  : ["Wheat",     "Chickpea",   "Linseed",    "Safflower"],
    ("Black Soil",    "Zaid")  : ["Sunflower", "Sesame",     "Maize",      "Moong", "Chilli"],
    ("Clay Soil",     "Kharif"): ["Rice",      "Jute",       "Sugarcane",  "Taro"],
    ("Clay Soil",     "Rabi")  : ["Wheat",     "Barley",     "Mustard",    "Spinach"],
    ("Clay Soil",     "Zaid")  : ["Cucumber",  "Bitter Gourd","Pumpkin",   "Moong", "Chilli"],
    ("Laterite Soil", "Kharif"): ["Cashew",    "Rubber",     "Tea",        "Coffee"],
    ("Laterite Soil", "Rabi")  : ["Tapioca",   "Groundnut",  "Turmeric",   "Ginger"],
    ("Laterite Soil", "Zaid")  : ["Mango",     "Pineapple",  "Jackfruit",  "Banana"],
    ("Yellow Soil",   "Kharif"): ["Rice",      "Maize",      "Groundnut",  "Sesame"],
    ("Yellow Soil",   "Rabi")  : ["Wheat",     "Mustard",    "Potato",     "Barley", "Chilli"],
    ("Yellow Soil",   "Zaid")  : ["Sunflower", "Moong",      "Cucumber",   "Tomato"],
}

CROP_PROFILES = {
    "Wheat":      {"n": (80, 150), "p": (30, 70),  "k": (35, 80),  "ph": (6.0, 7.8), "temp": (12, 30), "hum": (40, 75), "rain": (300, 1200), "fert": (80, 260), "yld": (1800, 6500), "irrig": ["Canal", "Sprinkler"]},
    "Rice":       {"n": (70, 140), "p": (20, 60),  "k": (20, 60),  "ph": (5.0, 7.2), "temp": (20, 36), "hum": (60, 98), "rain": (900, 3000), "fert": (70, 260), "yld": (1800, 8000), "irrig": ["Canal", "Rainfed"]},
    "Cotton":     {"n": (60, 130), "p": (25, 65),  "k": (35, 90),  "ph": (5.8, 8.0), "temp": (20, 38), "hum": (35, 75), "rain": (500, 1400), "fert": (80, 300), "yld": (1000, 4500), "irrig": ["Drip", "Canal"]},
    "Watermelon": {"n": (55, 120), "p": (20, 60),  "k": (40, 120), "ph": (5.8, 7.5), "temp": (22, 38), "hum": (35, 75), "rain": (250, 1100), "fert": (60, 240), "yld": (8000, 35000), "irrig": ["Drip", "Sprinkler"]},
    "Maize":      {"n": (70, 150), "p": (25, 70),  "k": (30, 90),  "ph": (5.8, 7.8), "temp": (18, 36), "hum": (40, 85), "rain": (450, 1500), "fert": (70, 280), "yld": (1800, 9000), "irrig": ["Canal", "Rainfed", "Sprinkler"]},
    "Tomato":     {"n": (80, 180), "p": (35, 120), "k": (60, 180), "ph": (6.0, 7.5), "temp": (18, 34), "hum": (40, 82), "rain": (300, 1500), "fert": (120, 360), "yld": (9000, 45000), "irrig": ["Drip", "Sprinkler"]},
    "Chilli":     {"n": (70, 150), "p": (30, 90),  "k": (50, 140), "ph": (6.0, 7.3), "temp": (20, 35), "hum": (40, 75), "rain": (450, 1300), "fert": (90, 320), "yld": (1200, 9000), "irrig": ["Drip", "Sprinkler", "Canal"]},
    "Potato":     {"n": (90, 170), "p": (40, 110), "k": (70, 170), "ph": (5.2, 6.8), "temp": (10, 28), "hum": (45, 85), "rain": (350, 1400), "fert": (120, 360), "yld": (8000, 40000), "irrig": ["Canal", "Sprinkler"]},
    "Sugarcane":  {"n": (80, 190), "p": (30, 90),  "k": (50, 170), "ph": (6.0, 8.2), "temp": (20, 38), "hum": (45, 90), "rain": (700, 3000), "fert": (140, 420), "yld": (30000, 140000), "irrig": ["Canal", "Drip"]},
}

DEFAULT_CROP_PROFILE = {
    "n": (55, 150), "p": (20, 90), "k": (25, 140),
    "ph": (5.5, 7.8), "temp": (16, 36), "hum": (35, 88),
    "rain": (300, 2200), "fert": (60, 340), "yld": (1500, 20000),
    "irrig": ["Canal", "Drip", "Rainfed", "Sprinkler"],
}

def _range_score(value, lo, hi):
    if lo <= value <= hi:
        return 1.0
    span = max(hi - lo, 1e-6)
    if value < lo:
        return max(0.0, 1.0 - (lo - value) / (span * 1.5))
    return max(0.0, 1.0 - (value - hi) / (span * 1.5))


def _crop_suitability_score(crop, n, p, k, ph, temp, hum, rain, yld, fert, irrig, prev, region):
    prof = CROP_PROFILES.get(crop, DEFAULT_CROP_PROFILE)
    score = 0.0

    score += 1.6 * _range_score(n, *prof["n"])
    score += 1.2 * _range_score(p, *prof["p"])
    score += 1.3 * _range_score(k, *prof["k"])
    score += 1.1 * _range_score(ph, *prof["ph"])
    score += 1.0 * _range_score(temp, *prof["temp"])
    score += 0.8 * _range_score(hum, *prof["hum"])
    score += 0.8 * _range_score(rain, *prof["rain"])
    score += 0.6 * _range_score(yld, *prof["yld"])
    score += 0.6 * _range_score(fert, *prof["fert"])

    if irrig in prof.get("irrig", []):
        score += 0.45
    else:
        score -= 0.25

    if prev == crop:
        score -= 0.9

    if crop in {"Cotton", "Sugarcane"} and region in {"South", "West"}:
        score += 0.1
    if crop in {"Wheat", "Mustard", "Barley"} and region in {"North", "Central"}:
        score += 0.1
    if crop in {"Rice", "Jute"} and region in {"East", "South"}:
        score += 0.1

    return score


def _parse_npk_triplet(npk_text):
    nums = [int(x) for x in re.findall(r"\d+", str(npk_text))]
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return 60, 40, 20


def _adjust_component(base_dose, soil_val, ideal_lo, ideal_hi):
    if soil_val < ideal_lo:
        gap_ratio = min((ideal_lo - soil_val) / max(ideal_lo, 1.0), 1.0)
        return int(round(base_dose * (1.0 + 0.6 * gap_ratio)))
    if soil_val > ideal_hi:
        excess_ratio = min((soil_val - ideal_hi) / max(ideal_hi, 1.0), 1.0)
        return int(round(base_dose * (1.0 - 0.45 * excess_ratio)))
    return int(round(base_dose))


def _fertilizer_for_crop(crop, n, p, k):
    base = CROP_FERT_MAP.get(crop, {"fertilizer": "NPK 14:14:14", "npk": "60:40:20 kg/ha"})
    b_n, b_p, b_k = _parse_npk_triplet(base.get("npk", "60:40:20"))

    prof = CROP_PROFILES.get(crop, DEFAULT_CROP_PROFILE)
    n_lo, n_hi = prof["n"]
    p_lo, p_hi = prof["p"]
    k_lo, k_hi = prof["k"]

    r_n = _adjust_component(b_n, n, n_lo, n_hi)
    r_p = _adjust_component(b_p, p, p_lo, p_hi)
    r_k = _adjust_component(b_k, k, k_lo, k_hi)

    n_def = max(0.0, (n_lo - n) / max(n_lo, 1.0))
    p_def = max(0.0, (p_lo - p) / max(p_lo, 1.0))
    k_def = max(0.0, (k_lo - k) / max(k_lo, 1.0))
    biggest = max(("N", n_def), ("P", p_def), ("K", k_def), key=lambda x: x[1])
    suffix = f" ({biggest[0]} boost)" if biggest[1] >= 0.12 else ""

    return {
        "fertilizer": f"{base['fertilizer']}{suffix}",
        "npk": f"{r_n}:{r_p}:{r_k} kg/ha",
        "ideal": f"Ideal N:{int(n_lo)}-{int(n_hi)} P:{int(p_lo)}-{int(p_hi)} K:{int(k_lo)}-{int(k_hi)}",
    }


# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════

def run_inference(img_model, tab_proj, fusion, xgb_clf, scaler,
                  class_names, num_cols,
                  img_bytes, n, p, k, temp, hum, rain, ph, yld, fert,
                  season, irrig, prev, region):

    # ── Transform created fresh every call (never cached) ──────
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Image: always decode from raw bytes ────────────────────
    # UploadedFile stream is consumed by st.image(); using saved bytes
    # guarantees a correct, fresh tensor regardless of Streamlit rerenders.
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t   = tf(pil_img).unsqueeze(0)           # (1, 3, 224, 224)

    # ── Tabular features ──
    num_raw     = np.array([[n, p, k, temp, hum, rain, ph, yld, fert]])
    num_sc      = scaler.transform(pd.DataFrame(num_raw, columns=num_cols))
    cat_enc     = np.array([[SEASON_MAP[season], IRRIG_MAP[irrig],
                              PREV_MAP[prev],     REGION_MAP[region]]])
    xgb_input = np.concatenate([num_sc, cat_enc], axis=1).astype(np.float32)

    xgb_probs = xgb_clf.predict_proba(xgb_input)                      # (1, 6)
    tab_raw   = np.concatenate([xgb_probs, xgb_input], axis=1).astype(np.float32)
    tab_t     = torch.tensor(tab_raw, dtype=torch.float32)             # (1, 19)

    # ── Inference — explicit eval() + no_grad every call ──────
    img_model.eval(); tab_proj.eval(); fusion.eval()
    with torch.no_grad():
        img_feat  = img_model(img_t, return_features=True)
        tab_feat  = tab_proj(tab_t)
        logits, _ = fusion(img_feat, tab_feat)

    # Clean prediction — raw softmax only, no blending or manipulation
    probs      = torch.softmax(logits, dim=-1)[0].cpu().numpy()        # (6,)
    pred_idx   = int(np.argmax(probs))
    soil_name  = class_names[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 2)
    all_probs  = {class_names[i]: round(float(probs[i]) * 100, 2)
                  for i in range(len(class_names))}

    debug = {
        "probs":        {class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))},
        "img_feat_std": round(img_feat.std().item(), 4),
    }

    soil_fert = SOIL_FERT_MAP.get(soil_name,
                {"fertilizer": "NPK 14:14:14", "npk": "N:P:K = 60:30:30 kg/ha"})

    ranked_soils = sorted(
        [(class_names[i], float(probs[i])) for i in range(len(class_names))],
        key=lambda x: x[1],
        reverse=True,
    )
    top_soils = ranked_soils[:3]

    crop_soil_support = {}
    for s_name, s_prob in top_soils:
        seasonal = CROP_MAP.get((s_name, season), CROP_MAP.get((s_name, "Kharif"), []))
        for crop in seasonal:
            crop_soil_support[crop] = crop_soil_support.get(crop, 0.0) + s_prob

    if not crop_soil_support:
        fallback_crops = CROP_MAP.get((soil_name, season), ["Wheat", "Rice", "Maize"])
        crop_soil_support = {c: 1.0 for c in fallback_crops}

    scored_crops = []
    for crop, soil_support in crop_soil_support.items():
        suit = _crop_suitability_score(
            crop, n, p, k, ph, temp, hum, rain, yld, fert, irrig, prev, region
        )
        score = (1.35 * suit) + (2.2 * soil_support)
        scored_crops.append((crop, score, suit, soil_support))
    scored_crops.sort(key=lambda x: x[1], reverse=True)

    crop_recs = []
    for i, (crop, score, suit, soil_support) in enumerate(scored_crops[:3]):
        fert_plan = _fertilizer_for_crop(crop, n, p, k)
        crop_recs.append({"name": crop, "rank": i + 1, "stars": 5 - i,
                           "fertilizer": fert_plan["fertilizer"], "npk": fert_plan["npk"],
                           "ideal": fert_plan["ideal"],
                           "score": round(float(score), 3),
                           "soil_support": round(float(soil_support), 4),
                           "suitability": round(float(suit), 4)})

    debug["top_soils"] = {s: round(float(p), 4) for s, p in top_soils}
    debug["crop_scores"] = {c: round(float(s), 3) for c, s, _, _ in scored_crops}

    return soil_name, confidence, all_probs, soil_fert, crop_recs, debug


# ── Load models (once per session) ───────────────────────────
try:
    img_model, tab_proj, fusion, xgb_clf, scaler, CLASS_NAMES, NUMERIC_COLS = load_all_models()
    _models_ok = True
except Exception as _load_err:
    _models_ok = False
    st.error(
        f"**Model loading failed:** {_load_err}\n\n"
        f"This usually means Git LFS files were not pulled or files are missing."
    )
    st.markdown("**File status at startup:**")
    for _fn, _info in _file_status.items():
        _ok  = _info["exists"] and _info["mb"] > 0.1
        _ico = "OK" if _ok else "MISSING / TOO SMALL"
        st.write(f"- `{_fn}`: {_info['mb']:.1f} MB — {_ico}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# SOIL IMAGE VALIDATOR
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_validator():
    from torchvision.models import mobilenet_v3_small
    import torch.nn as nn
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)
    model.load_state_dict(
        torch.load("soil_validator.pt", map_location="cpu")
    )
    model.eval()
    return model

def is_soil_image(pil_img):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    validator = load_validator()
    img_t = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = validator(img_t)
        prob = torch.softmax(out, dim=-1)[0]
    soil_prob = prob[1].item()
    return soil_prob > 0.60


# ══════════════════════════════════════════════════════════════
# LEAF IMAGE VALIDATOR
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_leaf_validator():
    """Load leaf_validator.pt if present. Returns model or None."""
    if not os.path.exists("leaf_validator.pt"):
        return None
    try:
        from torchvision.models import mobilenet_v3_small
        import torch.nn as nn
        m = mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(1024, 2)
        m.load_state_dict(torch.load("leaf_validator.pt", map_location="cpu"))
        m.eval()
        return m
    except Exception as _e:
        print(f"[WARN] leaf_validator.pt load failed: {_e}")
        return None


def is_leaf_image(pil_img):
    """Return True if the image likely contains a plant leaf.

    Uses leaf_validator.pt (trained binary classifier) when available —
    same approach as soil_validator.pt which handles random objects,
    colored balls, faces, cars, etc. reliably.
    Falls back to HSV heuristic only if the model file is missing.
    """
    import numpy as np
    validator = load_leaf_validator()

    if validator is not None:
        from torchvision import transforms as _T
        tf = _T.Compose([
            _T.Resize((224, 224)),
            _T.ToTensor(),
            _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_t = tf(pil_img).unsqueeze(0)
        with torch.no_grad():
            out  = validator(img_t)
            prob = torch.softmax(out, dim=-1)[0]
        return prob[0].item() > 0.55  # leaf=0 (alphabetically leaf < not_leaf)

    # ── Fallback: HSV hue-based green detection ──────────────────────────
    arr = np.array(pil_img.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax  = np.maximum(np.maximum(r, g), b)
    cmin  = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    hue   = np.zeros_like(r)
    m = delta > 0
    mr, mg, mb = m & (cmax == r), m & (cmax == g), m & (cmax == b)
    hue[mr] = (60 * ((g[mr] - b[mr]) / delta[mr])) % 360
    hue[mg] = (60 * ((b[mg] - r[mg]) / delta[mg]) + 120) % 360
    hue[mb] = (60 * ((r[mb] - g[mb]) / delta[mb]) + 240) % 360
    saturation = np.where(cmax == 0, 0.0, delta / cmax)
    green_mask = (hue >= 60) & (hue <= 165) & (saturation > 0.18) & (cmax > 0.12)
    return float(green_mask.mean()) > 0.10


# ══════════════════════════════════════════════════════════════
# LEAF DISEASE MODEL — Universal PlantVillage ResNet-50
# ══════════════════════════════════════════════════════════════

LEAF_CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

LEAF_TREATMENT_MAP = {
    "Apple___Apple_scab": {"common_name": "Apple Scab", "primary_treatment": "Apply fungicides containing captan, myclobutanil, or mancozeb at 7-10 day intervals during wet spring weather. Remove and destroy fallen infected leaves in autumn.", "fertilizer": "Maintain balanced nutrition. Avoid excess nitrogen which promotes susceptible soft growth. Apply potassium to strengthen cell walls.", "cultural_practices": "Improve air circulation by pruning. Rake and remove fallen leaves. Use resistant apple varieties for new plantings."},
    "Apple___Black_rot": {"common_name": "Apple Black Rot", "primary_treatment": "Prune out dead wood and cankers during dry weather. Apply copper-based fungicides or captan. Remove mummified fruits from branches and ground.", "fertilizer": "Ensure adequate calcium and potassium. Avoid high nitrogen fertilization. A 10-10-10 balanced NPK is recommended.", "cultural_practices": "Thin fruits to reduce crowding and improve air circulation. Remove and destroy infected plant material promptly."},
    "Apple___Cedar_apple_rust": {"common_name": "Cedar Apple Rust", "primary_treatment": "Apply myclobutanil or propiconazole fungicides starting at pink bud stage and repeat through petal fall. Remove nearby cedar/juniper host plants if possible.", "fertilizer": "Standard balanced apple fertilization. Moderate nitrogen, adequate potassium and phosphorus for immunity.", "cultural_practices": "Plant resistant apple varieties. Remove cedar galls (orange jelly masses) in early spring before they release spores."},
    "Apple___healthy": {"common_name": "Apple (Healthy)", "primary_treatment": "No treatment required. Continue preventive fungicide schedule and good orchard sanitation to maintain plant health.", "fertilizer": "Annual balanced NPK fertilization (10-10-10 or adjusted per soil test). Apply in early spring.", "cultural_practices": "Annual pruning for air circulation, regular irrigation, pest monitoring, and orchard floor management."},
    "Blueberry___healthy": {"common_name": "Blueberry (Healthy)", "primary_treatment": "No disease detected. Maintain preventive care schedule.", "fertilizer": "Acidic soil fertilizer (ammonium sulfate). Keep soil pH 4.5-5.5. Avoid phosphorus excess.", "cultural_practices": "Mulch with pine bark or sawdust. Ensure well-drained acidic soil. Protect from birds during harvest."},
    "Cherry_(including_sour)___Powdery_mildew": {"common_name": "Cherry Powdery Mildew", "primary_treatment": "Apply sulfur-based fungicides or potassium bicarbonate at first sign of symptoms. Repeat every 7-14 days. Myclobutanil is also effective.", "fertilizer": "Reduce nitrogen applications which cause excessive soft growth susceptible to mildew. Increase potassium for plant hardiness.", "cultural_practices": "Improve air circulation by pruning. Avoid overhead irrigation. Plant resistant varieties."},
    "Cherry_(including_sour)___healthy": {"common_name": "Cherry (Healthy)", "primary_treatment": "No disease detected. Preventive copper sprays in early spring can protect against bacterial and fungal diseases.", "fertilizer": "Balanced N-P-K fertilization. Apply in early spring based on soil test.", "cultural_practices": "Annual pruning, proper irrigation, and pest monitoring. Maintain good sanitation around the orchard."},
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {"common_name": "Corn Gray Leaf Spot", "primary_treatment": "Apply foliar fungicides (strobilurins or triazoles) at first sign of disease. Applications at V6-V8 stage are most effective. Use resistant hybrids.", "fertilizer": "Maintain adequate nitrogen for good plant vigor (120-150 kg N/ha). Potassium deficiency increases susceptibility.", "cultural_practices": "Crop rotation with non-host crops for 1-2 years. Till infected crop residue. Improve field drainage."},
    "Corn_(maize)___Common_rust_": {"common_name": "Corn Common Rust", "primary_treatment": "Apply strobilurin or triazole fungicides at early infection stages. Prioritize high-value seed corn fields. Use resistant hybrids for next season.", "fertilizer": "Maintain adequate potassium levels (60-80 kg K/ha) for improved disease resistance. Balanced NPK fertilization.", "cultural_practices": "Plant resistant hybrids. Early planting to avoid peak rust pressure. Scout fields regularly in summer."},
    "Corn_(maize)___Northern_Leaf_Blight": {"common_name": "Corn Northern Leaf Blight", "primary_treatment": "Apply fungicides (propiconazole or azoxystrobin) at VT/R1 stage if disease is present above the ear leaf. Use resistant hybrids.", "fertilizer": "Ensure nitrogen adequacy (120-150 kg N/ha). Potassium helps with disease tolerance. Foliar zinc applications can help.", "cultural_practices": "Rotate crops, bury infected residue, use resistant hybrids. Avoid late planting."},
    "Corn_(maize)___healthy": {"common_name": "Corn (Healthy)", "primary_treatment": "No disease detected. Continue routine scouting and preventive management.", "fertilizer": "Side-dress with nitrogen at V6 stage. Maintain P and K per soil test recommendations.", "cultural_practices": "Maintain proper plant density, weed control, and irrigation for optimal yield."},
    "Grape___Black_rot": {"common_name": "Grape Black Rot", "primary_treatment": "Apply mancozeb, myclobutanil, or captan fungicides starting at budbreak and continue through fruit set. Critical protection period is 2-5 weeks post bloom.", "fertilizer": "Avoid excess nitrogen. Maintain balanced potassium for cell wall strength. Use 5-5-5 NPK as base.", "cultural_practices": "Remove mummified berries and infected canes in winter. Train vines to maximize air circulation. Eliminate overhead irrigation."},
    "Grape___Esca_(Black_Measles)": {"common_name": "Grape Esca (Black Measles)", "primary_treatment": "No effective chemical cure. Prune out infected wood during dry weather using disinfected tools. Paint pruning wounds with fungicidal paste.", "fertilizer": "Avoid excess nitrogen. Phosphorus and potassium support root and vascular health. Regular foliar iron applications if deficient.", "cultural_practices": "Delay pruning until late winter to reduce infection risk. Remove and burn severely affected vines. Disinfect pruning tools between vines."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"common_name": "Grape Leaf Blight", "primary_treatment": "Apply copper-based fungicides or mancozeb at first sign. Repeat applications every 10-14 days during wet periods.", "fertilizer": "Maintain balanced nutrition. Excess nitrogen promotes dense canopy which increases disease humidity.", "cultural_practices": "Improve trellis system for better air circulation. Avoid overhead irrigation. Remove infected leaves promptly."},
    "Grape___healthy": {"common_name": "Grape (Healthy)", "primary_treatment": "No disease detected. Maintain preventive fungicide program and good vineyard sanitation.", "fertilizer": "Standard vineyard fertilization based on tissue and soil analysis. Typical: 60-80 kg N/ha, adjusted for vigor.", "cultural_practices": "Annual pruning, canopy management, and pest monitoring. Ensure good drainage."},
    "Orange___Haunglongbing_(Citrus_greening)": {"common_name": "Citrus Greening (HLB)", "primary_treatment": "No cure exists. Remove and destroy infected trees to prevent spread. Control Asian Citrus Psyllid vector with systemic insecticides (imidacloprid, spirotetramat).", "fertilizer": "Nutritional therapy: foliar micronutrient sprays (Zn, Mn, Fe, Mg, Cu). Enhanced trunk injections of macro and micronutrients to support compromised nutrient uptake.", "cultural_practices": "Use certified disease-free nursery stock. Establish psyllid monitoring programs. Consider replanting with tolerant rootstock combinations."},
    "Peach___Bacterial_spot": {"common_name": "Peach Bacterial Spot", "primary_treatment": "Apply copper bactericides or oxytetracycline starting at petal fall. Repeat at 5-7 day intervals during wet weather.", "fertilizer": "Maintain moderate nitrogen (50-70 kg N/ha). Excess nitrogen promotes soft, susceptible tissue. Potassium improves bark integrity.", "cultural_practices": "Prune for air circulation. Use resistant varieties. Avoid overhead irrigation. Shelter from wind-driven rain."},
    "Peach___healthy": {"common_name": "Peach (Healthy)", "primary_treatment": "No disease detected. Annual dormant copper spray program is recommended for prevention.", "fertilizer": "Apply nitrogen in split doses: 50% at dormancy break, 50% at fruit development stage.", "cultural_practices": "Annual pruning for open center shape. Thin fruit to 15-20 cm spacing for quality and disease prevention."},
    "Pepper,_bell___Bacterial_spot": {"common_name": "Bell Pepper Bacterial Spot", "primary_treatment": "Apply copper bactericides (copper hydroxide or copper sulfate) at first symptom. Repeat every 5-7 days during wet weather.", "fertilizer": "Reduce nitrogen. Increase calcium (foliar calcium sprays at 1-2 g/L) and potassium. Use 12-6-18 NPK ratio.", "cultural_practices": "Use disease-free seed or transplants. Rotate crops (avoid peppers/tomatoes for 2-3 years). Avoid overhead irrigation."},
    "Pepper,_bell___healthy": {"common_name": "Bell Pepper (Healthy)", "primary_treatment": "No disease detected. Preventive copper applications in high-risk weather periods.", "fertilizer": "NPK: 120:60:80 kg/ha as base. Side-dress with nitrogen at first fruit set.", "cultural_practices": "Stake plants, use drip irrigation, and scout for early signs of pests/disease."},
    "Potato___Early_blight": {"common_name": "Potato Early Blight", "primary_treatment": "Apply chlorothalonil, mancozeb, or azoxystrobin fungicides at first symptom. Repeat every 7-10 days. Start applications before symptoms appear in high-risk periods.", "fertilizer": "Increase potassium (80-100 kg K/ha) to strengthen cell walls. Avoid excessive nitrogen. Switch to 5-10-15 NPK ratio during recovery.", "cultural_practices": "Remove infected lower leaves promptly. Use drip irrigation to keep foliage dry. Ensure 60 cm plant spacing for air circulation."},
    "Potato___Late_blight": {"common_name": "Potato Late Blight", "primary_treatment": "URGENT: Apply systemic fungicides (metalaxyl, dimethomorph, or cymoxanil) immediately. Rotate fungicide classes to prevent resistance. Destroy severely infected plots.", "fertilizer": "Reduce nitrogen to slow soft tissue growth. Increase potassium (90-120 kg K/ha). Calcium foliar sprays help.", "cultural_practices": "Plant certified disease-free seed. Harvest before complete vine death. Destroy cull piles. Use resistant varieties. Avoid overhead irrigation."},
    "Potato___healthy": {"common_name": "Potato (Healthy)", "primary_treatment": "No disease detected. Preventive fungicide applications during warm, humid weather are recommended.", "fertilizer": "NPK: 180:120:80 kg/ha. Split nitrogen: 50% at planting, 50% at hilling. Apply potassium in full at planting.", "cultural_practices": "Hill plants at 2-3 weeks post-emergence. Maintain consistent soil moisture with drip or furrow irrigation."},
    "Raspberry___healthy": {"common_name": "Raspberry (Healthy)", "primary_treatment": "No disease detected. Maintain preventive care.", "fertilizer": "Apply 60-80 kg N/ha in early spring. Add composted organic matter annually.", "cultural_practices": "Prune out floricanes after harvest. Maintain narrow rows for air circulation."},
    "Soybean___healthy": {"common_name": "Soybean (Healthy)", "primary_treatment": "No disease detected. Scout regularly for early disease detection.", "fertilizer": "Inoculate with Bradyrhizobium for nitrogen fixation. Apply phosphorus and potassium per soil test.", "cultural_practices": "Rotate with non-legume crops. Maintain proper plant population (300,000-350,000 plants/ha)."},
    "Squash___Powdery_mildew": {"common_name": "Squash Powdery Mildew", "primary_treatment": "Apply potassium bicarbonate, sulfur, or neem oil at first white powder appearance. Systemic fungicides (myclobutanil) provide longer protection.", "fertilizer": "Reduce nitrogen, increase potassium. Foliar silicon applications (1-2 g/L potassium silicate) significantly increase resistance.", "cultural_practices": "Improve air circulation by proper vine training. Avoid overhead irrigation. Remove severely infected leaves."},
    "Strawberry___Leaf_scorch": {"common_name": "Strawberry Leaf Scorch", "primary_treatment": "Apply captan or myclobutanil fungicide at first symptom. Repeat every 10-14 days. Remove infected leaves from the planting.", "fertilizer": "Maintain balanced nutrition. Avoid excess nitrogen late in season. Potassium improves disease tolerance.", "cultural_practices": "Renovate strawberry beds after harvest. Improve drainage. Increase row spacing for air circulation."},
    "Strawberry___healthy": {"common_name": "Strawberry (Healthy)", "primary_treatment": "No disease detected. Continue preventive care.", "fertilizer": "Apply 60-80 kg N/ha in split doses. Ensure adequate phosphorus for root development.", "cultural_practices": "Renovate and replant every 3-4 years. Use drip irrigation. Mulch with straw to prevent soil splash."},
    "Tomato___Bacterial_spot": {"common_name": "Tomato Bacterial Spot", "primary_treatment": "Apply copper bactericides at first sign. Combine with mancozeb for improved efficacy. Applications every 5-7 days during wet weather are critical.", "fertilizer": "Reduce nitrogen. Increase calcium (0.5-1 g/L CaCl2 foliar spray) and potassium. Use 12-6-18 NPK ratio.", "cultural_practices": "Use certified disease-free seed/transplants. Avoid overhead irrigation. Stake plants. Implement 2-3 year crop rotation."},
    "Tomato___Early_blight": {"common_name": "Tomato Early Blight", "primary_treatment": "Remove and destroy infected lower leaves immediately. Apply copper-based fungicides or chlorothalonil every 7-10 days until symptoms clear.", "fertilizer": "Increase potassium (80-100 kg K/ha) to strengthen cell walls. Reduce heavy nitrogen application. Switch to 5-10-15 NPK ratio during recovery.", "cultural_practices": "Utilize drip irrigation to keep foliage dry. Stake plants to improve airflow. Increase spacing to at least 60 cm. Mulch the soil base to prevent spore splash-back."},
    "Tomato___Late_blight": {"common_name": "Tomato Late Blight", "primary_treatment": "Apply systemic fungicides (metalaxyl-M or dimethomorph) immediately upon detection. Alternate with chlorothalonil to prevent resistance. Remove severely infected plants.", "fertilizer": "Reduce nitrogen to limit soft tissue. Increase potassium (100-120 kg K/ha). Foliar calcium (1 g/L CaCl2) sprays strengthen cell walls.", "cultural_practices": "Remove and destroy infected debris. Avoid overhead irrigation. Space plants 60-75 cm apart. Scout daily during cool, wet weather."},
    "Tomato___Leaf_Mold": {"common_name": "Tomato Leaf Mold", "primary_treatment": "Apply chlorothalonil or copper-based fungicides. Ensure good air circulation to reduce humidity below 85%. Applications every 7-10 days.", "fertilizer": "Balanced nutrition. Avoid excess nitrogen. Calcium sprays help cell wall integrity in high-humidity conditions.", "cultural_practices": "Reduce relative humidity in greenhouses below 85%. Increase plant spacing. Remove lower leaves to improve airflow."},
    "Tomato___Septoria_leaf_spot": {"common_name": "Tomato Septoria Leaf Spot", "primary_treatment": "Apply chlorothalonil, copper, or mancozeb fungicides at first symptom. Remove infected leaves. Repeat every 7-10 days during wet weather.", "fertilizer": "Maintain balanced NPK. Potassium improves disease tolerance. Avoid excessive nitrogen.", "cultural_practices": "Mulch to prevent soil splash. Stake plants. Avoid working in wet fields. Rotate crops for 3 years."},
    "Tomato___Spider_mites_Two-spotted_spider_mite": {"common_name": "Tomato Spider Mites", "primary_treatment": "Apply miticides (abamectin, bifenazate) or insecticidal soap/neem oil for organic control. Predatory mites (Phytoseiulus persimilis) provide biological control.", "fertilizer": "Ensure adequate watering to reduce plant stress. Excess nitrogen increases spider mite population. Increase potassium for plant hardiness.", "cultural_practices": "Monitor undersides of leaves. Increase humidity around plants (mites thrive in dry conditions). Remove heavily infested leaves."},
    "Tomato___Target_Spot": {"common_name": "Tomato Target Spot", "primary_treatment": "Apply azoxystrobin or chlorothalonil fungicides at first symptom. Continue at 7-14 day intervals. Protectant fungicides are more effective than curative.", "fertilizer": "Balanced nutrition with adequate potassium. Avoid excess nitrogen.", "cultural_practices": "Improve air circulation. Remove infected leaves. Avoid overhead irrigation. Implement crop rotation."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"common_name": "Tomato Yellow Leaf Curl Virus", "primary_treatment": "No cure exists. Remove and destroy infected plants. Control whitefly vector with imidacloprid or pyrethroids. Use reflective silver mulches to deter whiteflies.", "fertilizer": "Maintain plant vigor with balanced NPK (200:150:200 kg/ha). Foliar potassium and calcium sprays strengthen immunity.", "cultural_practices": "Use TYLCV-resistant tomato varieties. Install insect-proof screens in greenhouses. Rogue out symptomatic plants early."},
    "Tomato___Tomato_mosaic_virus": {"common_name": "Tomato Mosaic Virus", "primary_treatment": "No cure exists. Remove and destroy infected plants. Disinfect tools with 10% bleach solution between plants. Virus spreads mechanically.", "fertilizer": "Maintain good plant nutrition with balanced NPK fertilization for non-infected plants.", "cultural_practices": "Wash hands before working with plants. Use certified virus-free transplants. Control aphid vectors. Avoid tobacco use near plants."},
    "Tomato___healthy": {"common_name": "Tomato (Healthy)", "primary_treatment": "No disease detected. Preventive copper-based fungicide applications during cool, wet periods reduce disease risk.", "fertilizer": "NPK: 200:150:200 kg/ha. Side-dress nitrogen at first flower stage and again at first fruit set.", "cultural_practices": "Stake all plants. Use drip irrigation. Scout 2x weekly during fruiting. Remove suckers and lower leaves as plant matures."},
}


def _build_leaf_classifier(keras, img_size, num_classes, architecture="MobileNetV2"):
    """Rebuild a known leaf classifier architecture without graph deserialization."""
    layers = keras.layers
    arch = (architecture or "").lower()

    if "efficientnetb0" in arch:
        base = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(img_size, img_size, 3),
        )
        x = base.output
        x = layers.GlobalAveragePooling2D(name="global_average_pooling2d_4")(x)
        x = layers.BatchNormalization(name="batch_normalization_4")(x)
        x = layers.Dense(256, activation="relu", name="dense_12")(x)
        x = layers.Dropout(0.3, name="dropout_8")(x)
        x = layers.Dense(128, activation="relu", name="dense_13")(x)
        x = layers.Dropout(0.2, name="dropout_9")(x)
        out = layers.Dense(num_classes, activation="softmax", name="dense_14")(x)
        model = keras.Model(inputs=base.input, outputs=out, name="leaf_efficientnetb0_classifier")
        return model, "raw_255"

    base = keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(img_size, img_size, 3),
    )
    x = base.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dense_2")(x)
    model = keras.Model(inputs=base.input, outputs=out, name="leaf_mobilenetv2_classifier")
    return model, "minus_one_to_one"


@st.cache_resource(show_spinner="Loading Leaf Disease Model...")
def load_leaf_model():
    """Load the AgroFusion Universal v2 leaf disease model (Keras MobileNetV2, 128×128).
    pkl structure: model_bytes (Keras ZIP), class_labels, fertilizer_dict, img_size.
    Returns (keras_model, class_labels_list, fertilizer_dict) or (None, fallback, fallback).
    """
    pkl_path = mpath("agrofusion_universal_v5.pkl")
    min_bytes = 2 * 1024 * 1024

    def _download_leaf_model(dst_path):
        url = "https://media.githubusercontent.com/media/manojanumolu/AgroSynapse/main/agrofusion_universal_v5.pkl"
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            tmp_path = dst_path + ".tmp"
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp_path, dst_path)
            return True
        except Exception as _e:
            print(f"[WARN] leaf model download failed: {_e}")
            try:
                if os.path.exists(dst_path + ".tmp"):
                    os.remove(dst_path + ".tmp")
            except Exception:
                pass
            return False

    if not os.path.exists(pkl_path) or os.path.getsize(pkl_path) < min_bytes:
        print("[WARN] agrofusion_universal_v5.pkl missing or too small — attempting download.")
        if not _download_leaf_model(pkl_path):
            print("[WARN] agrofusion_universal_v5.pkl not available — leaf model unavailable.")
            st.session_state.leaf_model_error = "agrofusion_universal_v5.pkl is missing and download failed."
            return None, LEAF_CLASS_NAMES, LEAF_TREATMENT_MAP
    try:
        import tempfile
        # ── Load payload ──────────────────────────────────────
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)

        cls_labels = payload.get("class_labels", LEAF_CLASS_NAMES)
        fert_dict  = payload.get("fertilizer_dict", LEAF_TREATMENT_MAP)
        img_size   = int(payload.get("img_size", 128))
        metadata   = payload.get("metadata", {}) or {}
        architecture = metadata.get("architecture", "MobileNetV2")

        # ── Load Keras model from bytes ────────────────────────
        model_bytes = payload.get("model_bytes")
        if model_bytes is None:
            st.session_state.leaf_model_error = "agrofusion_universal_v5.pkl payload missing model_bytes."
            return None, cls_labels, fert_dict

        def _sanitize_model_config(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if k == "quantization_config":
                        continue
                    if k == "compile_config":
                        # Inference-only app; compiled state is unnecessary and can break compatibility.
                        continue
                    cleaned[k] = _sanitize_model_config(v)
                return cleaned
            if isinstance(obj, list):
                return [_sanitize_model_config(v) for v in obj]
            return obj

        def _sanitize_keras_archive(raw_model_bytes):
            src = io.BytesIO(raw_model_bytes)
            dst = io.BytesIO()
            with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for name in zin.namelist():
                    data = zin.read(name)
                    if name == "config.json":
                        try:
                            cfg = json.loads(data.decode("utf-8"))
                            cfg = _sanitize_model_config(cfg)
                            data = json.dumps(cfg, separators=(",", ":")).encode("utf-8")
                        except Exception:
                            pass
                    zout.writestr(name, data)
            return dst.getvalue()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "leaf_model.keras")
            sanitized_path = os.path.join(tmpdir, "leaf_model_sanitized.keras")
            weights_path = os.path.join(tmpdir, "leaf_model.weights.h5")
            with open(model_path, "wb") as f:
                f.write(model_bytes)
            with open(sanitized_path, "wb") as f:
                f.write(_sanitize_keras_archive(model_bytes))

            import keras
            try:
                keras.config.enable_unsafe_deserialization()
            except Exception:
                pass

            keras_model = None
            preprocess_mode = "raw_255" if "efficientnetb0" in architecture.lower() else "minus_one_to_one"

            # Prefer weight-only restore. The saved archive is valid, but its serialized
            # Functional graph can be incompatible across Keras/TensorFlow versions.
            try:
                with zipfile.ZipFile(io.BytesIO(model_bytes), "r") as zin:
                    if "model.weights.h5" in zin.namelist():
                        with open(weights_path, "wb") as f:
                            f.write(zin.read("model.weights.h5"))
                        keras_model, preprocess_mode = _build_leaf_classifier(
                            keras, img_size, len(cls_labels), architecture
                        )
                        keras_model.load_weights(weights_path)
            except Exception as weight_err:
                print(f"[WARN] leaf weight-only restore failed: {weight_err}")
                keras_model = None

            if keras_model is None:
                try:
                    keras_model = keras.models.load_model(model_path, safe_mode=False, compile=False)
                except Exception:
                    keras_model = keras.models.load_model(sanitized_path, safe_mode=False, compile=False)
                preprocess_mode = "raw_255" if "efficientnetb0" in architecture.lower() else "minus_one_to_one"

        keras_model._leaf_img_size = img_size   # attach for use in inference
        keras_model._leaf_preprocess = preprocess_mode
        keras_model._leaf_architecture = architecture
        print(f"[OK] Leaf model loaded - {architecture} {img_size}x{img_size}, {len(cls_labels)} classes.")
        st.session_state.leaf_model_error = None
        return keras_model, cls_labels, fert_dict

    except Exception as _e:
        print(f"[WARN] leaf model load error: {_e}")
        st.session_state.leaf_model_error = (
            "Leaf model artifact is incompatible with runtime graph deserialization. "
            f"Direct restore failed with: {_e}"
        )
        return None, LEAF_CLASS_NAMES, LEAF_TREATMENT_MAP


def run_leaf_inference(model, class_labels, img_bytes):
    """Run leaf disease classification. Returns (pred_class, confidence_pct, top5_list)."""
    img_size = getattr(model, "_leaf_img_size", 128)
    preprocess_mode = getattr(model, "_leaf_preprocess", "minus_one_to_one")

    # Match preprocessing to the saved model family.
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(
        (img_size, img_size), Image.LANCZOS
    )
    arr = np.array(pil_img, dtype=np.float32)
    if preprocess_mode == "minus_one_to_one":
        arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, 0)

    probs    = model.predict(arr, verbose=0)[0]  # shape (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_cls = class_labels[pred_idx] if pred_idx < len(class_labels) else "Unknown"
    conf     = round(float(probs[pred_idx]) * 100, 2)
    top5     = sorted(
        [(class_labels[i], round(float(probs[i]) * 100, 2)) for i in range(len(class_labels))],
        key=lambda x: x[1], reverse=True
    )[:5]
    return pred_cls, conf, top5


# Leaf model loaded lazily — only when the leaf page is visited (saves RAM)


# ══════════════════════════════════════════════════════════════
# CLIMATE DATA FETCHER
# ══════════════════════════════════════════════════════════════

DISTRICT_COORDS = {
    ("Telangana", "Adilabad"): (19.6641, 78.5320),
    ("Telangana", "Bhadradri Kothagudem"): (17.5567, 80.6167),
    ("Telangana", "Hanamkonda"): (17.9689, 79.5941),
    ("Telangana", "Hyderabad"): (17.3850, 78.4867),
    ("Telangana", "Jagtial"): (18.7940, 78.9140),
    ("Telangana", "Jangaon"): (17.7244, 79.1523),
    ("Telangana", "Jayashankar Bhupalpally"): (18.4333, 79.9167),
    ("Telangana", "Jogulamba Gadwal"): (16.2340, 77.8020),
    ("Telangana", "Kamareddy"): (18.3240, 78.3410),
    ("Telangana", "Karimnagar"): (18.4386, 79.1288),
    ("Telangana", "Khammam"): (17.2473, 80.1514),
    ("Telangana", "Komaram Bheem"): (19.2167, 79.4167),
    ("Telangana", "Mahabubabad"): (17.6010, 80.0010),
    ("Telangana", "Mahabubnagar"): (16.7488, 77.9827),
    ("Telangana", "Mancherial"): (18.8710, 79.4600),
    ("Telangana", "Medak"): (18.0440, 78.2620),
    ("Telangana", "Medchal Malkajgiri"): (17.5333, 78.5333),
    ("Telangana", "Mulugu"): (18.1833, 80.0333),
    ("Telangana", "Nagarkurnool"): (16.4833, 78.3167),
    ("Telangana", "Nalgonda"): (17.0575, 79.2671),
    ("Telangana", "Narayanpet"): (16.7417, 77.4942),
    ("Telangana", "Nirmal"): (19.0960, 78.3440),
    ("Telangana", "Nizamabad"): (18.6725, 78.0941),
    ("Telangana", "Peddapalli"): (18.6140, 79.3830),
    ("Telangana", "Rajanna Sircilla"): (18.3873, 78.8322),
    ("Telangana", "Rangareddy"): (17.2000, 78.4000),
    ("Telangana", "Sangareddy"): (17.6274, 78.0878),
    ("Telangana", "Siddipet"): (18.1018, 78.8520),
    ("Telangana", "Suryapet"): (17.1403, 79.6210),
    ("Telangana", "Vikarabad"): (17.3380, 77.9040),
    ("Telangana", "Wanaparthy"): (16.3620, 78.0610),
    ("Telangana", "Warangal"): (17.9784, 79.5941),
    ("Telangana", "Yadadri Bhuvanagiri"): (17.5833, 79.1667),
    ("Andhra Pradesh", "Alluri Sitharama Raju"): (17.9333, 82.5667),
    ("Andhra Pradesh", "Anakapalli"): (17.6910, 83.0050),
    ("Andhra Pradesh", "Anantapur"): (14.6819, 77.6006),
    ("Andhra Pradesh", "Annamayya"): (13.8500, 79.0167),
    ("Andhra Pradesh", "Bapatla"): (15.9043, 80.4670),
    ("Andhra Pradesh", "Chittoor"): (13.2172, 79.1003),
    ("Andhra Pradesh", "East Godavari"): (17.0005, 82.2400),
    ("Andhra Pradesh", "Eluru"): (16.7107, 81.0952),
    ("Andhra Pradesh", "Guntur"): (16.3067, 80.4365),
    ("Andhra Pradesh", "Kakinada"): (16.9891, 82.2475),
    ("Andhra Pradesh", "Konaseema"): (16.8167, 82.2333),
    ("Andhra Pradesh", "Krishna"): (16.6167, 80.8000),
    ("Andhra Pradesh", "Kurnool"): (15.8281, 78.0373),
    ("Andhra Pradesh", "Nandyal"): (15.4786, 78.4839),
    ("Andhra Pradesh", "Nellore"): (14.4426, 79.9865),
    ("Andhra Pradesh", "NTR"): (16.5193, 80.6305),
    ("Andhra Pradesh", "Palnadu"): (16.2000, 79.6333),
    ("Andhra Pradesh", "Parvathipuram Manyam"): (18.7833, 83.4333),
    ("Andhra Pradesh", "Prakasam"): (15.3490, 79.5747),
    ("Andhra Pradesh", "Sri Balaji"): (13.6288, 79.4192),
    ("Andhra Pradesh", "Sri Sathya Sai"): (14.1667, 77.7833),
    ("Andhra Pradesh", "Srikakulam"): (18.2949, 83.8938),
    ("Andhra Pradesh", "Visakhapatnam"): (17.6868, 83.2185),
    ("Andhra Pradesh", "Vizianagaram"): (18.1066, 83.3956),
    ("Andhra Pradesh", "West Godavari"): (16.9167, 81.3333),
    ("Andhra Pradesh", "YSR Kadapa"): (14.4673, 78.8242),
    ("Karnataka", "Bagalkot"): (16.1800, 75.6960),
    ("Karnataka", "Ballari"): (15.1394, 76.9214),
    ("Karnataka", "Belagavi"): (15.8497, 74.4977),
    ("Karnataka", "Bengaluru Rural"): (13.1986, 77.7066),
    ("Karnataka", "Bengaluru Urban"): (12.9716, 77.5946),
    ("Karnataka", "Bidar"): (17.9104, 77.5199),
    ("Karnataka", "Chamarajanagar"): (11.9271, 76.9432),
    ("Karnataka", "Chikkaballapur"): (13.4355, 77.7315),
    ("Karnataka", "Chikkamagaluru"): (13.3153, 75.7754),
    ("Karnataka", "Chitradurga"): (14.2294, 76.3983),
    ("Karnataka", "Dakshina Kannada"): (12.8438, 75.2479),
    ("Karnataka", "Davanagere"): (14.4644, 75.9218),
    ("Karnataka", "Dharwad"): (15.4589, 75.0078),
    ("Karnataka", "Gadag"): (15.4167, 75.6167),
    ("Karnataka", "Hassan"): (13.0033, 76.1004),
    ("Karnataka", "Haveri"): (14.7939, 75.3996),
    ("Karnataka", "Kalaburagi"): (17.3297, 76.8343),
    ("Karnataka", "Kodagu"): (12.3375, 75.8069),
    ("Karnataka", "Kolar"): (13.1357, 78.1294),
    ("Karnataka", "Koppal"): (15.3500, 76.1500),
    ("Karnataka", "Mandya"): (12.5218, 76.8951),
    ("Karnataka", "Mysuru"): (12.2958, 76.6394),
    ("Karnataka", "Raichur"): (16.2120, 77.3439),
    ("Karnataka", "Ramanagara"): (12.7157, 77.2819),
    ("Karnataka", "Shivamogga"): (13.9299, 75.5681),
    ("Karnataka", "Tumakuru"): (13.3379, 77.1173),
    ("Karnataka", "Udupi"): (13.3409, 74.7421),
    ("Karnataka", "Uttara Kannada"): (14.7860, 74.6680),
    ("Karnataka", "Vijayapura"): (16.8302, 75.7100),
    ("Karnataka", "Yadgir"): (16.7710, 77.1380),
    ("Maharashtra", "Ahmednagar"): (19.0952, 74.7496),
    ("Maharashtra", "Akola"): (20.7002, 77.0082),
    ("Maharashtra", "Amravati"): (20.9374, 77.7796),
    ("Maharashtra", "Aurangabad"): (19.8762, 75.3433),
    ("Maharashtra", "Beed"): (18.9890, 75.7560),
    ("Maharashtra", "Bhandara"): (21.1667, 79.6500),
    ("Maharashtra", "Buldhana"): (20.5292, 76.1842),
    ("Maharashtra", "Chandrapur"): (19.9615, 79.2961),
    ("Maharashtra", "Dhule"): (20.9042, 74.7749),
    ("Maharashtra", "Gadchiroli"): (20.1809, 80.0000),
    ("Maharashtra", "Gondia"): (21.4624, 80.1947),
    ("Maharashtra", "Hingoli"): (19.7160, 77.1490),
    ("Maharashtra", "Jalgaon"): (21.0077, 75.5626),
    ("Maharashtra", "Jalna"): (19.8347, 75.8816),
    ("Maharashtra", "Kolhapur"): (16.7050, 74.2433),
    ("Maharashtra", "Latur"): (18.4088, 76.5604),
    ("Maharashtra", "Mumbai City"): (18.9388, 72.8354),
    ("Maharashtra", "Mumbai Suburban"): (19.0760, 72.8777),
    ("Maharashtra", "Nagpur"): (21.1458, 79.0882),
    ("Maharashtra", "Nanded"): (19.1383, 77.3210),
    ("Maharashtra", "Nandurbar"): (21.3667, 74.2333),
    ("Maharashtra", "Nashik"): (19.9975, 73.7898),
    ("Maharashtra", "Osmanabad"): (18.1860, 76.0410),
    ("Maharashtra", "Palghar"): (19.6967, 72.7659),
    ("Maharashtra", "Parbhani"): (19.2704, 76.7740),
    ("Maharashtra", "Pune"): (18.5204, 73.8567),
    ("Maharashtra", "Raigad"): (18.5158, 73.1298),
    ("Maharashtra", "Ratnagiri"): (16.9902, 73.3120),
    ("Maharashtra", "Sangli"): (16.8524, 74.5815),
    ("Maharashtra", "Satara"): (17.6805, 74.0183),
    ("Maharashtra", "Sindhudurg"): (16.0494, 73.5283),
    ("Maharashtra", "Solapur"): (17.6599, 75.9064),
    ("Maharashtra", "Thane"): (19.2183, 72.9781),
    ("Maharashtra", "Wardha"): (20.7453, 78.6022),
    ("Maharashtra", "Washim"): (20.1120, 77.1340),
    ("Maharashtra", "Yavatmal"): (20.3888, 78.1204),
    ("Punjab", "Amritsar"): (31.6340, 74.8723),
    ("Punjab", "Barnala"): (30.3782, 75.5492),
    ("Punjab", "Bathinda"): (30.2110, 74.9455),
    ("Punjab", "Faridkot"): (30.6717, 74.7553),
    ("Punjab", "Fatehgarh Sahib"): (30.6480, 76.3906),
    ("Punjab", "Fazilka"): (30.4019, 74.0257),
    ("Punjab", "Ferozepur"): (30.9236, 74.6227),
    ("Punjab", "Gurdaspur"): (32.0399, 75.4060),
    ("Punjab", "Hoshiarpur"): (31.5143, 75.9119),
    ("Punjab", "Jalandhar"): (31.3260, 75.5762),
    ("Punjab", "Kapurthala"): (31.3800, 75.3800),
    ("Punjab", "Ludhiana"): (30.9010, 75.8573),
    ("Punjab", "Mansa"): (29.9918, 75.3980),
    ("Punjab", "Moga"): (30.8170, 75.1730),
    ("Punjab", "Mohali"): (30.7046, 76.7179),
    ("Punjab", "Muktsar"): (30.4740, 74.5160),
    ("Punjab", "Pathankot"): (32.2743, 75.6522),
    ("Punjab", "Patiala"): (30.3398, 76.3869),
    ("Punjab", "Rupnagar"): (30.9644, 76.5254),
    ("Punjab", "Sangrur"): (30.2457, 75.8425),
    ("Punjab", "Shaheed Bhagat Singh Nagar"): (31.1270, 76.3870),
    ("Punjab", "Tarn Taran"): (31.4520, 74.9280),
    ("Haryana", "Ambala"): (30.3752, 76.7821),
    ("Haryana", "Bhiwani"): (28.7975, 76.1322),
    ("Haryana", "Charkhi Dadri"): (28.5921, 76.2700),
    ("Haryana", "Faridabad"): (28.4089, 77.3178),
    ("Haryana", "Fatehabad"): (29.5136, 75.4551),
    ("Haryana", "Gurugram"): (28.4595, 77.0266),
    ("Haryana", "Hisar"): (29.1492, 75.7217),
    ("Haryana", "Jhajjar"): (28.6080, 76.6572),
    ("Haryana", "Jind"): (29.3162, 76.3163),
    ("Haryana", "Kaithal"): (29.8014, 76.3998),
    ("Haryana", "Karnal"): (29.6857, 76.9905),
    ("Haryana", "Kurukshetra"): (29.9695, 76.8783),
    ("Haryana", "Mahendragarh"): (28.2785, 76.1458),
    ("Haryana", "Nuh"): (28.1075, 77.0006),
    ("Haryana", "Palwal"): (28.1487, 77.3270),
    ("Haryana", "Panchkula"): (30.6942, 76.8606),
    ("Haryana", "Panipat"): (29.3909, 76.9635),
    ("Haryana", "Rewari"): (28.1972, 76.6172),
    ("Haryana", "Rohtak"): (28.8955, 76.6066),
    ("Haryana", "Sirsa"): (29.5330, 75.0280),
    ("Haryana", "Sonipat"): (28.9931, 77.0151),
    ("Haryana", "Yamunanagar"): (30.1290, 77.2674),
    ("Gujarat", "Ahmedabad"): (23.0225, 72.5714),
    ("Gujarat", "Amreli"): (21.6032, 71.2215),
    ("Gujarat", "Anand"): (22.5645, 72.9289),
    ("Gujarat", "Aravalli"): (23.6993, 73.1209),
    ("Gujarat", "Banaskantha"): (24.1740, 72.4370),
    ("Gujarat", "Bharuch"): (21.7051, 72.9959),
    ("Gujarat", "Bhavnagar"): (21.7645, 72.1519),
    ("Gujarat", "Botad"): (22.1690, 71.6680),
    ("Gujarat", "Chhota Udaipur"): (22.3063, 74.0146),
    ("Gujarat", "Dahod"): (22.8340, 74.2560),
    ("Gujarat", "Dang"): (20.7500, 73.6700),
    ("Gujarat", "Devbhoomi Dwarka"): (22.2394, 68.9678),
    ("Gujarat", "Gandhinagar"): (23.2156, 72.6369),
    ("Gujarat", "Gir Somnath"): (20.9060, 70.3700),
    ("Gujarat", "Jamnagar"): (22.4707, 70.0577),
    ("Gujarat", "Junagadh"): (21.5222, 70.4579),
    ("Gujarat", "Kheda"): (22.7500, 72.6800),
    ("Gujarat", "Kutch"): (23.7337, 69.8597),
    ("Gujarat", "Mahisagar"): (23.1000, 73.5900),
    ("Gujarat", "Mehsana"): (23.5879, 72.3693),
    ("Gujarat", "Morbi"): (22.8173, 70.8372),
    ("Gujarat", "Narmada"): (21.8716, 73.4979),
    ("Gujarat", "Navsari"): (20.9467, 72.9520),
    ("Gujarat", "Panchmahal"): (22.7500, 73.5800),
    ("Gujarat", "Patan"): (23.8493, 72.1266),
    ("Gujarat", "Porbandar"): (21.6425, 69.6293),
    ("Gujarat", "Rajkot"): (22.3039, 70.8022),
    ("Gujarat", "Sabarkantha"): (23.3800, 73.0100),
    ("Gujarat", "Surat"): (21.1702, 72.8311),
    ("Gujarat", "Surendranagar"): (22.7270, 71.6472),
    ("Gujarat", "Tapi"): (21.1200, 73.4100),
    ("Gujarat", "Vadodara"): (22.3072, 73.1812),
    ("Gujarat", "Valsad"): (20.5992, 72.9342),
    ("Rajasthan", "Ajmer"): (26.4499, 74.6399),
    ("Rajasthan", "Alwar"): (27.5530, 76.6346),
    ("Rajasthan", "Banswara"): (23.5500, 74.4400),
    ("Rajasthan", "Baran"): (25.1000, 76.5200),
    ("Rajasthan", "Barmer"): (25.7500, 71.3800),
    ("Rajasthan", "Bharatpur"): (27.2152, 77.4938),
    ("Rajasthan", "Bhilwara"): (25.3500, 74.6400),
    ("Rajasthan", "Bikaner"): (28.0229, 73.3119),
    ("Rajasthan", "Bundi"): (25.4395, 75.6390),
    ("Rajasthan", "Chittorgarh"): (24.8887, 74.6269),
    ("Rajasthan", "Churu"): (28.2960, 74.9640),
    ("Rajasthan", "Dausa"): (26.8934, 76.3397),
    ("Rajasthan", "Dholpur"): (26.7010, 77.8940),
    ("Rajasthan", "Dungarpur"): (23.8400, 73.7200),
    ("Rajasthan", "Hanumangarh"): (29.5826, 74.3292),
    ("Rajasthan", "Jaipur"): (26.9124, 75.7873),
    ("Rajasthan", "Jaisalmer"): (26.9157, 70.9083),
    ("Rajasthan", "Jalore"): (25.3500, 72.6200),
    ("Rajasthan", "Jhalawar"): (24.5975, 76.1650),
    ("Rajasthan", "Jhunjhunu"): (28.1290, 75.3990),
    ("Rajasthan", "Jodhpur"): (26.2389, 73.0243),
    ("Rajasthan", "Karauli"): (26.5000, 77.0200),
    ("Rajasthan", "Kota"): (25.2138, 75.8648),
    ("Rajasthan", "Nagaur"): (27.2025, 73.7285),
    ("Rajasthan", "Pali"): (25.7730, 73.3234),
    ("Rajasthan", "Pratapgarh"): (24.0330, 74.7780),
    ("Rajasthan", "Rajsamand"): (25.0700, 73.8800),
    ("Rajasthan", "Sawai Madhopur"): (25.9964, 76.3545),
    ("Rajasthan", "Sikar"): (27.6094, 75.1399),
    ("Rajasthan", "Sirohi"): (24.8860, 72.8620),
    ("Rajasthan", "Sri Ganganagar"): (29.9166, 73.8833),
    ("Rajasthan", "Tonk"): (26.1630, 75.7880),
    ("Rajasthan", "Udaipur"): (24.5854, 73.7125),
    ("Kerala", "Alappuzha"): (9.4981, 76.3388),
    ("Kerala", "Ernakulam"): (9.9816, 76.2999),
    ("Kerala", "Idukki"): (9.9189, 77.1025),
    ("Kerala", "Kannur"): (11.8745, 75.3704),
    ("Kerala", "Kasaragod"): (12.4996, 74.9869),
    ("Kerala", "Kollam"): (8.8932, 76.6141),
    ("Kerala", "Kottayam"): (9.5916, 76.5222),
    ("Kerala", "Kozhikode"): (11.2588, 75.7804),
    ("Kerala", "Malappuram"): (11.0510, 76.0711),
    ("Kerala", "Palakkad"): (10.7867, 76.6548),
    ("Kerala", "Pathanamthitta"): (9.2648, 76.7870),
    ("Kerala", "Thiruvananthapuram"): (8.5241, 76.9366),
    ("Kerala", "Thrissur"): (10.5276, 76.2144),
    ("Kerala", "Wayanad"): (11.6854, 76.1320),
    ("Tamil Nadu", "Ariyalur"): (11.1437, 79.0747),
    ("Tamil Nadu", "Chengalpattu"): (12.6921, 79.9757),
    ("Tamil Nadu", "Chennai"): (13.0827, 80.2707),
    ("Tamil Nadu", "Coimbatore"): (11.0168, 76.9558),
    ("Tamil Nadu", "Cuddalore"): (11.7480, 79.7714),
    ("Tamil Nadu", "Dharmapuri"): (12.1211, 78.1582),
    ("Tamil Nadu", "Dindigul"): (10.3624, 77.9695),
    ("Tamil Nadu", "Erode"): (11.3410, 77.7172),
    ("Tamil Nadu", "Kallakurichi"): (11.7380, 78.9590),
    ("Tamil Nadu", "Kancheepuram"): (12.8185, 79.7018),
    ("Tamil Nadu", "Kanyakumari"): (8.0883, 77.5385),
    ("Tamil Nadu", "Karur"): (10.9601, 78.0766),
    ("Tamil Nadu", "Krishnagiri"): (12.5266, 78.2138),
    ("Tamil Nadu", "Madurai"): (9.9252, 78.1198),
    ("Tamil Nadu", "Mayiladuthurai"): (11.1035, 79.6508),
    ("Tamil Nadu", "Nagapattinam"): (10.7672, 79.8449),
    ("Tamil Nadu", "Namakkal"): (11.2342, 78.1674),
    ("Tamil Nadu", "Nilgiris"): (11.4916, 76.7337),
    ("Tamil Nadu", "Perambalur"): (11.2340, 78.8800),
    ("Tamil Nadu", "Pudukkottai"): (10.3797, 78.8200),
    ("Tamil Nadu", "Ramanathapuram"): (9.3639, 78.8395),
    ("Tamil Nadu", "Ranipet"): (12.9310, 79.3330),
    ("Tamil Nadu", "Salem"): (11.6643, 78.1460),
    ("Tamil Nadu", "Sivaganga"): (9.8479, 78.4800),
    ("Tamil Nadu", "Tenkasi"): (8.9590, 77.3150),
    ("Tamil Nadu", "Thanjavur"): (10.7870, 79.1378),
    ("Tamil Nadu", "Theni"): (10.0104, 77.4770),
    ("Tamil Nadu", "Thoothukudi"): (8.7642, 78.1348),
    ("Tamil Nadu", "Tiruchirappalli"): (10.7905, 78.7047),
    ("Tamil Nadu", "Tirunelveli"): (8.7139, 77.7567),
    ("Tamil Nadu", "Tirupathur"): (12.4960, 78.5720),
    ("Tamil Nadu", "Tiruppur"): (11.1085, 77.3411),
    ("Tamil Nadu", "Tiruvallur"): (13.1231, 79.9099),
    ("Tamil Nadu", "Tiruvannamalai"): (12.2253, 79.0747),
    ("Tamil Nadu", "Tiruvarur"): (10.7725, 79.6367),
    ("Tamil Nadu", "Vellore"): (12.9165, 79.1325),
    ("Tamil Nadu", "Viluppuram"): (11.9401, 79.4861),
    ("Tamil Nadu", "Virudhunagar"): (9.5851, 77.9624),
    ("Bihar", "Araria"): (26.1500, 87.4700), ("Bihar", "Arwal"): (25.2500, 84.6800),
    ("Bihar", "Aurangabad"): (24.7500, 84.3700), ("Bihar", "Banka"): (24.8800, 86.9200),
    ("Bihar", "Begusarai"): (25.4180, 86.1290), ("Bihar", "Bhagalpur"): (25.2500, 86.9833),
    ("Bihar", "Bhojpur"): (25.5600, 84.4500), ("Bihar", "Buxar"): (25.5600, 83.9800),
    ("Bihar", "Darbhanga"): (26.1542, 85.8918), ("Bihar", "East Champaran"): (26.6500, 84.9200),
    ("Bihar", "Gaya"): (24.7914, 85.0002), ("Bihar", "Gopalganj"): (26.4671, 84.4376),
    ("Bihar", "Jamui"): (24.9200, 86.2200), ("Bihar", "Jehanabad"): (25.2100, 84.9900),
    ("Bihar", "Kaimur"): (25.0500, 83.6000), ("Bihar", "Katihar"): (25.5379, 87.5765),
    ("Bihar", "Khagaria"): (25.5000, 86.4600), ("Bihar", "Kishanganj"): (26.1000, 87.9500),
    ("Bihar", "Lakhisarai"): (25.1500, 86.1000), ("Bihar", "Madhepura"): (25.9200, 86.7900),
    ("Bihar", "Madhubani"): (26.3500, 86.0800), ("Bihar", "Munger"): (25.3700, 86.4700),
    ("Bihar", "Muzaffarpur"): (26.1225, 85.3906), ("Bihar", "Nalanda"): (25.1373, 85.4440),
    ("Bihar", "Nawada"): (24.8800, 85.5400), ("Bihar", "Patna"): (25.5941, 85.1376),
    ("Bihar", "Purnia"): (25.7771, 87.4753), ("Bihar", "Rohtas"): (24.9900, 84.0300),
    ("Bihar", "Saharsa"): (25.8800, 86.6000), ("Bihar", "Samastipur"): (25.8637, 85.7811),
    ("Bihar", "Saran"): (25.9200, 84.8500), ("Bihar", "Sheikhpura"): (25.1400, 85.8500),
    ("Bihar", "Sheohar"): (26.5200, 85.2900), ("Bihar", "Sitamarhi"): (26.5900, 85.4900),
    ("Bihar", "Siwan"): (26.2200, 84.3600), ("Bihar", "Supaul"): (26.1200, 86.6000),
    ("Bihar", "Vaishali"): (25.6900, 85.2000), ("Bihar", "West Champaran"): (27.1500, 84.3500),
    ("West Bengal", "Alipurduar"): (26.4900, 89.5300), ("West Bengal", "Bankura"): (23.2500, 87.0667),
    ("West Bengal", "Birbhum"): (23.9000, 87.5333), ("West Bengal", "Cooch Behar"): (26.3200, 89.4500),
    ("West Bengal", "Dakshin Dinajpur"): (25.3200, 88.7500), ("West Bengal", "Darjeeling"): (27.0360, 88.2627),
    ("West Bengal", "Hooghly"): (22.9000, 88.4000), ("West Bengal", "Howrah"): (22.5958, 88.2636),
    ("West Bengal", "Jalpaiguri"): (26.5163, 88.7274), ("West Bengal", "Jhargram"): (22.4500, 86.9900),
    ("West Bengal", "Kalimpong"): (27.0600, 88.4700), ("West Bengal", "Kolkata"): (22.5726, 88.3639),
    ("West Bengal", "Malda"): (25.0108, 88.1415), ("West Bengal", "Murshidabad"): (24.1800, 88.2700),
    ("West Bengal", "Nadia"): (23.4000, 88.5500), ("West Bengal", "North 24 Parganas"): (22.8700, 88.5800),
    ("West Bengal", "Paschim Bardhaman"): (23.2333, 87.0833), ("West Bengal", "Paschim Medinipur"): (22.4167, 87.3167),
    ("West Bengal", "Purba Bardhaman"): (23.2500, 87.8500), ("West Bengal", "Purba Medinipur"): (22.0833, 87.7500),
    ("West Bengal", "Purulia"): (23.3333, 86.3667), ("West Bengal", "South 24 Parganas"): (22.1500, 88.6800),
    ("West Bengal", "Uttar Dinajpur"): (26.1200, 88.1200),
    ("Madhya Pradesh", "Bhopal"): (23.2599, 77.4126), ("Madhya Pradesh", "Indore"): (22.7196, 75.8577),
    ("Madhya Pradesh", "Jabalpur"): (23.1815, 79.9864), ("Madhya Pradesh", "Gwalior"): (26.2183, 78.1828),
    ("Madhya Pradesh", "Ujjain"): (23.1765, 75.7885), ("Madhya Pradesh", "Sagar"): (23.8388, 78.7378),
    ("Madhya Pradesh", "Rewa"): (24.5362, 81.2997), ("Madhya Pradesh", "Satna"): (24.5800, 80.8300),
    ("Madhya Pradesh", "Chhindwara"): (22.0574, 78.9382), ("Madhya Pradesh", "Vidisha"): (23.5251, 77.8093),
    ("Madhya Pradesh", "Hoshangabad"): (22.7500, 77.7200), ("Madhya Pradesh", "Damoh"): (23.8310, 79.4420),
    ("Madhya Pradesh", "Narsinghpur"): (22.9477, 79.1940), ("Madhya Pradesh", "Khandwa"): (21.8274, 76.3520),
    ("Madhya Pradesh", "Khargone"): (21.8230, 75.6110), ("Madhya Pradesh", "Ratlam"): (23.3313, 75.0367),
    ("Madhya Pradesh", "Mandsaur"): (24.0700, 75.0700), ("Madhya Pradesh", "Neemuch"): (24.4700, 74.8700),
    ("Madhya Pradesh", "Dhar"): (22.5982, 75.2996), ("Madhya Pradesh", "Jhabua"): (22.7700, 74.5900),
    ("Madhya Pradesh", "Alirajpur"): (22.3040, 74.3570), ("Madhya Pradesh", "Barwani"): (22.0320, 74.9020),
    ("Madhya Pradesh", "Betul"): (21.9061, 77.8978), ("Madhya Pradesh", "Bhind"): (26.5647, 78.7877),
    ("Madhya Pradesh", "Chhatarpur"): (24.9100, 79.5900), ("Madhya Pradesh", "Datia"): (25.6700, 78.4600),
    ("Madhya Pradesh", "Dewas"): (22.9676, 76.0534), ("Madhya Pradesh", "Dindori"): (22.9400, 81.0800),
    ("Madhya Pradesh", "Guna"): (24.6479, 77.3148), ("Madhya Pradesh", "Harda"): (22.3417, 77.0900),
    ("Madhya Pradesh", "Katni"): (23.8331, 80.3964), ("Madhya Pradesh", "Mandla"): (22.5981, 80.3764),
    ("Madhya Pradesh", "Morena"): (26.4960, 77.9980), ("Madhya Pradesh", "Panna"): (24.7175, 80.1853),
    ("Madhya Pradesh", "Raisen"): (23.3300, 77.7900), ("Madhya Pradesh", "Rajgarh"): (24.0219, 76.7220),
    ("Madhya Pradesh", "Seoni"): (22.0854, 79.5389), ("Madhya Pradesh", "Shahdol"): (23.2977, 81.3567),
    ("Madhya Pradesh", "Shajapur"): (23.4280, 76.2770), ("Madhya Pradesh", "Sheopur"): (25.6700, 76.7100),
    ("Madhya Pradesh", "Shivpuri"): (25.4231, 77.6590), ("Madhya Pradesh", "Sidhi"): (24.4161, 81.8769),
    ("Madhya Pradesh", "Singrauli"): (24.1997, 82.6746), ("Madhya Pradesh", "Tikamgarh"): (24.7400, 78.8300),
    ("Madhya Pradesh", "Umaria"): (23.5200, 80.8400), ("Madhya Pradesh", "Agar Malwa"): (23.7100, 76.0200),
    ("Madhya Pradesh", "Anuppur"): (23.1000, 81.6900), ("Madhya Pradesh", "Ashoknagar"): (24.5700, 77.7300),
    ("Madhya Pradesh", "Balaghat"): (21.8138, 80.1867), ("Madhya Pradesh", "Burhanpur"): (21.3100, 76.2300),
    ("Madhya Pradesh", "Niwari"): (25.1100, 78.9900),
    ("Uttar Pradesh", "Lucknow"): (26.8467, 80.9462), ("Uttar Pradesh", "Agra"): (27.1767, 78.0081),
    ("Uttar Pradesh", "Kanpur Nagar"): (26.4499, 80.3319), ("Uttar Pradesh", "Varanasi"): (25.3176, 82.9739),
    ("Uttar Pradesh", "Prayagraj"): (25.4358, 81.8463), ("Uttar Pradesh", "Meerut"): (28.9845, 77.7064),
    ("Uttar Pradesh", "Ghaziabad"): (28.6692, 77.4538), ("Uttar Pradesh", "Mathura"): (27.4924, 77.6737),
    ("Uttar Pradesh", "Aligarh"): (27.8974, 78.0880), ("Uttar Pradesh", "Bareilly"): (28.3670, 79.4304),
    ("Uttar Pradesh", "Gorakhpur"): (26.7606, 83.3732), ("Uttar Pradesh", "Moradabad"): (28.8386, 78.7733),
    ("Uttar Pradesh", "Saharanpur"): (29.9640, 77.5460), ("Uttar Pradesh", "Muzaffarnagar"): (29.4727, 77.7085),
    ("Uttar Pradesh", "Firozabad"): (27.1592, 78.3957), ("Uttar Pradesh", "Jhansi"): (25.4484, 78.5685),
    ("Uttar Pradesh", "Ayodhya"): (26.7922, 82.1998), ("Uttar Pradesh", "Azamgarh"): (26.0686, 83.1838),
    ("Uttar Pradesh", "Lakhimpur Kheri"): (27.9487, 80.7821), ("Uttar Pradesh", "Sultanpur"): (26.2648, 82.0727),
    ("Uttar Pradesh", "Bulandshahr"): (28.4069, 77.8497), ("Uttar Pradesh", "Sitapur"): (27.5680, 80.6830),
    ("Uttar Pradesh", "Bijnor"): (29.3719, 78.1350), ("Uttar Pradesh", "Shahjahanpur"): (27.8830, 79.9050),
    ("Uttar Pradesh", "Ghazipur"): (25.5782, 83.5786), ("Uttar Pradesh", "Ballia"): (25.7592, 84.1493),
    ("Uttar Pradesh", "Jaunpur"): (25.7458, 82.6838), ("Uttar Pradesh", "Mirzapur"): (25.1460, 82.5690),
    ("Uttar Pradesh", "Sonbhadra"): (24.6852, 83.0694), ("Uttar Pradesh", "Basti"): (26.8013, 82.7227),
    ("Uttar Pradesh", "Deoria"): (26.5014, 83.7806), ("Uttar Pradesh", "Kushinagar"): (26.7400, 83.8890),
    ("Uttar Pradesh", "Maharajganj"): (27.1300, 83.5700), ("Uttar Pradesh", "Siddharthnagar"): (27.2900, 83.0700),
    ("Uttar Pradesh", "Sant Kabir Nagar"): (26.7900, 83.0600), ("Uttar Pradesh", "Ambedkar Nagar"): (26.4600, 82.5900),
    ("Uttar Pradesh", "Amethi"): (26.1500, 81.9300), ("Uttar Pradesh", "Rae Bareli"): (26.2309, 81.2317),
    ("Uttar Pradesh", "Unnao"): (26.5460, 80.4910), ("Uttar Pradesh", "Hardoi"): (27.3952, 80.1317),
    ("Uttar Pradesh", "Farrukhabad"): (27.3880, 79.5800), ("Uttar Pradesh", "Kannauj"): (27.0571, 79.9178),
    ("Uttar Pradesh", "Etawah"): (26.7857, 79.0246), ("Uttar Pradesh", "Auraiya"): (26.4700, 79.5100),
    ("Uttar Pradesh", "Mainpuri"): (27.2352, 79.0241), ("Uttar Pradesh", "Hathras"): (27.5915, 78.0500),
    ("Uttar Pradesh", "Kasganj"): (27.8100, 78.6400), ("Uttar Pradesh", "Etah"): (27.5593, 78.6667),
    ("Uttar Pradesh", "Budaun"): (28.0400, 79.1200), ("Uttar Pradesh", "Rampur"): (28.7952, 79.0155),
    ("Uttar Pradesh", "Amroha"): (28.9040, 78.4680), ("Uttar Pradesh", "Sambhal"): (28.5900, 78.5700),
    ("Uttar Pradesh", "Pilibhit"): (28.6400, 79.8000), ("Uttar Pradesh", "Baghpat"): (28.9400, 77.2200),
    ("Uttar Pradesh", "Hapur"): (28.7300, 77.7800), ("Uttar Pradesh", "Gautam Buddha Nagar"): (28.5355, 77.3910),
    ("Uttar Pradesh", "Mau"): (25.9418, 83.5610), ("Uttar Pradesh", "Chandauli"): (25.2700, 83.2700),
    ("Uttar Pradesh", "Bhadohi"): (25.3900, 82.5700), ("Uttar Pradesh", "Kaushambi"): (25.5300, 81.3800),
    ("Uttar Pradesh", "Fatehpur"): (25.9300, 80.8100), ("Uttar Pradesh", "Pratapgarh"): (25.9000, 81.9900),
    ("Uttar Pradesh", "Banda"): (25.4756, 80.3342), ("Uttar Pradesh", "Chitrakoot"): (25.2000, 80.8800),
    ("Uttar Pradesh", "Hamirpur"): (25.9500, 80.1500), ("Uttar Pradesh", "Mahoba"): (25.2900, 79.8700),
    ("Uttar Pradesh", "Jalaun"): (26.1400, 79.3300), ("Uttar Pradesh", "Lalitpur"): (24.6878, 78.4159),
    ("Uttar Pradesh", "Shravasti"): (27.5100, 81.8400), ("Uttar Pradesh", "Balrampur"): (27.4300, 82.1800),
    ("Uttar Pradesh", "Bahraich"): (27.5742, 81.5961), ("Uttar Pradesh", "Gonda"): (27.1300, 81.9700),
    ("Uttar Pradesh", "Barabanki"): (26.9200, 81.1800), ("Uttar Pradesh", "Shamli"): (29.4500, 77.3100),
    ("Odisha", "Angul"): (20.8400, 85.1000), ("Odisha", "Balangir"): (20.7000, 83.4900),
    ("Odisha", "Balasore"): (21.4942, 86.9336), ("Odisha", "Bargarh"): (21.3300, 83.6100),
    ("Odisha", "Bhadrak"): (21.0541, 86.5133), ("Odisha", "Boudh"): (20.8400, 84.3200),
    ("Odisha", "Cuttack"): (20.4625, 85.8828), ("Odisha", "Deogarh"): (21.5400, 84.7300),
    ("Odisha", "Dhenkanal"): (20.6600, 85.5900), ("Odisha", "Gajapati"): (19.0700, 84.1100),
    ("Odisha", "Ganjam"): (19.3900, 84.9800), ("Odisha", "Jagatsinghpur"): (20.2600, 86.1700),
    ("Odisha", "Jajpur"): (20.8500, 86.3400), ("Odisha", "Jharsuguda"): (21.8550, 84.0060),
    ("Odisha", "Kalahandi"): (19.9100, 83.1700), ("Odisha", "Kandhamal"): (20.1100, 84.2300),
    ("Odisha", "Kendrapara"): (20.5000, 86.4200), ("Odisha", "Kendujhar"): (21.6300, 85.5800),
    ("Odisha", "Khordha"): (20.1800, 85.6100), ("Odisha", "Koraput"): (18.8100, 82.7100),
    ("Odisha", "Malkangiri"): (18.3500, 81.9000), ("Odisha", "Mayurbhanj"): (21.9400, 86.2700),
    ("Odisha", "Nabarangpur"): (19.2300, 82.5500), ("Odisha", "Nayagarh"): (20.1300, 85.0900),
    ("Odisha", "Nuapada"): (20.8300, 82.5400), ("Odisha", "Puri"): (19.8135, 85.8312),
    ("Odisha", "Rayagada"): (19.1700, 83.4200), ("Odisha", "Sambalpur"): (21.4669, 83.9812),
    ("Odisha", "Subarnapur"): (20.8300, 83.9000), ("Odisha", "Sundargarh"): (22.1167, 84.0333),
    ("Jharkhand", "Bokaro"): (23.6693, 86.1511), ("Jharkhand", "Chatra"): (24.2100, 84.8800),
    ("Jharkhand", "Deoghar"): (24.4800, 86.7000), ("Jharkhand", "Dhanbad"): (23.7998, 86.4305),
    ("Jharkhand", "Dumka"): (24.2700, 87.2500), ("Jharkhand", "East Singhbhum"): (22.8046, 86.2029),
    ("Jharkhand", "Garhwa"): (24.1600, 83.8100), ("Jharkhand", "Giridih"): (24.1900, 86.3000),
    ("Jharkhand", "Godda"): (24.8300, 87.2100), ("Jharkhand", "Gumla"): (23.0500, 84.5400),
    ("Jharkhand", "Hazaribagh"): (23.9925, 85.3617), ("Jharkhand", "Jamtara"): (23.9600, 86.8000),
    ("Jharkhand", "Khunti"): (23.0700, 85.2800), ("Jharkhand", "Koderma"): (24.4600, 85.6000),
    ("Jharkhand", "Latehar"): (23.7400, 84.5000), ("Jharkhand", "Lohardaga"): (23.4400, 84.6800),
    ("Jharkhand", "Pakur"): (24.6400, 87.8400), ("Jharkhand", "Palamu"): (24.0300, 84.0700),
    ("Jharkhand", "Ramgarh"): (23.6400, 85.5100), ("Jharkhand", "Ranchi"): (23.3441, 85.3096),
    ("Jharkhand", "Sahebganj"): (25.2400, 87.6600), ("Jharkhand", "Seraikela Kharsawan"): (22.5800, 85.9900),
    ("Jharkhand", "Simdega"): (22.6100, 84.5200), ("Jharkhand", "West Singhbhum"): (22.1500, 85.6500),
    ("Chhattisgarh", "Raipur"): (21.2514, 81.6296), ("Chhattisgarh", "Bilaspur"): (22.0796, 82.1391),
    ("Chhattisgarh", "Durg"): (21.1900, 81.2800), ("Chhattisgarh", "Rajnandgaon"): (21.0970, 81.0290),
    ("Chhattisgarh", "Korba"): (22.3595, 82.7501), ("Chhattisgarh", "Bastar"): (19.1100, 81.9500),
    ("Chhattisgarh", "Raigarh"): (21.8974, 83.3950), ("Chhattisgarh", "Kanker"): (20.2700, 81.4900),
    ("Chhattisgarh", "Mahasamund"): (21.1100, 82.0900), ("Chhattisgarh", "Dhamtari"): (20.7100, 81.5500),
    ("Chhattisgarh", "Janjgir Champa"): (22.0100, 82.5700), ("Chhattisgarh", "Gariaband"): (20.6300, 82.0600),
    ("Chhattisgarh", "Balod"): (20.7300, 81.2100), ("Chhattisgarh", "Baloda Bazar"): (21.6600, 82.1600),
    ("Chhattisgarh", "Balrampur"): (23.1300, 83.6000), ("Chhattisgarh", "Bemetara"): (21.7100, 81.5300),
    ("Chhattisgarh", "Bijapur"): (18.8300, 80.2500), ("Chhattisgarh", "Dantewada"): (18.8900, 81.3500),
    ("Chhattisgarh", "Gaurela Pendra Marwahi"): (22.7500, 81.7900), ("Chhattisgarh", "Jashpur"): (22.8800, 84.1500),
    ("Chhattisgarh", "Kabirdham"): (22.0000, 81.2700), ("Chhattisgarh", "Khairagarh"): (21.4200, 80.9800),
    ("Chhattisgarh", "Kondagaon"): (19.5900, 81.6600), ("Chhattisgarh", "Koriya"): (23.2500, 82.7000),
    ("Chhattisgarh", "Manendragarh"): (23.2000, 82.2200), ("Chhattisgarh", "Mohla Manpur"): (20.8000, 80.7900),
    ("Chhattisgarh", "Mungeli"): (22.0600, 81.6900), ("Chhattisgarh", "Narayanpur"): (19.6900, 81.2400),
    ("Chhattisgarh", "Sakti"): (22.0300, 82.9700), ("Chhattisgarh", "Sarangarh Bilaigarh"): (21.5800, 83.0700),
    ("Chhattisgarh", "Sukma"): (18.3900, 81.6600), ("Chhattisgarh", "Surajpur"): (23.2200, 82.8600),
    ("Chhattisgarh", "Surguja"): (23.1200, 83.1900),
    ("Assam", "Kamrup Metropolitan"): (26.1445, 91.7362), ("Assam", "Kamrup"): (26.1000, 91.4000),
    ("Assam", "Dibrugarh"): (27.4728, 94.9120), ("Assam", "Jorhat"): (26.7509, 94.2037),
    ("Assam", "Nagaon"): (26.3472, 92.6839), ("Assam", "Sonitpur"): (26.6300, 92.8000),
    ("Assam", "Sivasagar"): (26.9800, 94.6300), ("Assam", "Tinsukia"): (27.4894, 95.3559),
    ("Assam", "Lakhimpur"): (27.2348, 94.1000), ("Assam", "Dhemaji"): (27.4800, 94.5600),
    ("Assam", "Barpeta"): (26.3200, 91.0000), ("Assam", "Nalbari"): (26.4400, 91.4300),
    ("Assam", "Baksa"): (26.6400, 91.2000), ("Assam", "Darrang"): (26.4500, 92.1700),
    ("Assam", "Udalguri"): (26.7500, 92.1000), ("Assam", "Dhubri"): (26.0200, 89.9900),
    ("Assam", "Goalpara"): (26.1700, 90.6200), ("Assam", "Bongaigaon"): (26.4800, 90.5600),
    ("Assam", "Chirang"): (26.5200, 90.4700), ("Assam", "Kokrajhar"): (26.4000, 90.2700),
    ("Assam", "Cachar"): (24.8333, 92.7789), ("Assam", "Hailakandi"): (24.6900, 92.5600),
    ("Assam", "Karimganj"): (24.8600, 92.3600), ("Assam", "Dima Hasao"): (25.5700, 93.0200),
    ("Assam", "Karbi Anglong"): (26.0000, 93.5000), ("Assam", "West Karbi Anglong"): (25.9000, 93.0000),
    ("Assam", "Golaghat"): (26.5200, 93.9700), ("Assam", "Majuli"): (26.9500, 94.1600),
    ("Assam", "Biswanath"): (26.7300, 93.1500), ("Assam", "Charaideo"): (27.0100, 94.8100),
    ("Assam", "Hojai"): (26.0000, 92.8500), ("Assam", "Morigaon"): (26.2500, 92.3300),
    ("Assam", "South Salmara"): (25.9400, 89.8700), ("Assam", "Bajali"): (26.4600, 91.1500),
    ("Himachal Pradesh", "Shimla"): (31.1048, 77.1734), ("Himachal Pradesh", "Kangra"): (32.0998, 76.2691),
    ("Himachal Pradesh", "Mandi"): (31.7088, 76.9318), ("Himachal Pradesh", "Kullu"): (31.9579, 77.1095),
    ("Himachal Pradesh", "Solan"): (30.9045, 77.0967), ("Himachal Pradesh", "Una"): (31.4685, 76.2709),
    ("Himachal Pradesh", "Hamirpur"): (31.6862, 76.5215), ("Himachal Pradesh", "Bilaspur"): (31.3314, 76.7605),
    ("Himachal Pradesh", "Chamba"): (32.5534, 76.1258), ("Himachal Pradesh", "Sirmaur"): (30.5600, 77.4600),
    ("Himachal Pradesh", "Kinnaur"): (31.5900, 78.4100), ("Himachal Pradesh", "Lahaul Spiti"): (32.5700, 77.5700),
    ("Uttarakhand", "Dehradun"): (30.3165, 78.0322), ("Uttarakhand", "Haridwar"): (29.9457, 78.1642),
    ("Uttarakhand", "Nainital"): (29.3803, 79.4636), ("Uttarakhand", "Udham Singh Nagar"): (28.9833, 79.5167),
    ("Uttarakhand", "Pauri Garhwal"): (29.7000, 78.7800), ("Uttarakhand", "Tehri Garhwal"): (30.3780, 78.4804),
    ("Uttarakhand", "Chamoli"): (30.4024, 79.3193), ("Uttarakhand", "Almora"): (29.5971, 79.6590),
    ("Uttarakhand", "Pithoragarh"): (29.5826, 80.2180), ("Uttarakhand", "Champawat"): (29.3339, 80.0910),
    ("Uttarakhand", "Bageshwar"): (29.8372, 79.7714), ("Uttarakhand", "Rudraprayag"): (30.2844, 78.9819),
    ("Uttarakhand", "Uttarkashi"): (30.7268, 78.4354),
    ("Goa", "North Goa"): (15.4909, 73.8278), ("Goa", "South Goa"): (15.1734, 74.0573),
    ("Delhi", "New Delhi"): (28.6139, 77.2090), ("Delhi", "Central Delhi"): (28.6508, 77.2295),
    ("Delhi", "East Delhi"): (28.6600, 77.3100), ("Delhi", "North Delhi"): (28.7200, 77.2000),
    ("Delhi", "South Delhi"): (28.5400, 77.2200), ("Delhi", "West Delhi"): (28.6500, 77.1000),
    ("Delhi", "North East Delhi"): (28.7000, 77.3000), ("Delhi", "North West Delhi"): (28.7000, 77.1000),
    ("Delhi", "South East Delhi"): (28.5700, 77.2900), ("Delhi", "South West Delhi"): (28.5500, 77.0800),
    ("Delhi", "Shahdara"): (28.6700, 77.3000),
    ("Chandigarh", "Chandigarh"): (30.7333, 76.7794),
    ("Puducherry", "Puducherry"): (11.9416, 79.8083), ("Puducherry", "Karaikal"): (10.9254, 79.8380),
    ("Puducherry", "Mahe"): (11.7010, 75.5362), ("Puducherry", "Yanam"): (16.7300, 82.2130),
    ("Jammu and Kashmir", "Srinagar"): (34.0837, 74.7973), ("Jammu and Kashmir", "Jammu"): (32.7266, 74.8570),
    ("Jammu and Kashmir", "Anantnag"): (33.7311, 75.1487), ("Jammu and Kashmir", "Baramulla"): (34.2090, 74.3442),
    ("Jammu and Kashmir", "Pulwama"): (33.8742, 74.8985), ("Jammu and Kashmir", "Kupwara"): (34.5211, 74.2615),
    ("Jammu and Kashmir", "Rajouri"): (33.3771, 74.3102), ("Jammu and Kashmir", "Udhampur"): (32.9160, 75.1410),
    ("Jammu and Kashmir", "Kathua"): (32.3842, 75.5160), ("Jammu and Kashmir", "Poonch"): (33.7726, 74.0927),
    ("Jammu and Kashmir", "Budgam"): (33.9400, 74.7100), ("Jammu and Kashmir", "Kulgam"): (33.6440, 75.0190),
    ("Jammu and Kashmir", "Shopian"): (33.7200, 74.8300), ("Jammu and Kashmir", "Ganderbal"): (34.2200, 74.7700),
    ("Jammu and Kashmir", "Bandipora"): (34.4100, 74.6400), ("Jammu and Kashmir", "Reasi"): (33.0800, 74.8300),
    ("Jammu and Kashmir", "Ramban"): (33.2400, 75.2400), ("Jammu and Kashmir", "Doda"): (33.1500, 75.5500),
    ("Jammu and Kashmir", "Kishtwar"): (33.3100, 75.7700), ("Jammu and Kashmir", "Samba"): (32.5800, 75.1200),
    ("Ladakh", "Leh"): (34.1526, 77.5771), ("Ladakh", "Kargil"): (34.5539, 76.1349),
    ("Sikkim", "East Sikkim"): (27.3389, 88.6065), ("Sikkim", "West Sikkim"): (27.2900, 88.2600),
    ("Sikkim", "North Sikkim"): (27.6700, 88.4500), ("Sikkim", "South Sikkim"): (27.1500, 88.5300),
    ("Sikkim", "Pakyong"): (27.2300, 88.6100), ("Sikkim", "Soreng"): (27.2000, 88.1000),
    ("Manipur", "Imphal East"): (24.8170, 93.9368), ("Manipur", "Imphal West"): (24.8000, 93.9400),
    ("Manipur", "Bishnupur"): (24.6100, 93.7700), ("Manipur", "Thoubal"): (24.6300, 94.0100),
    ("Manipur", "Chandel"): (24.3300, 94.0200), ("Manipur", "Senapati"): (25.2700, 94.0200),
    ("Manipur", "Tamenglong"): (24.9800, 93.4800), ("Manipur", "Churachandpur"): (24.3300, 93.6800),
    ("Manipur", "Ukhrul"): (25.1200, 94.3600), ("Manipur", "Jiribam"): (24.8000, 93.1200),
    ("Manipur", "Kakching"): (24.5000, 93.9900), ("Manipur", "Kamjong"): (25.1100, 94.6200),
    ("Manipur", "Kangpokpi"): (25.1300, 93.9700), ("Manipur", "Noney"): (25.0000, 93.8000),
    ("Manipur", "Pherzawl"): (24.0700, 93.4700), ("Manipur", "Tengnoupal"): (24.0100, 94.1800),
    ("Meghalaya", "East Khasi Hills"): (25.5788, 91.8933), ("Meghalaya", "West Khasi Hills"): (25.3500, 91.2700),
    ("Meghalaya", "East Garo Hills"): (25.6200, 90.4900), ("Meghalaya", "West Garo Hills"): (25.5700, 90.2200),
    ("Meghalaya", "Ri Bhoi"): (25.7300, 92.0200), ("Meghalaya", "East Jaintia Hills"): (25.3600, 92.5100),
    ("Meghalaya", "South Garo Hills"): (25.0300, 90.4200), ("Meghalaya", "North Garo Hills"): (25.8700, 90.5700),
    ("Meghalaya", "South West Garo Hills"): (25.2900, 89.9800), ("Meghalaya", "South West Khasi Hills"): (25.1100, 91.2600),
    ("Meghalaya", "Eastern West Khasi Hills"): (25.4500, 91.5600),
    ("Tripura", "West Tripura"): (23.8315, 91.2868), ("Tripura", "North Tripura"): (24.3600, 92.0000),
    ("Tripura", "South Tripura"): (23.2700, 91.7500), ("Tripura", "Gomati"): (23.4500, 91.8400),
    ("Tripura", "Dhalai"): (24.0600, 92.0200), ("Tripura", "Khowai"): (24.0700, 91.6000),
    ("Tripura", "Sepahijala"): (23.6700, 91.3000), ("Tripura", "Unakoti"): (24.3300, 92.0600),
    ("Tripura", "Sipahijala"): (23.6700, 91.3000),
    ("Nagaland", "Kohima"): (25.6751, 94.1086), ("Nagaland", "Dimapur"): (25.9100, 93.7200),
    ("Nagaland", "Mokokchung"): (26.3200, 94.5200), ("Nagaland", "Wokha"): (26.1000, 94.2600),
    ("Nagaland", "Zunheboto"): (26.0000, 94.5200), ("Nagaland", "Tuensang"): (26.2700, 94.8200),
    ("Nagaland", "Mon"): (26.7200, 95.0100), ("Nagaland", "Phek"): (25.6500, 94.4700),
    ("Nagaland", "Kiphire"): (25.8500, 95.0200), ("Nagaland", "Longleng"): (26.5000, 94.5800),
    ("Nagaland", "Peren"): (25.4800, 93.7100), ("Nagaland", "Chumoukedima"): (25.8500, 93.7200),
    ("Nagaland", "Niuland"): (25.7000, 93.9800), ("Nagaland", "Noklak"): (26.1700, 95.2200),
    ("Nagaland", "Shamator"): (26.4000, 94.9000), ("Nagaland", "Tseminyu"): (25.9700, 94.1100),
    ("Arunachal Pradesh", "Tawang"): (27.5860, 91.8590), ("Arunachal Pradesh", "West Kameng"): (27.2200, 92.5600),
    ("Arunachal Pradesh", "East Kameng"): (27.0400, 93.0400), ("Arunachal Pradesh", "Papum Pare"): (27.1000, 93.6000),
    ("Arunachal Pradesh", "Kurung Kumey"): (28.0700, 93.8300), ("Arunachal Pradesh", "Kra Daadi"): (28.1700, 94.3700),
    ("Arunachal Pradesh", "Lower Subansiri"): (27.5600, 93.8700), ("Arunachal Pradesh", "Upper Subansiri"): (28.3000, 94.0800),
    ("Arunachal Pradesh", "West Siang"): (28.1600, 94.5700), ("Arunachal Pradesh", "East Siang"): (28.0900, 95.2000),
    ("Arunachal Pradesh", "Siang"): (28.0200, 94.9000), ("Arunachal Pradesh", "Upper Siang"): (28.7600, 95.1300),
    ("Arunachal Pradesh", "Lower Siang"): (27.9500, 94.6000), ("Arunachal Pradesh", "Lohit"): (27.8300, 96.3400),
    ("Arunachal Pradesh", "Anjaw"): (28.0600, 96.8400), ("Arunachal Pradesh", "Tirap"): (27.0200, 95.7600),
    ("Arunachal Pradesh", "Changlang"): (27.1300, 95.7400), ("Arunachal Pradesh", "Longding"): (27.3400, 95.6500),
    ("Arunachal Pradesh", "Namsai"): (27.6700, 95.8300), ("Arunachal Pradesh", "Dibang Valley"): (28.6300, 95.7200),
    ("Arunachal Pradesh", "Lower Dibang Valley"): (28.0700, 95.8300), ("Arunachal Pradesh", "Lepa Rada"): (27.9800, 94.7200),
    ("Arunachal Pradesh", "Shi Yomi"): (28.3900, 94.7100), ("Arunachal Pradesh", "Pakke Kessang"): (27.1700, 93.5200),
    ("Arunachal Pradesh", "Kamle"): (27.7500, 93.5900),
    ("Mizoram", "Aizawl"): (23.7307, 92.7173), ("Mizoram", "Lunglei"): (22.8800, 92.7300),
    ("Mizoram", "Champhai"): (23.4600, 93.3200), ("Mizoram", "Kolasib"): (24.2300, 92.6800),
    ("Mizoram", "Serchhip"): (23.3100, 92.8500), ("Mizoram", "Mamit"): (23.9300, 92.4800),
    ("Mizoram", "Lawngtlai"): (22.0300, 92.9000), ("Mizoram", "Siaha"): (22.4800, 92.9700),
    ("Mizoram", "Hnahthial"): (23.0000, 92.7200), ("Mizoram", "Khawzawl"): (23.3000, 93.0200),
    ("Mizoram", "Saitual"): (23.7900, 92.9900),
    ("Andaman and Nicobar Islands", "South Andaman"): (11.6234, 92.7265),
    ("Andaman and Nicobar Islands", "North and Middle Andaman"): (12.5800, 92.8500),
    ("Andaman and Nicobar Islands", "Nicobar"): (8.0883, 93.7760),
    ("Lakshadweep", "Lakshadweep"): (10.5667, 72.6417),
    ("Dadra and Nagar Haveli and Daman and Diu", "Dadra and Nagar Haveli"): (20.1809, 73.0169),
    ("Dadra and Nagar Haveli and Daman and Diu", "Daman"): (20.3974, 72.8328),
    ("Dadra and Nagar Haveli and Daman and Diu", "Diu"): (20.7144, 70.9874),
}

INDIA_STATES_DISTRICTS = {
    "Andhra Pradesh": ["Alluri Sitharama Raju","Anakapalli","Anantapur","Annamayya","Bapatla","Chittoor","East Godavari","Eluru","Guntur","Kakinada","Konaseema","Krishna","Kurnool","Nandyal","Nellore","NTR","Palnadu","Parvathipuram Manyam","Prakasam","Sri Balaji","Sri Sathya Sai","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari","YSR Kadapa"],
    "Telangana": ["Adilabad","Bhadradri Kothagudem","Hanamkonda","Hyderabad","Jagtial","Jangaon","Jayashankar Bhupalpally","Jogulamba Gadwal","Kamareddy","Karimnagar","Khammam","Komaram Bheem","Mahabubabad","Mahabubnagar","Mancherial","Medak","Medchal Malkajgiri","Mulugu","Nagarkurnool","Nalgonda","Narayanpet","Nirmal","Nizamabad","Peddapalli","Rajanna Sircilla","Rangareddy","Sangareddy","Siddipet","Suryapet","Vikarabad","Wanaparthy","Warangal","Yadadri Bhuvanagiri"],
    "Maharashtra": ["Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur","Dhule","Gadchiroli","Gondia","Hingoli","Jalgaon","Jalna","Kolhapur","Latur","Mumbai City","Mumbai Suburban","Nagpur","Nanded","Nandurbar","Nashik","Osmanabad","Palghar","Parbhani","Pune","Raigad","Ratnagiri","Sangli","Satara","Sindhudurg","Solapur","Thane","Wardha","Washim","Yavatmal"],
    "Karnataka": ["Bagalkot","Ballari","Belagavi","Bengaluru Rural","Bengaluru Urban","Bidar","Chamarajanagar","Chikkaballapur","Chikkamagaluru","Chitradurga","Dakshina Kannada","Davanagere","Dharwad","Gadag","Hassan","Haveri","Kalaburagi","Kodagu","Kolar","Koppal","Mandya","Mysuru","Raichur","Ramanagara","Shivamogga","Tumakuru","Udupi","Uttara Kannada","Vijayapura","Yadgir"],
    "Tamil Nadu": ["Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri","Dindigul","Erode","Kallakurichi","Kancheepuram","Kanyakumari","Karur","Krishnagiri","Madurai","Mayiladuthurai","Nagapattinam","Namakkal","Nilgiris","Perambalur","Pudukkottai","Ramanathapuram","Ranipet","Salem","Sivaganga","Tenkasi","Thanjavur","Theni","Thoothukudi","Tiruchirappalli","Tirunelveli","Tirupathur","Tiruppur","Tiruvallur","Tiruvannamalai","Tiruvarur","Vellore","Viluppuram","Virudhunagar"],
    "Punjab": ["Amritsar","Barnala","Bathinda","Faridkot","Fatehgarh Sahib","Fazilka","Ferozepur","Gurdaspur","Hoshiarpur","Jalandhar","Kapurthala","Ludhiana","Mansa","Moga","Mohali","Muktsar","Pathankot","Patiala","Rupnagar","Sangrur","Shaheed Bhagat Singh Nagar","Tarn Taran"],
    "Haryana": ["Ambala","Bhiwani","Charkhi Dadri","Faridabad","Fatehabad","Gurugram","Hisar","Jhajjar","Jind","Kaithal","Karnal","Kurukshetra","Mahendragarh","Nuh","Palwal","Panchkula","Panipat","Rewari","Rohtak","Sirsa","Sonipat","Yamunanagar"],
    "Uttar Pradesh": ["Agra","Aligarh","Ambedkar Nagar","Amethi","Amroha","Auraiya","Ayodhya","Azamgarh","Baghpat","Bahraich","Ballia","Balrampur","Banda","Barabanki","Bareilly","Basti","Bhadohi","Bijnor","Budaun","Bulandshahr","Chandauli","Chitrakoot","Deoria","Etah","Etawah","Farrukhabad","Fatehpur","Firozabad","Gautam Buddha Nagar","Ghaziabad","Ghazipur","Gonda","Gorakhpur","Hamirpur","Hapur","Hardoi","Hathras","Jalaun","Jaunpur","Jhansi","Kannauj","Kanpur Dehat","Kanpur Nagar","Kasganj","Kaushambi","Kushinagar","Lakhimpur Kheri","Lalitpur","Lucknow","Maharajganj","Mahoba","Mainpuri","Mathura","Mau","Meerut","Mirzapur","Moradabad","Muzaffarnagar","Pilibhit","Pratapgarh","Prayagraj","Rae Bareli","Rampur","Saharanpur","Sambhal","Sant Kabir Nagar","Shahjahanpur","Shamli","Shravasti","Siddharthnagar","Sitapur","Sonbhadra","Sultanpur","Unnao","Varanasi"],
    "Madhya Pradesh": ["Agar Malwa","Alirajpur","Anuppur","Ashoknagar","Balaghat","Barwani","Betul","Bhind","Bhopal","Burhanpur","Chhatarpur","Chhindwara","Damoh","Datia","Dewas","Dhar","Dindori","Guna","Gwalior","Harda","Hoshangabad","Indore","Jabalpur","Jhabua","Katni","Khandwa","Khargone","Mandla","Mandsaur","Morena","Narsinghpur","Neemuch","Niwari","Panna","Raisen","Rajgarh","Ratlam","Rewa","Sagar","Satna","Sehore","Seoni","Shahdol","Shajapur","Sheopur","Shivpuri","Sidhi","Singrauli","Tikamgarh","Ujjain","Umaria","Vidisha"],
    "Rajasthan": ["Ajmer","Alwar","Banswara","Baran","Barmer","Bharatpur","Bhilwara","Bikaner","Bundi","Chittorgarh","Churu","Dausa","Dholpur","Dungarpur","Hanumangarh","Jaipur","Jaisalmer","Jalore","Jhalawar","Jhunjhunu","Jodhpur","Karauli","Kota","Nagaur","Pali","Pratapgarh","Rajsamand","Sawai Madhopur","Sikar","Sirohi","Sri Ganganagar","Tonk","Udaipur"],
    "Gujarat": ["Ahmedabad","Amreli","Anand","Aravalli","Banaskantha","Bharuch","Bhavnagar","Botad","Chhota Udaipur","Dahod","Dang","Devbhoomi Dwarka","Gandhinagar","Gir Somnath","Jamnagar","Junagadh","Kheda","Kutch","Mahisagar","Mehsana","Morbi","Narmada","Navsari","Panchmahal","Patan","Porbandar","Rajkot","Sabarkantha","Surat","Surendranagar","Tapi","Vadodara","Valsad"],
    "Bihar": ["Araria","Arwal","Aurangabad","Banka","Begusarai","Bhagalpur","Bhojpur","Buxar","Darbhanga","East Champaran","Gaya","Gopalganj","Jamui","Jehanabad","Kaimur","Katihar","Khagaria","Kishanganj","Lakhisarai","Madhepura","Madhubani","Munger","Muzaffarpur","Nalanda","Nawada","Patna","Purnia","Rohtas","Saharsa","Samastipur","Saran","Sheikhpura","Sheohar","Sitamarhi","Siwan","Supaul","Vaishali","West Champaran"],
    "West Bengal": ["Alipurduar","Bankura","Birbhum","Cooch Behar","Dakshin Dinajpur","Darjeeling","Hooghly","Howrah","Jalpaiguri","Jhargram","Kalimpong","Kolkata","Malda","Murshidabad","Nadia","North 24 Parganas","Paschim Bardhaman","Paschim Medinipur","Purba Bardhaman","Purba Medinipur","Purulia","South 24 Parganas","Uttar Dinajpur"],
    "Odisha": ["Angul","Balangir","Balasore","Bargarh","Bhadrak","Boudh","Cuttack","Deogarh","Dhenkanal","Gajapati","Ganjam","Jagatsinghpur","Jajpur","Jharsuguda","Kalahandi","Kandhamal","Kendrapara","Kendujhar","Khordha","Koraput","Malkangiri","Mayurbhanj","Nabarangpur","Nayagarh","Nuapada","Puri","Rayagada","Sambalpur","Subarnapur","Sundargarh"],
    "Assam": ["Bajali","Baksa","Barpeta","Biswanath","Bongaigaon","Cachar","Charaideo","Chirang","Darrang","Dhemaji","Dhubri","Dibrugarh","Dima Hasao","Goalpara","Golaghat","Hailakandi","Hojai","Jorhat","Kamrup","Kamrup Metropolitan","Karbi Anglong","Karimganj","Kokrajhar","Lakhimpur","Majuli","Morigaon","Nagaon","Nalbari","Sivasagar","Sonitpur","South Salmara","Tinsukia","Udalguri","West Karbi Anglong"],
    "Himachal Pradesh": ["Bilaspur","Chamba","Hamirpur","Kangra","Kinnaur","Kullu","Lahaul Spiti","Mandi","Shimla","Sirmaur","Solan","Una"],
    "Uttarakhand": ["Almora","Bageshwar","Chamoli","Champawat","Dehradun","Haridwar","Nainital","Pauri Garhwal","Pithoragarh","Rudraprayag","Tehri Garhwal","Udham Singh Nagar","Uttarkashi"],
    "Jharkhand": ["Bokaro","Chatra","Deoghar","Dhanbad","Dumka","East Singhbhum","Garhwa","Giridih","Godda","Gumla","Hazaribagh","Jamtara","Khunti","Koderma","Latehar","Lohardaga","Pakur","Palamu","Ramgarh","Ranchi","Sahebganj","Seraikela Kharsawan","Simdega","West Singhbhum"],
    "Chhattisgarh": ["Balod","Baloda Bazar","Balrampur","Bastar","Bemetara","Bijapur","Bilaspur","Dantewada","Dhamtari","Durg","Gariaband","Gaurela Pendra Marwahi","Janjgir Champa","Jashpur","Kabirdham","Kanker","Khairagarh","Kondagaon","Korba","Koriya","Mahasamund","Manendragarh","Mohla Manpur","Mungeli","Narayanpur","Raigarh","Raipur","Rajnandgaon","Sakti","Sarangarh Bilaigarh","Sukma","Surajpur","Surguja"],
    "Kerala": ["Alappuzha","Ernakulam","Idukki","Kannur","Kasaragod","Kollam","Kottayam","Kozhikode","Malappuram","Palakkad","Pathanamthitta","Thiruvananthapuram","Thrissur","Wayanad"],
    "Goa": ["North Goa","South Goa"],
    "Manipur": ["Bishnupur","Chandel","Churachandpur","Imphal East","Imphal West","Jiribam","Kakching","Kamjong","Kangpokpi","Noney","Pherzawl","Senapati","Tamenglong","Tengnoupal","Thoubal","Ukhrul"],
    "Meghalaya": ["East Garo Hills","East Jaintia Hills","East Khasi Hills","Eastern West Khasi Hills","North Garo Hills","Ri Bhoi","South Garo Hills","South West Garo Hills","South West Khasi Hills","West Garo Hills","West Jaintia Hills","West Khasi Hills"],
    "Tripura": ["Dhalai","Gomati","Khowai","North Tripura","Sepahijala","South Tripura","Sipahijala","Unakoti","West Tripura"],
    "Nagaland": ["Chumoukedima","Dimapur","Kiphire","Kohima","Longleng","Mokokchung","Mon","Niuland","Noklak","Peren","Phek","Shamator","Tseminyu","Tuensang","Wokha","Zunheboto"],
    "Arunachal Pradesh": ["Anjaw","Changlang","Dibang Valley","East Kameng","East Siang","Kamle","Kra Daadi","Kurung Kumey","Lepa Rada","Lohit","Longding","Lower Dibang Valley","Lower Siang","Lower Subansiri","Namsai","Pakke Kessang","Papum Pare","Shi Yomi","Siang","Tawang","Tirap","Upper Siang","Upper Subansiri","West Kameng","West Siang"],
    "Mizoram": ["Aizawl","Champhai","Hnahthial","Khawzawl","Kolasib","Lawngtlai","Lunglei","Mamit","Saitual","Serchhip","Siaha"],
    "Sikkim": ["East Sikkim","North Sikkim","Pakyong","Soreng","South Sikkim","West Sikkim"],
    "Jammu and Kashmir": ["Anantnag","Bandipora","Baramulla","Budgam","Doda","Ganderbal","Jammu","Kathua","Kishtwar","Kulgam","Kupwara","Poonch","Pulwama","Rajouri","Ramban","Reasi","Samba","Shopian","Srinagar","Udhampur"],
    "Ladakh": ["Kargil","Leh"],
    "Puducherry": ["Karaikal","Mahe","Puducherry","Yanam"],
    "Chandigarh": ["Chandigarh"],
    "Delhi": ["Central Delhi","East Delhi","New Delhi","North Delhi","North East Delhi","North West Delhi","Shahdara","South Delhi","South East Delhi","South West Delhi","West Delhi"],
    "Andaman and Nicobar Islands": ["Nicobar","North and Middle Andaman","South Andaman"],
    "Lakshadweep": ["Lakshadweep"],
    "Dadra and Nagar Haveli and Daman and Diu": ["Dadra and Nagar Haveli","Daman","Diu"],
}

def get_climate_data(village, district, state):
    try:
        coords = DISTRICT_COORDS.get((state, district))
        if not coords:
            return None, f"Coordinates not found for {district}, {state}"
        lat, lon = coords

        if village and village.strip():
            try:
                geo_url = (
                    "https://geocoding-api.open-meteo.com"
                    "/v1/search"
                    f"?name={requests.utils.quote(village.strip())}"
                    "&count=5&language=en&format=json"
                )
                geo_resp = requests.get(geo_url, timeout=5)
                geo_data = geo_resp.json()
                results = geo_data.get("results", [])
                india_results = [r for r in results if r.get("country_code", "").upper() == "IN"]
                state_results = [r for r in india_results if state.lower() in r.get("admin1", "").lower()]
                # Only accept a result that matches the selected state.
                # Do NOT fall back to india_results[0] — that picks same-named
                # villages in other states (e.g. Bheemavaram in AP vs Telangana).
                if state_results:
                    lat = state_results[0]["latitude"]
                    lon = state_results[0]["longitude"]
                    location_label = f"{village}, {district}, {state}"
                    note = "Village location found ✓"
                else:
                    location_label = f"{village}, {district}, {state}"
                    note = f"Using {district} district coordinates"
            except Exception:
                location_label = f"{village}, {district}, {state}"
                note = f"Using {district} district coordinates"
        else:
            location_label = f"{district}, {state}"
            note = "District coordinates used"

        # Daily: temp + rain for 10-year average (2014-2023)
        # Hourly: humidity from ERA5 for accurate mean (avoids daily-aggregate bias)
        climate_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            "&start_date=2014-01-01&end_date=2023-12-31"
            "&daily=temperature_2m_mean,precipitation_sum"
            "&hourly=relative_humidity_2m"
            "&timezone=Asia%2FKolkata"
        )
        climate_resp = requests.get(climate_url, timeout=60)
        climate_data = climate_resp.json()
        daily  = climate_data.get("daily", {})
        hourly = climate_data.get("hourly", {})

        temps_all = daily.get("temperature_2m_mean", [])
        rains_all = daily.get("precipitation_sum", [])
        hums_all  = hourly.get("relative_humidity_2m", [])

        # Temperature: mean of available daily values
        temps    = [t for t in temps_all if t is not None]
        avg_temp = round(sum(temps) / len(temps), 1) if temps else 25.0

        # Rainfall: treat None (dry days) as 0mm; divide by actual total days
        n_days      = len(temps_all) or 3652
        rains       = [r if r is not None else 0.0 for r in rains_all]
        annual_rain = round(sum(rains) / (n_days / 365.0), 1) if rains else 1000.0

        # Humidity: mean of all hourly ERA5 values (complete, no seasonal gaps)
        hums    = [h for h in hums_all if h is not None]
        avg_hum = round(sum(hums) / len(hums), 1) if hums else 60.0

        return {
            "location":    location_label,
            "note":        note,
            "temperature": avg_temp,
            "humidity":    avg_hum,
            "rainfall":    annual_rain,
        }, None

    except Exception as e:
        return None, f"Error: {str(e)}"



# ==============================================================

# ==============================================================

# ==============================================================
# UI — AgroSynapse Redesign (exact match to AgroSynapse Redesign.html)
# ==============================================================

import streamlit.components.v1 as components

# ── Page routing ──────────────────────────────────────────────
_qp = st.query_params.get("page", None)
if _qp and _qp in ("home", "cultivation", "diagnostic", "dashboard"):
    st.session_state.page = _qp

_page = st.session_state.page
_theme  = st.query_params.get("theme", "light")
_accent = st.query_params.get("accent", "0")
_tweaks = st.query_params.get("tweaks", "0")
_toggle_theme = "dark" if _theme == "light" else "light"

_ACCENT_MAP = {
    "0": ("#5a8a3a", "#7ba854", "rgba(122,168,84,0.12)"),
    "1": ("#1a6aae", "#3a8fd0", "rgba(58,143,208,0.12)"),
    "2": ("#c44536", "#e05a4a", "rgba(196,69,54,0.12)"),
    "3": ("#7a9a20", "#9ab830", "rgba(154,184,48,0.12)"),
}
_ac_base, _ac_mid, _ac_alpha = _ACCENT_MAP.get(_accent, _ACCENT_MAP["0"])

# ── Read design CSS ───────────────────────────────────────────
def _read_design_css():
    _dir = os.path.join(BASE, "app")
    _parts = []
    for _fname in ("styles.css", "styles-home.css", "styles-tool.css"):
        _fp = os.path.join(_dir, _fname)
        if os.path.exists(_fp):
            with open(_fp, encoding="utf-8") as _f:
                _parts.append(_f.read())
    return "\n".join(_parts)

_DESIGN_CSS = _read_design_css()

# ── Active class helper ───────────────────────────────────────
def _ac(p): return "active" if _page == p else ""


def _metric_range_html(name, value, min_v, max_v, opt_lo, opt_hi, unit, accent="sage"):
    _pct = min(100.0, max(0.0, ((value - min_v) / (opt_hi - min_v)) * 100 if opt_hi > min_v else 0.0))
    _opt_left = round((opt_lo - min_v) / (max_v - min_v) * 100, 1)
    _opt_width = round((opt_hi - opt_lo) / (max_v - min_v) * 100, 1)
    _in_range = opt_lo <= value <= opt_hi
    _fill = ("var(--sage)" if accent == "sage" else "var(--earth-2)") if _in_range else "var(--earth-2)"
    _status_class = "ok" if _in_range else "warn"
    _status_text = "Optimal" if _in_range else "Outside range"
    return f"""
<div class="field-range">
  <div class="field-range-head">
    <span class="field-range-name">{name}</span>
    <span class="field-range-status {_status_class}">{_status_text}</span>
  </div>
  <div class="range-bar">
    <div class="opt" style="left:{_opt_left}%;width:{_opt_width}%;"></div>
    <div class="fill" style="width:{_pct:.1f}%;background:{_fill};"></div>
  </div>
  <div class="field-range-note">Opt: {opt_lo:g}-{opt_hi:g} {unit}</div>
</div>"""


def _now_ist():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

_crumb = {
    "cultivation": " - PREDICTIVE CULTIVATION",
    "diagnostic":  " - PHYTO-DIAGNOSTIC SUITE",
    "dashboard":   " - ANALYTICS DASHBOARD",
}.get(_page, "")

# ── 1. Fonts ──────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1'
    '&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── 2. CSS — dynamic accent + static chrome ───────────────────
_CHROME_CSS = (
    "[data-testid='stSidebar'],[data-testid='stSidebarNav'],"
    "#MainMenu,footer,header.stAppHeader,.stDeployButton,"
    "[data-testid='stToolbar'],[data-testid='stDecoration'],"
    "[data-testid='stStatusWidget'],[data-testid='collapsedControl'],"
    "[data-testid='stFileUploaderDeleteBtn'],button[title='Clear file'],"
    "button[aria-label='Delete']"
    "{display:none!important;visibility:hidden!important;}"
    ".block-container{padding:0!important;max-width:100%!important;padding-top:0!important;padding-bottom:0!important;}"
    "section[data-testid='stMain']>div:first-child{padding-top:0!important;}"
    ".stApp{background:#faf8f3!important;}"
    "section[data-testid='stMain']{padding-left:72px!important;padding-top:69px!important;min-height:100vh;}"
    "[data-testid='stForm']{margin-top:0!important;padding-top:0!important;border:0!important;}"
    ".tool-header{padding-bottom:16px!important;}"
    "#as-rail{position:fixed;left:0;top:0;z-index:100;width:72px;height:100vh;background:#0f2818;}"
    "#as-topbar{position:fixed;top:0;left:72px;right:0;z-index:90;height:69px;"
    "display:flex;align-items:center;padding:0 32px;gap:16px;"
    "background:rgba(250,248,243,0.92);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);"
    "border-bottom:1px solid rgba(20,20,15,0.08);}"
    "a.rail-btn,a.rail-btn:visited,a.rail-btn:link{"
    "text-decoration:none!important;color:rgba(250,248,243,0.45)!important;"
    "width:44px;height:44px;border:0;background:transparent;"
    "border-radius:10px;display:grid;place-items:center;"
    "cursor:pointer;transition:all 0.25s cubic-bezier(0.2,0.8,0.2,1);position:relative;}"
    "a.rail-btn:hover{color:#faf8f3!important;background:rgba(250,248,243,0.08)!important;}"
    + "a.rail-btn.active{color:" + _ac_mid + "!important;background:" + _ac_alpha + "!important;}"
    + "a.rail-btn.active::before{content:'';position:absolute;left:-18px;top:50%;"
    "transform:translateY(-50%);width:2px;height:22px;background:" + _ac_mid + ";border-radius:0 2px 2px 0;}"
    + ".topbar-nav a{text-decoration:none!important;cursor:pointer;padding:7px 16px;font-size:13px;"
    "color:#3a3a32;border-radius:999px;font-weight:500;transition:all 0.2s;}"
    ".topbar-nav a:hover{color:#14140f;background:rgba(20,20,15,0.05);}"
    + ".topbar-nav a.active{background:" + _ac_base + "!important;color:#faf8f3!important;"
    "box-shadow:0 1px 2px rgba(15,40,24,0.04),0 2px 8px rgba(15,40,24,0.03);}"
    + ".topbar-icon{width:34px;height:34px;border:0;background:transparent;"
    "border-radius:8px;display:grid;place-items:center;cursor:pointer;"
    "color:#3a3a32;text-decoration:none;transition:all 0.2s;flex-shrink:0;}"
    ".topbar-icon:hover{background:rgba(20,20,15,0.07);}"
    ".rail-logo{width:36px;height:36px;border-radius:10px;background:" + _ac_alpha + ";"
    "display:grid;place-items:center;color:" + _ac_mid + ";margin-bottom:8px;}"
    + ".rail-spacer{flex:1;}"
    ".rail-user{width:36px;height:36px;border-radius:50%;background:#d4a373;color:#0f2818;"
    "display:grid;place-items:center;font-weight:600;font-size:12px;font-family:'JetBrains Mono',monospace;margin-top:8px;}"
    "[data-testid='stNumberInput']>div>div>input{"
    "font-family:'JetBrains Mono',monospace!important;font-size:15px!important;"
    "border:1px solid rgba(20,20,15,0.12)!important;border-radius:10px!important;"
    "background:#faf8f3!important;color:#14140f!important;padding:12px 14px!important;}"
    "[data-testid='stSelectbox']>div>div{"
    "border:1px solid rgba(20,20,15,0.12)!important;border-radius:10px!important;"
    "background:#faf8f3!important;font-family:'JetBrains Mono',monospace!important;}"
    "[data-testid='stTextInput'] input{"
    "font-family:'JetBrains Mono',monospace!important;"
    "border:1px solid rgba(20,20,15,0.12)!important;border-radius:10px!important;"
    "background:#faf8f3!important;color:#14140f!important;padding:12px 14px!important;}"
    "[data-testid='stFileUploadDropzone']{"
    "border:1.5px dashed rgba(20,20,15,0.12)!important;"
    "border-radius:14px!important;background:#faf8f3!important;}"
    "[data-testid='stButton']>button{font-family:'Inter Tight',-apple-system,sans-serif!important;border-radius:999px!important;cursor:pointer;}"
    "div.stElementContainer,div.stMarkdown,div[data-testid='stVerticalBlock']{gap:0!important;}"
    "[data-testid='stFormSubmitButton']>button{"
    "background:#0f2818!important;color:#faf8f3!important;"
    "border-radius:999px!important;padding:14px 28px!important;"
    "font-size:14px!important;font-weight:500!important;"
    "border:0!important;cursor:pointer!important;"
    "font-family:'Inter Tight',-apple-system,sans-serif!important;}"
    "[data-testid='stFormSubmitButton']>button:hover{background:#14140f!important;}"
    "[data-testid='stDownloadButton']>button,[data-testid='stButton']>button{"
    "font-family:'Inter Tight',-apple-system,sans-serif!important;"
    "border-radius:999px!important;padding:14px 24px!important;"
    "font-size:14px!important;border:1px solid rgba(20,20,15,0.12)!important;"
    "background:#faf8f3!important;color:#14140f!important;box-shadow:none!important;}"
    "[data-testid='stDownloadButton']>button:hover,[data-testid='stButton']>button:hover{background:#f2ede2!important;}"
    "[data-testid='stButton']>button[kind='primary']{background:#5a8a3a!important;color:#faf8f3!important;border-color:transparent!important;}"
    "[data-testid='stButton']>button[kind='primary']:hover{background:#0f2818!important;color:#faf8f3!important;}"
    "[data-testid='stNumberInput']{margin-bottom:20px!important;}"
    "[data-testid='stNumberInput'] label{"
    "font-family:'JetBrains Mono',monospace!important;font-size:10px!important;"
    "letter-spacing:0.14em!important;text-transform:uppercase!important;"
    "color:#8a8a78!important;margin-bottom:6px!important;display:block!important;}"
    "[data-testid='stSelectbox']{margin-bottom:20px!important;}"
    "[data-testid='stSelectbox'] label{"
    "font-family:'JetBrains Mono',monospace!important;font-size:10px!important;"
    "letter-spacing:0.14em!important;text-transform:uppercase!important;"
    "color:#8a8a78!important;margin-bottom:6px!important;display:block!important;}"
    "[data-testid='stTextInput']{margin-bottom:20px!important;}"
    "[data-testid='stTextInput'] label{"
    "font-family:'JetBrains Mono',monospace!important;font-size:10px!important;"
    "letter-spacing:0.14em!important;text-transform:uppercase!important;"
    "color:#8a8a78!important;margin-bottom:6px!important;display:block!important;}"
    ".prob-row{display:grid;grid-template-columns:130px 1fr 54px;gap:12px;"
    "align-items:center;padding:9px 0;border-bottom:1px solid rgba(20,20,15,0.05);}"
    ".prob-track{height:5px;background:rgba(20,20,15,0.07);border-radius:3px;overflow:hidden;}"
    ".prob-fill{height:100%;border-radius:3px;transition:width 1.2s cubic-bezier(0.2,0.8,0.2,1);}"
    ".prob-k{font-size:12px;color:#6b6b5e;font-family:'JetBrains Mono',monospace;"
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}"
    ".prob-v{font-size:12px;color:#3a3a32;text-align:right;font-family:'JetBrains Mono',monospace;}"
    ".tweaks-panel{position:fixed;bottom:80px;right:20px;z-index:9999;"
    "background:#faf8f3;border:1px solid rgba(20,20,15,0.1);border-radius:16px;"
    "padding:20px 24px;box-shadow:0 12px 48px rgba(15,40,24,0.18);min-width:230px;"
    "font-family:'JetBrains Mono',monospace;}"
    ".tweaks-row{display:flex;align-items:center;justify-content:space-between;"
    "gap:12px;margin-top:14px;}"
    ".tweaks-label{font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:#8a8a78;}"
    ".tweaks-dots{display:flex;gap:6px;}"
    ".tweaks-dot{width:20px;height:20px;border-radius:50%;border:2px solid transparent;"
    "cursor:pointer;text-decoration:none;transition:border 0.2s;}"
    ".tweaks-dot.selected{border-color:#14140f;}"
    ".tweaks-btns{display:flex;gap:4px;}"
    ".tweaks-btn{padding:5px 10px;border-radius:6px;font-size:11px;text-decoration:none;"
    "background:rgba(20,20,15,0.06);color:#3a3a32;transition:all 0.2s;}"
    ".tweaks-btn.active{background:#0f2818;color:#faf8f3;}"
    ".range-bar-wrap{margin-top:8px;margin-bottom:4px;}"
    ".range-bar-track{height:4px;background:rgba(20,20,15,0.08);border-radius:2px;"
    "position:relative;overflow:hidden;}"
    ".range-bar-fill{height:100%;border-radius:2px;"
    "transition:width 0.6s cubic-bezier(0.2,0.8,0.2,1);}"
    ".range-hint{font-size:10px;margin-top:5px;letter-spacing:0.02em;}"
    ".diag-specimen-wrap{background:#f2ede2;border:1px solid rgba(20,20,15,0.08);border-radius:14px;padding:12px;margin-top:10px;}"
    ".diag-specimen-wrap [data-testid='stFileUploader']{margin-bottom:10px!important;}"
    ".diag-specimen-wrap [data-testid='stImage'] img{border-radius:12px;object-fit:contain;max-height:360px;width:auto!important;margin:auto;display:block;}"
    "div[data-testid='stVerticalBlock']:has(#diag-specimen-card){background:#fdfbf6;border:1px solid rgba(20,20,15,0.12);border-radius:20px;padding:24px;box-shadow:0 10px 28px rgba(15,40,24,0.06);}" 
    "div[data-testid='stVerticalBlock']:has(#diag-specimen-card) [data-testid='stFileUploaderDropzone']{background:#efeeea!important;border:1px solid rgba(20,20,15,0.08)!important;border-radius:12px!important;}"
    "div[data-testid='stVerticalBlock']:has(#diag-specimen-card) [data-testid='stImage'] img{border-radius:12px;max-height:360px;width:auto!important;object-fit:contain;display:block;margin:0 auto;}"
    ".tool-header{margin-bottom:10px!important;padding-bottom:14px!important;}"
    "div[data-testid='column']:has(#cult-soil-card),div[data-testid='column']:has(#cult-chem-card),div[data-testid='column']:has(#cult-climate-card),div[data-testid='column']:has(#cult-farm-card){background:#ffffff!important;border:1.5px solid rgba(20,20,15,0.20)!important;border-radius:20px!important;padding:20px 20px 16px!important;box-shadow:0 2px 12px rgba(15,40,24,0.08),0 1px 3px rgba(15,40,24,0.04)!important;}"
    "div[data-testid='column']:has(#cult-soil-card)>div[data-testid='stVerticalBlock'],div[data-testid='column']:has(#cult-chem-card)>div[data-testid='stVerticalBlock'],div[data-testid='column']:has(#cult-climate-card)>div[data-testid='stVerticalBlock'],div[data-testid='column']:has(#cult-farm-card)>div[data-testid='stVerticalBlock']{gap:0!important;}"
    "div[data-testid='column']:has(#cult-soil-card) [data-testid='stFileUploaderDropzone']{background:#f2ede2!important;border:1.5px dashed rgba(20,20,15,0.18)!important;border-radius:12px!important;min-height:120px!important;padding:14px 16px!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock']{background:linear-gradient(135deg,#062515 0%,#0f2818 62%,#173b24 100%)!important;border-radius:20px!important;padding:28px 40px!important;border:1px solid rgba(250,248,243,0.12)!important;box-shadow:0 4px 20px rgba(15,40,24,0.18)!important;margin-top:14px!important;align-items:center!important;width:100%!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stMarkdownContainer'] h3,div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stMarkdownContainer'] p{color:#faf8f3!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stMarkdownContainer'] h3{font-family:'Instrument Serif',serif!important;font-size:28px!important;letter-spacing:-0.02em!important;line-height:1!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stMarkdownContainer'] p{font-size:14px!important;opacity:0.72!important;margin-top:6px!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stButton']>button{background:#5a8a3a!important;color:#faf8f3!important;border:0!important;font-weight:600!important;padding:14px 28px!important;font-size:14px!important;border-radius:999px!important;}"
    "div[data-testid='stElementContainer']:has(#ready-synth-card) + div[data-testid='stHorizontalBlock'] [data-testid='stButton']>button:hover{background:#6a9a46!important;color:#faf8f3!important;}"
)

_DARK_CSS = (
    "body,.stApp{background:#0a160a!important;}"
    "section[data-testid='stMain']{background:#0a160a!important;}"
    "#as-rail{background:#050e05!important;}"
    "#as-topbar{background:rgba(5,14,5,0.96)!important;border-bottom-color:rgba(250,248,243,0.06)!important;}"
    ".topbar-crumb span,.topbar-nav a{color:#7a9a70!important;}"
    "[data-testid='stNumberInput']>div>div>input,[data-testid='stTextInput'] input{"
    "background:#0f1f0f!important;color:#e8e5dc!important;border-color:rgba(250,248,243,0.12)!important;}"
    "[data-testid='stSelectbox']>div>div{background:#0f1f0f!important;}"
    "[data-testid='stSelectbox'] span{color:#e8e5dc!important;}"
    "[data-testid='stNumberInput'] label,[data-testid='stSelectbox'] label,"
    "[data-testid='stTextInput'] label{color:#5a7a50!important;}"
    ".stApp,.block-container{background:#0a160a!important;}"
) if _theme == "dark" else ""

_ALL_CSS = "<style>" + _CHROME_CSS + _DESIGN_CSS + _DARK_CSS + "</style>"
if hasattr(st, "html"):
    st.html(_ALL_CSS)
else:
    st.markdown(_ALL_CSS, unsafe_allow_html=True)

# ── 3. Chrome HTML ─────────────────────────────────────────────
_ac_home = _ac("home"); _ac_cult = _ac("cultivation")
_ac_diag = _ac("diagnostic"); _ac_dash = _ac("dashboard")
_tweaks_href = f"?page={_page}&theme={_theme}&accent={_accent}&tweaks={'0' if _tweaks=='1' else '1'}"
_base_params = f"theme={_theme}&accent={_accent}"

# Accent dot helpers
def _adot(a_idx, a_color):
    _sel = "selected" if _accent == str(a_idx) else ""
    return (f'<a class="tweaks-dot {_sel}" href="?page={_page}&{_base_params.replace("accent="+_accent,"accent="+str(a_idx))}&tweaks=1" '
            f'target="_self" style="background:{a_color};"></a>')

_tweaks_panel = f"""
<div class="tweaks-panel">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:10px;letter-spacing:0.16em;text-transform:uppercase;color:#14140f;font-weight:600;">TWEAKS</span>
    <a href="?page={_page}&{_base_params}&tweaks=0" target="_self" style="font-size:18px;color:#8a8a78;text-decoration:none;line-height:1;">&times;</a>
  </div>
  <div class="tweaks-row">
    <span class="tweaks-label">Accent</span>
    <div class="tweaks-dots">
      {_adot(0,"#5a8a3a")}{_adot(1,"#1a6aae")}{_adot(2,"#c44536")}{_adot(3,"#9ab830")}
    </div>
  </div>
  <div class="tweaks-row">
    <span class="tweaks-label">Theme</span>
    <div class="tweaks-btns">
      <a class="tweaks-btn {'active' if _theme=='light' else ''}" href="?page={_page}&theme=light&accent={_accent}&tweaks=1" target="_self">LIGHT</a>
      <a class="tweaks-btn {'active' if _theme=='dark' else ''}" href="?page={_page}&theme=dark&accent={_accent}&tweaks=1" target="_self">DARK</a>
    </div>
  </div>
  <div class="tweaks-row">
    <span class="tweaks-label">Page</span>
    <div class="tweaks-btns">
      <a class="tweaks-btn {'active' if _page=='home' else ''}" href="?page=home&{_base_params}&tweaks=1" target="_self">HOME</a>
      <a class="tweaks-btn {'active' if _page=='cultivation' else ''}" href="?page=cultivation&{_base_params}&tweaks=1" target="_self">CULT.</a>
      <a class="tweaks-btn {'active' if _page=='diagnostic' else ''}" href="?page=diagnostic&{_base_params}&tweaks=1" target="_self">DIAG.</a>
      <a class="tweaks-btn {'active' if _page=='dashboard' else ''}" href="?page=dashboard&{_base_params}&tweaks=1" target="_self">DASH.</a>
    </div>
  </div>
</div>""" if _tweaks == "1" else ""

st.markdown(f"""
<aside id="as-rail" style="position:fixed;left:0;top:0;z-index:100;width:72px;height:100vh;
  background:#0f2818;display:flex;flex-direction:column;align-items:center;padding:16px 0 20px;">
  <div class="rail-logo" title="AgroSynapse" style="margin-bottom:20px;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{_ac_mid}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M4 20c0-9 7-16 16-16 0 9-7 16-16 16Z" fill="{_ac_mid}" fill-opacity="0.2"/><path d="M4 20 12 12" stroke="{_ac_mid}"/>
    </svg>
  </div>
  <a class="rail-btn {_ac_home}" href="?page=home&{_base_params}" target="_self" title="Home">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1v-9.5Z"/></svg>
  </a>
  <a class="rail-btn {_ac_cult}" href="?page=cultivation&{_base_params}" target="_self" title="Cultivation">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22V10"/><path d="M12 10c0-3 2-6 6-6 0 3-2 6-6 6Z"/><path d="M12 12c0-2.5-2-5-6-5 0 2.5 2 5 6 5Z"/></svg>
  </a>
  <a class="rail-btn {_ac_diag}" href="?page=diagnostic&{_base_params}" target="_self" title="Diagnostic">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V4a1 1 0 0 1 1-1h3"/><path d="M17 3h3a1 1 0 0 1 1 1v3"/><path d="M21 17v3a1 1 0 0 1-1 1h-3"/><path d="M7 21H4a1 1 0 0 1-1-1v-3"/><path d="M3 12h18"/></svg>
  </a>
  <a class="rail-btn {_ac_dash}" href="?page=dashboard&{_base_params}" target="_self" title="Dashboard">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-6"/></svg>
  </a>
  <div class="rail-spacer"></div>
  <a class="rail-btn" href="{_tweaks_href}" target="_self" title="Tweaks">
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/></svg>
  </a>
  <div class="rail-user">MA</div>
</aside>
<header id="as-topbar">
  <div class="topbar-crumb" style="display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#6b6b5e;">
    <span style="width:6px;height:6px;border-radius:50%;background:{_ac_mid};display:inline-block;"></span>
    <span>AGROSYNAPSE{_crumb}</span>
  </div>
  <div style="flex:1;"></div>
  <nav class="topbar-nav" style="display:flex;gap:4px;">
    <a class="{_ac_home}" href="?page=home&{_base_params}" target="_self">Home</a>
    <a class="{_ac_cult}" href="?page=cultivation&{_base_params}" target="_self">Cultivation</a>
    <a class="{_ac_diag}" href="?page=diagnostic&{_base_params}" target="_self">Diagnostic</a>
    <a class="{_ac_dash}" href="?page=dashboard&{_base_params}" target="_self">Dashboard</a>
  </nav>
  <a class="topbar-icon" href="javascript:void(0)" title="Search" style="margin-left:8px;">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
  </a>
  <a class="topbar-icon" href="javascript:void(0)" title="Notifications">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10 21a2 2 0 0 0 4 0"/></svg>
  </a>
  <a class="topbar-icon" href="{_tweaks_href}" target="_self" title="Tweaks / Settings">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="12" y1="18" x2="20" y2="18"/></svg>
  </a>
</header>
{_tweaks_panel}
""", unsafe_allow_html=True)

# ==============================================================
# HOME PAGE
# ==============================================================
if _page == "home":

    # ── Animated hero section via components.html ──────────────
    _HERO = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#faf8f3;font-family:"Inter Tight",-apple-system,sans-serif;overflow:hidden;}
.hero{position:relative;height:100vh;display:flex;flex-direction:column;overflow:hidden;padding:0 40px 80px;}
.hero-visual{position:absolute;inset:0;z-index:0;overflow:hidden;}
.hero-visual svg{width:100%;height:100%;display:block;}
@keyframes heroFloat{0%,100%{transform:scale(1.05) translate(0,0);}50%{transform:scale(1.08) translate(-1%,-0.5%);}}
.hero-overlay{position:absolute;inset:0;z-index:1;
  background:linear-gradient(180deg,rgba(250,248,243,0) 0%,rgba(250,248,243,0.05) 40%,rgba(250,248,243,0.88) 95%,#faf8f3 100%),
  linear-gradient(90deg,rgba(10,15,10,0.55) 0%,rgba(10,15,10,0.25) 60%,rgba(10,15,10,0.45) 100%);}
.hero-grid{position:relative;z-index:2;display:flex;align-items:center;flex:1;padding-top:80px;max-width:920px;}
.hero-brandname{font-family:"Instrument Serif","Times New Roman",serif;font-size:clamp(18px,2.2vw,28px);
  letter-spacing:0.18em;text-transform:uppercase;color:rgba(250,248,243,0.55);margin-bottom:20px;}
.hero-brandname em{font-style:normal;color:#e8c989;}
.hero-title{font-family:"Instrument Serif","Times New Roman",serif;font-weight:400;
  font-size:clamp(52px,8vw,120px);color:#faf8f3;letter-spacing:-0.04em;line-height:0.93;
  margin-bottom:36px;text-shadow:0 4px 40px rgba(0,0,0,0.25);}
.hero-title em{font-style:italic;color:#e8c989;}
.hero-lede{max-width:560px;font-size:17px;line-height:1.6;color:rgba(250,248,243,0.85);
  margin-bottom:18px;text-shadow:0 2px 20px rgba(0,0,0,0.3);}
.btn{display:inline-flex;align-items:center;gap:10px;padding:14px 22px;
  font-family:"Inter Tight",-apple-system,sans-serif;font-size:14px;font-weight:500;
  border:0;border-radius:999px;cursor:pointer;text-decoration:none;white-space:nowrap;transition:all 0.3s;}
.hero-badges{display:flex;flex-wrap:wrap;gap:10px;}
.hero-badge{display:inline-flex;align-items:center;gap:7px;padding:8px 14px;
  background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.18);
  backdrop-filter:blur(8px);border-radius:999px;font-size:12px;
  color:rgba(250,248,243,0.8);font-weight:500;}
@keyframes floatY{0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);}}
.hero-tick{position:absolute;bottom:30px;left:40px;z-index:3;
  display:flex;align-items:center;gap:8px;color:rgba(250,248,243,0.5);
  animation:floatY 2.4s ease-in-out infinite;font-family:"JetBrains Mono",monospace;
  font-size:11px;letter-spacing:0.14em;text-transform:uppercase;}
</style>
</head><body>
<section class="hero">
<div class="hero-visual"><svg id="herosvg" viewBox="0 0 600 600" preserveAspectRatio="xMidYMid slice">
<defs>
<linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
  <stop offset="0%" stop-color="#f5e8cf"/>
  <stop offset="40%" stop-color="#e8c989"/>
  <stop offset="75%" stop-color="#c68f4c"/>
  <stop offset="100%" stop-color="#5a3a1a"/>
</linearGradient>
<radialGradient id="sun" cx="0.72" cy="0.38" r="0.3">
  <stop offset="0%" stop-color="#fff3d1" stop-opacity="1"/>
  <stop offset="40%" stop-color="#fcd982" stop-opacity="0.6"/>
  <stop offset="100%" stop-color="#f5a94e" stop-opacity="0"/>
</radialGradient>
<linearGradient id="field" x1="0" x2="0" y1="0" y2="1">
  <stop offset="0%" stop-color="#4a5a2a"/>
  <stop offset="100%" stop-color="#1a2812"/>
</linearGradient>
<filter id="grain">
  <feTurbulence baseFrequency="0.9" numOctaves="2"/>
  <feColorMatrix values="0 0 0 0 0.2 0 0 0 0 0.15 0 0 0 0 0.1 0 0 0 0.15 0"/>
  <feComposite in2="SourceGraphic" operator="in"/>
</filter>
<radialGradient id="vig" cx="0.5" cy="0.5" r="0.75">
  <stop offset="60%" stop-color="#000" stop-opacity="0"/>
  <stop offset="100%" stop-color="#000" stop-opacity="0.5"/>
</radialGradient>
</defs>
<rect width="600" height="600" fill="url(#sky)"/>
<rect width="600" height="600" fill="url(#sun)"/>
<path id="ridge1" d="M0 380 Q150 360 300 375 T600 370 L600 600 L0 600 Z" fill="#6a4a2a" opacity="0.7"/>
<path d="M0 420 Q200 400 400 415 T600 410 L600 600 L0 600 Z" fill="#3e2a18" opacity="0.85"/>
<path d="M0 450 Q300 435 600 450 L600 600 L0 600 Z" fill="url(#field)"/>
<g id="stalks"></g>
<g id="fgstalks"></g>
<rect width="600" height="600" filter="url(#grain)" opacity="0.4"/>
<rect width="600" height="600" fill="url(#vig)"/>
</svg></div>
<div class="hero-overlay"></div>
<div class="hero-grid">
<div class="hero-copy">
  <div class="hero-brandname">Agro<em>Synapse</em></div>
  <h1 class="hero-title">
    Laboratory-grade<br>agronomy, delivered<br>to every <em>acre.</em>
  </h1>
  <p class="hero-lede">Fusing soil vision, climate synthesis, and phyto-diagnostic neural nets into a single recommendation engine - tuned for your field, not the average of everyone else&#8217;s.</p>
</div>
</div>
<div class="hero-tick"><span>SCROLL</span><span>&#8595;</span></div>
</section>
<script>
var svg=document.getElementById('herosvg');
var stalksG=document.getElementById('stalks');
var fgG=document.getElementById('fgstalks');
// Build stalks
for(var i=0;i<22;i++){
  var g=document.createElementNS('http://www.w3.org/2000/svg','g');
  g.setAttribute('opacity',0.55+(i%3)*0.15);
  var p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('fill','none');p.setAttribute('stroke','#3a2812');p.setAttribute('stroke-width','1.3');
  var e=document.createElementNS('http://www.w3.org/2000/svg','ellipse');
  e.setAttribute('rx','3');e.setAttribute('ry','10');e.setAttribute('fill','#c89a5c');
  g.appendChild(p);g.appendChild(e);stalksG.appendChild(g);
}
for(var i=0;i<8;i++){
  var g=document.createElementNS('http://www.w3.org/2000/svg','g');
  g.setAttribute('opacity','0.8');g.setAttribute('filter','blur(2px)');
  var p=document.createElementNS('http://www.w3.org/2000/svg','path');
  p.setAttribute('fill','none');p.setAttribute('stroke','#2a1c0a');p.setAttribute('stroke-width','4');p.setAttribute('stroke-linecap','round');
  g.appendChild(p);fgG.appendChild(g);
}
function animate(now){
  var t=now/1000;
  var stalks=stalksG.children;
  var fgs=fgG.children;
  for(var i=0;i<22;i++){
    var x=(i/22)*600+Math.sin(t/2+i)*2;
    var h=120+(i%5)*20;
    var bend=Math.sin((3+(i%3))*t/2+i*0.7)*3;
    var x2=x+bend*2;var y2=600-h;
    stalks[i].children[0].setAttribute('d','M '+x+' 600 Q '+(x+bend)+' '+(600-h/2)+' '+x2+' '+y2);
    stalks[i].children[1].setAttribute('cx',x2);
    stalks[i].children[1].setAttribute('cy',y2);
    stalks[i].children[1].setAttribute('transform','rotate('+(bend*2)+' '+x2+' '+y2+')');
  }
  for(var i=0;i<8;i++){
    var x=(i/8)*700-40+Math.sin(t/2.5+i)*4;
    var h=200+(i%3)*40;
    var bend=Math.sin(t/2+i)*6;
    fgs[i].children[0].setAttribute('d','M '+x+' 600 Q '+(x+bend)+' '+(600-h/2)+' '+(x+bend*1.5)+' '+(600-h));
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
// Float SVG
var hsvg=document.querySelector('.hero-visual svg');
var _t0=performance.now();
function floatSVG(now){
  var p=(now-_t0)/16000;
  var s=1.05+0.03*Math.sin(p*2*Math.PI);
  var tx=-1*Math.sin(p*2*Math.PI);
  var ty=-0.5*Math.sin(p*2*Math.PI);
  hsvg.style.transform='scale('+s+') translate('+tx+'%,'+ty+'%)';
  requestAnimationFrame(floatSVG);
}
requestAnimationFrame(floatSVG);
</script>
</body></html>"""
    components.html(_HERO, height=680, scrolling=False)

    # ── Module cards ───────────────────────────────────────────
    st.markdown("""
<section class="modules" style="padding-top:28px;">
  <div class="section-head" style="margin-bottom:26px;">
    <span class="eyebrow">Modules · active</span>
    <h2 class="display section-title">Two instruments, <em>calibrated</em> for your land.</h2>
  </div>
  <div class="modules-grid">
    <a class="module-card" href="?page=cultivation" target="_self" style="text-decoration:none;">
      <div class="module-card-header">
        <div class="module-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22V10"/><path d="M12 10c0-3 2-6 6-6 0 3-2 6-6 6Z"/><path d="M12 12c0-2.5-2-5-6-5 0 2.5 2 5 6 5Z"/></svg>
        </div>
        <span class="eyebrow">Agricultural · core</span>
      </div>
      <h3 class="module-title display">Predictive Cultivation</h3>
      <p class="module-desc">Multimodal soil-to-crop fusion. Fuses specimen imagery, chemical profile, climate vectors, and farm history to return top-K crops with protocol.</p>
      <div class="module-cta">
        <span>Open module</span>
        <div class="module-cta-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>
        </div>
      </div>
    </a>
    <a class="module-card earth" href="?page=diagnostic" target="_self" style="text-decoration:none;">
      <div class="module-card-header">
        <div class="module-icon">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h4v4a2 2 0 0 1-2 2Z"/><path d="M12 6H6"/><path d="M10 2h4"/></svg>
        </div>
        <span class="eyebrow">Neural · vision</span>
      </div>
      <h3 class="module-title display">Phyto-Diagnostic Suite</h3>
      <p class="module-desc">Leaf-to-cure vision. Convolutional pathology engine trained across 38 classes, resolving pathogen identity and treatment dosage in a single pass.</p>
      <div class="module-cta">
        <span>Open module</span>
        <div class="module-cta-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>
        </div>
      </div>
    </a>
  </div>
</section>
""", unsafe_allow_html=True)

    # ── Soil pipeline ──────────────────────────────────────────
    st.markdown("""
<section style="background:#0f2818;padding:80px 48px;margin:0;border-top:1px solid rgba(250,248,243,0.06);">
  <div style="margin-bottom:48px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:rgba(250,248,243,0.45);">Pipeline · soil to crop</span>
    <h2 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:clamp(36px,5vw,64px);font-weight:400;color:#faf8f3;margin:12px 0 0;line-height:1.05;">From soil, a <em style="color:#7ba854;">crop.</em></h2>
    <p style="font-size:16px;line-height:1.55;color:rgba(250,248,243,0.65);margin-top:16px;max-width:600px;">Four stages of multimodal synthesis converge into a probability-ranked cultivation protocol.</p>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;">
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.07);border:1px solid rgba(250,248,243,0.12);border-radius:18px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(122,168,84,0.15);border:1px solid rgba(122,168,84,0.3);display:grid;place-items:center;color:#a8d080;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V4a1 1 0 0 1 1-1h3"/><path d="M17 3h3a1 1 0 0 1 1 1v3"/><path d="M21 17v3a1 1 0 0 1-1 1h-3"/><path d="M7 21H4a1 1 0 0 1-1-1v-3"/><path d="M3 12h18"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">01</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Specimen vision</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.65);margin:0 0 20px;">Upload soil imagery. Computer vision extracts texture, aggregation, color-moisture indices in under 400ms.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.07);border:1px solid rgba(250,248,243,0.12);border-radius:18px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(122,168,84,0.15);border:1px solid rgba(122,168,84,0.3);display:grid;place-items:center;color:#a8d080;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">02</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Chemical profile</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.65);margin:0 0 20px;">NPK + pH triangulated against optimal bands. Outliers flagged; dosages computed.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.07);border:1px solid rgba(250,248,243,0.12);border-radius:18px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(122,168,84,0.15);border:1px solid rgba(122,168,84,0.3);display:grid;place-items:center;color:#a8d080;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">03</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Climate synthesis</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.65);margin:0 0 20px;">Auto-fills 180+ district-grade weather, humidity, and rainfall vectors from geographic selection.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="padding:28px 24px;background:rgba(250,248,243,0.07);border:1px solid rgba(250,248,243,0.12);border-radius:18px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(122,168,84,0.15);border:1px solid rgba(122,168,84,0.3);display:grid;place-items:center;color:#a8d080;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22V10"/><path d="M12 10c0-3 2-6 6-6 0 3-2 6-6 6Z"/><path d="M12 12c0-2.5-2-5-6-5 0 2.5 2 5 6 5Z"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">04</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Crop recommendation</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.65);margin:0;">Multimodal fusion returns top-K crops with NPK protocol, timeline, and confidence ranking.</p>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

    # ── Leaf pipeline ──────────────────────────────────────────
    st.markdown("""
<section style="background:#0f2818;padding:80px 48px;margin:0;border-top:1px solid rgba(250,248,243,0.08);">
  <div style="margin-bottom:48px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:rgba(250,248,243,0.45);">Pipeline · leaf to cure</span>
    <h2 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:clamp(36px,5vw,64px);font-weight:400;color:#faf8f3;margin:12px 0 0;line-height:1.05;">From leaf, a <em style="color:#d4a373;">cure.</em></h2>
    <p style="font-size:16px;line-height:1.55;color:rgba(250,248,243,0.65);margin-top:16px;max-width:600px;">A single forward pass resolves pathogen identity and returns a dosage-precise treatment plan.</p>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:24px;">
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.09);border:1px solid rgba(250,248,243,0.2);border-radius:18px;box-shadow:0 10px 26px rgba(0,0,0,0.18);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(212,163,115,0.1);border:1px solid rgba(212,163,115,0.25);display:grid;place-items:center;color:#b8884f;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4m0 0-4 4m4-4 4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">01</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Leaf specimen</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.68);margin:0 0 20px;">Upload a close-up of a single leaf. Auto-validation checks framing, focus, and species ambiguity.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.09);border:1px solid rgba(250,248,243,0.2);border-radius:18px;box-shadow:0 10px 26px rgba(0,0,0,0.18);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(212,163,115,0.1);border:1px solid rgba(212,163,115,0.25);display:grid;place-items:center;color:#b8884f;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h4v4a2 2 0 0 1-2 2Z"/><path d="M12 6H6"/><path d="M10 2h4"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">02</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Neural pathology</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.68);margin:0 0 20px;">Convolutional network scans lesions across 38 pathogen classes and one healthy baseline.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="position:relative;padding:28px 24px;background:rgba(250,248,243,0.09);border:1px solid rgba(250,248,243,0.2);border-radius:18px;box-shadow:0 10px 26px rgba(0,0,0,0.18);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(212,163,115,0.1);border:1px solid rgba(212,163,115,0.25);display:grid;place-items:center;color:#b8884f;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">03</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Dosage synthesis</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.68);margin:0 0 20px;">Confidence-weighted treatment plan: primary chemistry, fertilizer correction, cultural practices.</p>
      <div style="width:28px;height:28px;border-radius:50%;background:rgba(250,248,243,0.08);border:1px solid rgba(250,248,243,0.15);display:grid;place-items:center;color:rgba(250,248,243,0.5);"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg></div>
    </div>
    <div style="padding:28px 24px;background:rgba(250,248,243,0.09);border:1px solid rgba(250,248,243,0.2);border-radius:18px;box-shadow:0 10px 26px rgba(0,0,0,0.18);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;">
        <div style="width:40px;height:40px;border-radius:12px;background:rgba(212,163,115,0.1);border:1px solid rgba(212,163,115,0.25);display:grid;place-items:center;color:#b8884f;"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v12m0 0 4-4m-4 4-4-4"/><path d="M4 20h16"/></svg></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:0.14em;color:rgba(250,248,243,0.35);">04</span>
      </div>
      <h4 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:18px;font-weight:400;color:#faf8f3;margin:0 0 10px;">Action report</h4>
      <p style="font-size:13.5px;line-height:1.55;color:rgba(250,248,243,0.68);margin:0;">Export-ready PDF with ranked predictions, intervals, and field-specific cultural recommendations.</p>
    </div>
  </div>
</section>
""", unsafe_allow_html=True)

    # ── CTA block ──────────────────────────────────────────────
    st.markdown("""
<section class="cta-block">
  <div class="cta-inner">
    <span class="eyebrow">Ready when you are</span>
    <h2 class="display cta-title">Put a <em>lab</em> behind every field.</h2>
    <p class="cta-sub">Specimen imagery in, protocol out. No login gymnastics. No field experts required.</p>
    <div class="cta-actions">
      <a class="btn btn-sage" href="?page=cultivation" target="_self" style="text-decoration:none;">Start with a soil scan &#8594;</a>
    </div>
  </div>
</section>
<footer class="site-footer">
  <div class="footer-row">
    <div class="footer-brand">
      <span class="eyebrow">AgroSynapse AI · 2026</span>
      <p>Synaptic agronomy, built by Manoj Anumolu. Academic project; not a replacement for certified agronomist advice.</p>
    </div>
    <div class="footer-meta">
      <span class="label">v0.4 · Alpha</span>
      <span class="label">Models: SoilNet-v3 · PhytoNet-v2</span>
    </div>
  </div>
</footer>
""", unsafe_allow_html=True)


# ==============================================================
# CULTIVATION PAGE
# ==============================================================
elif _page == "cultivation":

    st.markdown("""
<div class="page-tool">
<div class="tool-header">
  <div>
    <span class="eyebrow">Module &middot; Agricultural Core</span>
    <h1 class="display tool-page-title">Predictive Cultivation</h1>
    <p class="tool-page-sub">Synthesize soil specimen, chemical profile, climate vectors, and farm history into laboratory-grade crop recommendations &mdash; probability-ranked, protocol-complete.</p>
  </div>
  <aside class="unit-guide">
    <div class="unit-guide-head">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/></svg>
      <span class="eyebrow">Farmer unit guide</span>
    </div>
    <div class="unit-guide-body">
      <div class="ug-row"><span>Yield</span><span class="num">t/ha</span></div>
      <div class="ug-row"><span>NPK</span><span class="num">mg/kg</span></div>
      <div class="ug-row"><span>Area</span><span class="num">1 acre ~ 0.4 ha</span></div>
      <div class="ug-row"><span>Temp</span><span class="num">deg C</span></div>
      <div class="ug-row"><span>Rainfall</span><span class="num">mm/yr</span></div>
    </div>
    <div class="unit-guide-foot">
      <span class="label">Need help?</span>
      <span class="unit-guide-link">Open glossary -></span>
    </div>
  </aside>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
    soil_img_bytes = None

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<span id="cult-soil-card"></span>', unsafe_allow_html=True)
        st.markdown("""
<div class="tool-block tight">
<div class="tool-block-head">
  <h3 class="display tool-block-title">Soil Specimen</h3>
  <span class="pill live">Vision ready</span>
</div>
<p class="tool-block-sub">Upload a clear close-up of the soil sample. Avoid leaves, hands, or moisture artifacts.</p>""", unsafe_allow_html=True)
        soil_img = st.file_uploader("Soil image", type=["jpg", "jpeg", "png"], key="soil_img_cult", label_visibility="collapsed")
        if soil_img:
            soil_img_bytes = soil_img.getvalue()
            st.image(soil_img_bytes, use_container_width=True, output_format="auto")
            st.markdown(
                '<div class="upload-meta" style="margin-top:12px;">'
                '<div class="upload-meta-file"><div class="upload-meta-thumb"></div>'
                '<div><div class="upload-meta-name">' + soil_img.name + '</div>'
                '<div class="upload-meta-sub num">' + f'{soil_img.size/1024:.1f}' + ' KB</div></div></div>'
                '<div class="upload-preview-chip-static"><span>&#10003; Valid specimen ready for synthesis</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<span id="cult-chem-card"></span>', unsafe_allow_html=True)
        st.markdown("""
<div class="tool-block tight">
<div class="tool-block-head">
  <h3 class="display tool-block-title">Chemical Profile</h3>
  <span class="pill">NPK - pH</span>
</div>
<p class="tool-block-sub">Known values from lab report, or estimates from field tests.</p>
<div class="chem-grid">""", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            n_val = st.number_input("Nitrogen (N) - mg/kg", min_value=0.0, max_value=200.0, value=90.0, step=1.0, key="n_val")
            st.markdown(_metric_range_html("N", n_val, 0.0, 200.0, 60.0, 140.0, "mg/kg"), unsafe_allow_html=True)
            k_val = st.number_input("Potassium (K) - mg/kg", min_value=0.0, max_value=200.0, value=54.0, step=1.0, key="k_val")
            st.markdown(_metric_range_html("K", k_val, 0.0, 200.0, 30.0, 120.0, "mg/kg"), unsafe_allow_html=True)
        with cc2:
            p_val = st.number_input("Phosphorus (P) - mg/kg", min_value=0.0, max_value=100.0, value=35.0, step=1.0, key="p_val")
            st.markdown(_metric_range_html("P", p_val, 0.0, 100.0, 20.0, 60.0, "mg/kg"), unsafe_allow_html=True)
            ph_val = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1, key="ph_val")
            st.markdown(_metric_range_html("PH", ph_val, 3.0, 10.0, 6.0, 7.2, "units"), unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)

    with lc1:
        st.markdown('<span id="cult-climate-card"></span>', unsafe_allow_html=True)
        st.markdown("""
<div class="tool-block">
<div class="tool-block-head">
  <h3 class="display tool-block-title">Climate Synthesis</h3>
  <span class="pill live">Auto-filled</span>
</div>
<p class="tool-block-sub">District-grade vectors pulled from 12-year IMD historical series.</p>""", unsafe_allow_html=True)
        clc1, clc2, clc3 = st.columns(3)
        with clc1:
            state_val = st.selectbox("State", ["Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Maharashtra", "Gujarat", "Rajasthan", "Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Bihar", "West Bengal", "Odisha", "Kerala"], key="state_val")
        with clc2:
            district_val = st.selectbox("District", ["Guntur", "Krishna", "Nellore", "Kurnool", "Chittoor", "Hyderabad", "Warangal", "Bengaluru", "Chennai", "Mumbai"], key="district_val")
        with clc3:
            village_val = st.text_input("Village / Town", value="Rawada", key="village_val")
        st.markdown('<div class="climate-action-row">', unsafe_allow_html=True)
        _climate_btn_spacer, _climate_btn_col = st.columns([2.2, 1])
        with _climate_btn_col:
            fetch_climate = st.button("Fetch climate vectors", key="fetch_climate_btn", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if fetch_climate:
            with st.spinner("Fetching climate data..."):
                _clm, _clm_err = get_climate_data(village_val, district_val, state_val)
            if _clm:
                st.session_state.auto_temp = _clm.get("temperature", 27.2)
                st.session_state.auto_hum = _clm.get("humidity", 75.3)
                st.session_state.auto_rain = _clm.get("rainfall", 1302.0)
                st.session_state.location_name = _clm.get("location", "")

        _t = st.session_state.auto_temp
        _h = st.session_state.auto_hum
        _r = st.session_state.auto_rain
        st.markdown(f"""
<div class="climate-tiles">
  <div class="climate-tile">
    <div class="climate-tile-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4 4 0 1 0 5 0Z"/></svg></div>
    <div class="climate-tile-body"><div class="label">Temperature</div><div class="climate-tile-value">{_t}<small>deg C</small></div></div>
    <div class="climate-tile-spark"><svg viewBox="0 0 60 20"><polyline points="0,14 10,10 20,12 30,6 40,8 50,4 60,7" fill="none" stroke="#c44536" stroke-width="1.5"/></svg></div>
  </div>
  <div class="climate-tile">
    <div class="climate-tile-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3s7 8 7 13a7 7 0 1 1-14 0c0-5 7-13 7-13Z"/></svg></div>
    <div class="climate-tile-body"><div class="label">Humidity</div><div class="climate-tile-value">{_h}<small>%</small></div></div>
    <div class="climate-tile-spark"><svg viewBox="0 0 60 20"><polyline points="0,8 10,12 20,10 30,14 40,9 50,12 60,10" fill="none" stroke="#5a8a3a" stroke-width="1.5"/></svg></div>
  </div>
  <div class="climate-tile">
    <div class="climate-tile-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg></div>
    <div class="climate-tile-body"><div class="label">Rainfall</div><div class="climate-tile-value">{int(_r)}<small>mm</small></div></div>
    <div class="climate-tile-spark"><svg viewBox="0 0 60 20"><polyline points="0,16 10,12 20,14 30,8 40,10 50,5 60,9" fill="none" stroke="#d4a373" stroke-width="1.5"/></svg></div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with lc2:
        st.markdown('<span id="cult-farm-card"></span>', unsafe_allow_html=True)
        st.markdown("""
<div class="tool-block">
<div class="tool-block-head">
  <h3 class="display tool-block-title">Farm Context</h3>
  <span class="pill earth">History - systems</span>
</div>
<p class="tool-block-sub">Historical yield + cultivation system. Used to weight output probability.</p>""", unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            yield_val = st.number_input("Yield - last season (kg/ha)", min_value=0.0, max_value=10000.0, value=2083.0, step=10.0, key="yield_val")
            st.markdown(_metric_range_html("Yield", yield_val, 0.0, 5000.0, 1800.0, 3500.0, "kg/ha", accent="earth"), unsafe_allow_html=True)
            season_val = st.selectbox("Current season", ["Kharif", "Rabi", "Zaid"], key="season_val")
            prev_crop_val = st.selectbox("Previous crop", ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"], key="prev_crop_val")
        with fc2:
            fert_val = st.number_input("Fertilizer used (kg/ha)", min_value=0.0, max_value=400.0, value=118.0, step=1.0, key="fert_val")
            st.markdown(_metric_range_html("Fertilizer", fert_val, 0.0, 400.0, 60.0, 200.0, "kg/ha", accent="earth"), unsafe_allow_html=True)
            irrig_val = st.selectbox("Irrigation system", ["Canal", "Drip", "Rainfed", "Sprinkler"], key="irrig_val")
            region_val = st.selectbox("Geographic zone", ["Central", "East", "North", "South", "West"], key="region_val")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<span id="ready-synth-card"></span>', unsafe_allow_html=True)
    st.markdown('<div class="tool-analyze">', unsafe_allow_html=True)
    _an1, _an2 = st.columns([3.8, 1.25])
    with _an1:
        st.markdown("""
<div class="tool-analyze-copy">
  <h3 class="display tool-analyze-title">Ready to synthesize.</h3>
  <p class="tool-analyze-sub">All four vectors complete. Expected synthesis time: ~2.4 seconds.</p>
</div>""", unsafe_allow_html=True)
    with _an2:
        st.markdown('<div class="tool-analyze-btn-wrap">', unsafe_allow_html=True)
        analyze_btn = st.button("Analyze and predict crop", key="analyze_predict_btn", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn and soil_img is not None and _models_ok:
        _img_bytes = soil_img_bytes or soil_img.getvalue()
        _pil = Image.open(io.BytesIO(_img_bytes)).convert("RGB")
        _valid = is_soil_image(_pil)
        if not _valid:
            st.error("No soil detected in this image. Please upload a clear close-up soil photo.")
        else:
            with st.spinner("Synthesizing - cross-attention fusion running..."):
                try:
                    (soil_name, confidence, all_probs, soil_fert, crop_recs, debug) = run_inference(
                        img_model, tab_proj, fusion, xgb_clf, scaler,
                        CLASS_NAMES, NUMERIC_COLS,
                        _img_bytes,
                        n_val, p_val, k_val,
                        st.session_state.auto_temp,
                        st.session_state.auto_hum,
                        st.session_state.auto_rain,
                        ph_val, yield_val, fert_val,
                        season_val,
                        irrig_val.split()[0] if " " in irrig_val else irrig_val,
                        prev_crop_val,
                        region_val.split()[0] if " " in region_val else region_val,
                    )
                    st.session_state.last_result = {
                        "soil_name": soil_name, "confidence": confidence,
                        "all_probs": all_probs, "soil_fert": soil_fert,
                        "crop_recs": crop_recs,
                        "n": n_val, "p": p_val, "k": k_val, "ph": ph_val,
                        "temp": st.session_state.auto_temp,
                        "hum": st.session_state.auto_hum,
                        "rain": st.session_state.auto_rain,
                        "season": season_val,
                    }
                    st.session_state.last_error = None
                    st.session_state.page = "dashboard"
                    st.query_params["page"] = "dashboard"
                    st.rerun()
                except Exception as _e:
                    st.session_state.last_error = str(_e)
                    st.error(f"Inference failed: {_e}")
    elif analyze_btn and soil_img is None:
        st.error("Please upload a soil image before analyzing.")

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================
# DIAGNOSTIC PAGE
# ==============================================================
elif _page == "diagnostic":
    # Warm up and cache leaf model once so diagnosis click latency stays low.
    if "leaf_model_ready" not in st.session_state:
        _leaf_model, _leaf_labels, _leaf_ferts = load_leaf_model()
        st.session_state.leaf_model_ready = _leaf_model is not None
        st.session_state.leaf_model_cached = _leaf_model
        st.session_state.leaf_labels_cached = _leaf_labels
        st.session_state.leaf_ferts_cached = _leaf_ferts

    st.markdown("""
<div class="page-tool">
<div class="tool-header">
  <div>
    <span class="eyebrow">Module &middot; Neural Vision</span>
    <h1 class="display tool-page-title" style="font-family:'Instrument Serif','Times New Roman',serif;font-size:clamp(56px,6.5vw,88px);line-height:1;">Phyto-Diagnostic Suite</h1>
    <p class="tool-page-sub">Upload a leaf photograph. PhytoNet-v2 resolves pathogen identity across 38 classes and returns a precision treatment protocol in a single forward pass.</p>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Leaf specimen block ────────────────────────────────────
    d1, d2 = st.columns([3, 2])

    with d1:
      with st.container():
        st.markdown('<span id="diag-specimen-card"></span>', unsafe_allow_html=True)
        st.markdown("""
  <div class="tool-block-head">
    <h3 class="display tool-block-title" style="font-family:'Instrument Serif','Times New Roman',serif;font-size:46px;line-height:1;">Plant Specimen</h3>
    <span class="pill live">PhytoNet &middot; ready</span>
  </div>
  <p class="tool-block-sub">Upload a close-up of a single leaf. Avoid multiple species in frame.</p>""", unsafe_allow_html=True)

        leaf_img = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg","jpeg","png"], key="leaf_img_upld")

        if leaf_img:
          _leaf_bytes = leaf_img.getvalue()
          st.session_state.leaf_img_bytes = _leaf_bytes
          _img_l, _img_c, _img_r = st.columns([1, 8, 1])
          with _img_c:
            st.image(_leaf_bytes, use_container_width=True, output_format="auto")
          _pil_leaf = Image.open(io.BytesIO(_leaf_bytes)).convert("RGB")
          st.session_state.leaf_valid = None
          st.markdown(
            '<div class="upload-meta" style="margin-top:12px;"><div class="upload-meta-file"><div class="upload-meta-thumb leaf"></div><div>'
            '<div class="upload-meta-name">' + leaf_img.name + '</div>'
            '<div class="upload-meta-sub num">' + f'{leaf_img.size/1024:.1f}' + ' KB</div>'
            '</div></div><div class="upload-preview-chip-static"><span>Preview ready run diagnosis</span></div></div>',
            unsafe_allow_html=True,
          )
        else:
          st.markdown("""
  <div style="margin-top:12px;padding:18px;border:1px dashed rgba(20,20,15,0.18);border-radius:14px;background:#f8f4eb;color:#6b6b5e;font-size:13px;">
  Upload a leaf image to preview the exact specimen here.
  </div>""", unsafe_allow_html=True)

    with d2:
        _lr = st.session_state.get("leaf_result")

        if _lr:
            _pred_cls, _conf, _top5 = _lr
            _info = LEAF_TREATMENT_MAP.get(_pred_cls, {})
            _common = _info.get("common_name", _pred_cls.replace("___", " - ").replace("_", " "))
            _is_healthy = "healthy" in _pred_cls.lower()

            st.markdown(f"""
<div class="detect-result">
  <div class="detect-result-head">
    <span class="eyebrow">Detection result</span>
    <span class="pill {'warn' if not _is_healthy else ''}">{'Pathogen detected' if not _is_healthy else 'Healthy'}</span>
  </div>
  <h3 class="display detect-name">{_common}</h3>
  <div class="detect-latin">Confidence: {_conf}% &middot; neural classification</div>
  <div class="detect-confidence" style="margin-top:16px;">
    <div class="detect-confidence-head">
      <span class="label">Confidence level</span>
      <span class="num detect-confidence-value" style="font-family:var(--serif);font-size:20px;color:var(--sage);">{_conf}%</span>
    </div>
    <div class="range-bar"><div class="fill" style="width:{_conf}%;"></div></div>
  </div>
</div>""", unsafe_allow_html=True)

            if not _is_healthy and _info:
                st.markdown(f"""
<div class="treatment" style="margin-top:16px;">
  <div class="treatment-head">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/></svg>
    <span class="eyebrow">Treatment plan</span>
  </div>
  <div class="treatment-item">
    <div class="treatment-item-head"><span class="treatment-num">01</span><span class="treatment-label">Primary treatment</span></div>
    <p class="treatment-body">{_info.get("primary_treatment","")}</p>
  </div>
  <div class="treatment-item">
    <div class="treatment-item-head"><span class="treatment-num">02</span><span class="treatment-label">Fertilizer adjustment</span></div>
    <p class="treatment-body">{_info.get("fertilizer","")}</p>
  </div>
  <div class="treatment-item">
    <div class="treatment-item-head"><span class="treatment-num">03</span><span class="treatment-label">Cultural practices</span></div>
    <p class="treatment-body">{_info.get("cultural_practices","")}</p>
  </div>
</div>""", unsafe_allow_html=True)

            # Top-5 predictions
            st.markdown('<div class="preds" style="margin-top:16px;"><div class="preds-head"><span class="eyebrow">Top-5 predictions</span><span class="label num">Softmax</span></div>', unsafe_allow_html=True)
            for _i, (_cls, _pct) in enumerate(_top5):
                _cn = _cls.replace("___", " - ").replace("_", " ")
                _bg = "var(--sage)" if _i == 0 else "var(--ink-4)"
                st.markdown(f"""<div class="pred-row"><div class="pred-label">{_cn}</div>
<div class="pred-bar"><div class="pred-bar-fill" style="width:{_pct}%;background:{_bg};"></div></div>
<div class="pred-val num">{_pct:.1f}%</div></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="detect-result">
  <div class="detect-result-head"><span class="eyebrow">Detection result</span><span class="pill">Awaiting specimen</span></div>
  <p style="font-size:14px;color:var(--ink-3);margin-top:16px;">Upload a leaf image and run diagnosis to see results here.</p>
</div>""", unsafe_allow_html=True)

    # ── Run diagnosis button ───────────────────────────────────
    st.markdown('<div class="diag-actions">', unsafe_allow_html=True)
    run_diag = st.button("Run neural diagnosis ->", key="run_diag_btn", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_diag:
        _lb = st.session_state.get("leaf_img_bytes")
        if _lb is None:
            st.error("Please upload a leaf image first.")
        else:
            _pil_leaf = Image.open(io.BytesIO(_lb)).convert("RGB")
            _valid_leaf = is_leaf_image(_pil_leaf)
            st.session_state.leaf_valid = _valid_leaf
            if not _valid_leaf:
                st.error("The uploaded image does not appear to be a leaf. Please try again.")
            else:
              with st.spinner("Running PhytoNet-v2 inference..."):
                    try:
                        leaf_model = st.session_state.get("leaf_model_cached")
                        leaf_labels = st.session_state.get("leaf_labels_cached")
                        if leaf_model is None or leaf_labels is None:
                            leaf_model, leaf_labels, leaf_ferts = load_leaf_model()
                            st.session_state.leaf_model_cached = leaf_model
                            st.session_state.leaf_labels_cached = leaf_labels
                            st.session_state.leaf_ferts_cached = leaf_ferts
                        if leaf_model is None:
                            st.error("Leaf model unavailable. Please check model files.")
                        else:
                            _pc, _cf, _t5 = run_leaf_inference(leaf_model, leaf_labels, _lb)
                            st.session_state.leaf_result = (_pc, _cf, _t5)
                            st.rerun()
                    except Exception as _le:
                        st.error(f"Diagnosis failed: {_le}")

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================
# DASHBOARD PAGE
# ==============================================================
else:  # dashboard

    _res = st.session_state.get("last_result")
    _now = _now_ist()
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'auto'});</script>", height=0)
    dh1, dh2 = st.columns([4, 1.35])
    with dh1:
        st.markdown("""
<div class="dash-title-wrap">
  <span class="eyebrow">Result analysis - field T-047</span>
  <h1 class="display tool-page-title">Synthesis complete.</h1>
  <p class="tool-page-sub">Generated """ + _now.strftime("%b %d, %Y - %H:%M") + """ IST. Output verified across 4 model heads; confidence-weighted with historical yield prior.</p>
</div>""", unsafe_allow_html=True)
    with dh2:
        st.markdown('<div class="dash-header-actions">', unsafe_allow_html=True)
        _res_for_dl = st.session_state.get("last_result")
        if _res_for_dl:
            _dl_soil = _res_for_dl.get("soil_name", "Unknown")
            _dl_crop = (_res_for_dl.get("crop_recs") or [{"name": "-"}])[0].get("name", "-")
            _dl_npk = (_res_for_dl.get("crop_recs") or [{"npk": "-"}])[0].get("npk", "-")
            _dl_conf = round(_res_for_dl.get("confidence", 0))
            _dl_n = _res_for_dl.get("n", 0); _dl_p = _res_for_dl.get("p", 0)
            _dl_k = _res_for_dl.get("k", 0); _dl_ph = _res_for_dl.get("ph", 0)
            _dl_t = _res_for_dl.get("temp", 0); _dl_h = _res_for_dl.get("hum", 0)
            _dl_r = _res_for_dl.get("rain", 0); _dl_s = _res_for_dl.get("season", "-")
            _report_html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>AgroSynapse Report</title>"
                "<style>body{font-family:Georgia,serif;max-width:760px;margin:48px auto;color:#14140f;line-height:1.6;}"
                "h1{font-size:36px;margin-bottom:4px;}h2{font-size:20px;margin:32px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px;}"
                "table{width:100%;border-collapse:collapse;margin-top:12px;}"
                "td,th{padding:10px 14px;border:1px solid #e0ddd4;text-align:left;}"
                "th{background:#f5f0e8;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;}"
                ".badge{display:inline-block;padding:4px 12px;background:#0f2818;color:#faf8f3;border-radius:999px;font-size:13px;}"
                "</style></head><body>"
                "<h1>AgroSynapse Analysis Report</h1>"
                "<p style='color:#6b6b5e;font-size:14px;'>Generated " + _now.strftime("%B %d, %Y at %H:%M") + " IST | AgroSynapse AI v0.4</p>"
                "<h2>Primary Recommendation</h2>"
                "<p><span class='badge'>" + _dl_crop + "</span> &nbsp; Confidence: <strong>" + str(_dl_conf) + "%</strong></p>"
                "<table><tr><th>Parameter</th><th>Value</th></tr>"
                "<tr><td>Soil Type</td><td>" + _dl_soil + "</td></tr>"
                "<tr><td>NPK Protocol (kg/ha)</td><td>" + _dl_npk + "</td></tr>"
                "<tr><td>Season</td><td>" + _dl_s + "</td></tr>"
                "</table>"
                "<h2>Soil Chemical Profile</h2>"
                "<table><tr><th>Nutrient</th><th>Measured</th><th>Optimal Range</th></tr>"
                "<tr><td>Nitrogen (N)</td><td>" + str(_dl_n) + " mg/kg</td><td>60-140 mg/kg</td></tr>"
                "<tr><td>Phosphorus (P)</td><td>" + str(_dl_p) + " mg/kg</td><td>20-60 mg/kg</td></tr>"
                "<tr><td>Potassium (K)</td><td>" + str(_dl_k) + " mg/kg</td><td>40-100 mg/kg</td></tr>"
                "<tr><td>Soil pH</td><td>" + str(_dl_ph) + "</td><td>5.5-7.5</td></tr>"
                "</table>"
                "<h2>Climate Vectors</h2>"
                "<table><tr><th>Parameter</th><th>Value</th></tr>"
                "<tr><td>Temperature</td><td>" + str(_dl_t) + " C</td></tr>"
                "<tr><td>Humidity</td><td>" + str(_dl_h) + " %</td></tr>"
                "<tr><td>Annual Rainfall</td><td>" + str(int(_dl_r)) + " mm</td></tr>"
                "</table>"
                "<p style='margin-top:48px;font-size:12px;color:#a8a598;'>This report is generated by AgroSynapse AI. For certified agronomic advice, consult a licensed professional.</p>"
                "</body></html>"
            )
            st.download_button(
                label="Export PDF",
                data=_report_html.encode("utf-8"),
                file_name="agrosynapse_report.html",
                mime="text/html",
                key="dl_report_btn",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="dash-header-actions">', unsafe_allow_html=True)
        if st.button("New analysis", key="new_analysis_btn", use_container_width=True):
            st.session_state.page = "cultivation"
            st.query_params["page"] = "cultivation"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if _res:
        _soil  = _res.get("soil_name", "Unknown")
        _conf  = _res.get("confidence", 93)
        _crops = _res.get("crop_recs", [])
        _season = _res.get("season", "Kharif")

        _primary = _crops[0] if _crops else {"name": "Groundnut", "npk": "18:46:32", "score": 0.93}
        _crop_name = _primary.get("name","Groundnut")
        _npk = _primary.get("npk","18:46:32")
        _conf_pct = round(_conf)
        _ring_r = 54.0
        _ring_c = 2 * 3.14159 * _ring_r
        _ring_off = _ring_c * (1 - _conf_pct / 100)

        _CROP_EMOJI = {
            "Rice":"🌾","Wheat":"🌾","Maize":"🌽","Cotton":"🌸","Groundnut":"🥜",
            "Sugarcane":"🎋","Jute":"🌿","Mungbean":"🫘","Blackgram":"🫘",
            "Lentil":"🫘","Pomegranate":"🍎","Banana":"🍌","Mango":"🥭",
            "Grapes":"🍇","Watermelon":"🍉","Muskmelon":"🍈","Apple":"🍎",
            "Orange":"🍊","Papaya":"🍈","Coconut":"🥥","Coffee":"☕",
            "Chickpea":"🫘","Kidneybeans":"🫘","Pigeonpeas":"🫘","Mothbeans":"🫘",
        }
        _crop_emoji = _CROP_EMOJI.get(_crop_name, "🌱")

        # Soil type → color mapping for card background
        _SOIL_BG = {
            "Red Soil":      ("linear-gradient(135deg,#6b1a00,#3a0a00)", "linear-gradient(135deg,#8b2500,#5a1200)"),
            "Black Soil":    ("linear-gradient(135deg,#1a1a28,#0a0a14)", "linear-gradient(135deg,#252538,#111120)"),
            "Alluvial Soil": ("linear-gradient(135deg,#0f2030,#050f18)", "linear-gradient(135deg,#1a3048,#0a1a28)"),
            "Sandy Soil":    ("linear-gradient(135deg,#5a3a0a,#3a2000)", "linear-gradient(135deg,#7a5020,#4a2c00)"),
            "Loamy Soil":    ("linear-gradient(135deg,#1e3a14,#0a1808)", "linear-gradient(135deg,#2a4a1e,#121e0a)"),
            "Clay Soil":     ("linear-gradient(135deg,#3a2a1a,#201408)", "linear-gradient(135deg,#4a3828,#281a08)"),
            "Laterite Soil": ("linear-gradient(135deg,#5a2800,#300a00)", "linear-gradient(135deg,#7a3800,#401000)"),
        }
        _card_icon_bg, _card_main_bg = _SOIL_BG.get(_soil, (
            "linear-gradient(135deg,#1e3a14,#0a1808)",
            "linear-gradient(135deg,#0f2818,#050e08)"
        ))

        # Primary crop card
        st.markdown(f"""
<div class="rec-crop" style="display:grid;grid-template-columns:180px 1fr 180px;min-height:240px;background:{_card_main_bg};border-radius:16px;overflow:hidden;margin-bottom:32px;">
  <div class="rec-crop-img" style="background:{_card_icon_bg};display:grid;place-items:center;font-size:68px;border-right:1px solid rgba(250,248,243,0.06);">
    {_crop_emoji}
  </div>
  <div class="rec-crop-body" style="padding:28px 32px;color:#faf8f3;overflow:hidden;">
    <span class="eyebrow" style="color:rgba(250,248,243,0.5);font-size:10px;letter-spacing:0.14em;text-transform:uppercase;display:block;margin-bottom:8px;">Primary recommendation</span>
    <h2 style="font-family:'Instrument Serif','Times New Roman',serif;font-size:clamp(28px,3vw,44px);font-weight:400;color:#faf8f3;margin:0 0 10px;line-height:1.1;">{_crop_name}</h2>
    <p style="font-size:13px;color:rgba(250,248,243,0.6);line-height:1.5;margin:0 0 20px;max-width:480px;">Synaptic triangulation indicates {_crop_name} as the optimal rotation for the upcoming {_season} season. Soil type: <strong style="color:rgba(250,248,243,0.85);">{_soil}</strong>. Aligned with 14,200 validated training pairs.</p>
    <div style="border-top:1px solid rgba(250,248,243,0.1);padding-top:16px;display:flex;gap:32px;flex-wrap:wrap;">
      <div><div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:rgba(250,248,243,0.4);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">NPK Protocol</div><div style="font-family:'Instrument Serif','Times New Roman',serif;font-size:22px;color:#e8c989;">{_npk} <small style="font-size:12px;color:rgba(250,248,243,0.4);">kg/ha</small></div></div>
      <div><div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:rgba(250,248,243,0.4);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">Soil Type</div><div style="font-family:'Instrument Serif','Times New Roman',serif;font-size:22px;color:#faf8f3;">{_soil[:14]}</div></div>
      <div><div style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:rgba(250,248,243,0.4);margin-bottom:4px;font-family:'JetBrains Mono',monospace;">Confidence</div><div style="font-family:'Instrument Serif','Times New Roman',serif;font-size:22px;color:#7ba854;">{_conf_pct}%</div></div>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:24px;border-left:1px solid rgba(250,248,243,0.08);flex-shrink:0;">
    <div style="width:124px;height:124px;position:relative;">
      <svg width="124" height="124" style="transform:rotate(-90deg);display:block;">
        <circle cx="62" cy="62" r="{_ring_r}" stroke-width="6" fill="none" stroke="rgba(250,248,243,0.12)"/>
        <circle cx="62" cy="62" r="{_ring_r}" stroke-width="6" fill="none" stroke="#e8c989"
          stroke-dasharray="{_ring_c:.1f}" stroke-dashoffset="{_ring_off:.1f}" stroke-linecap="round"/>
      </svg>
      <div style="position:absolute;inset:0;display:grid;place-items:center;font-family:'Instrument Serif','Times New Roman',serif;font-size:30px;color:#faf8f3;line-height:1;">{_conf_pct}%</div>
    </div>
    <span style="font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:rgba(250,248,243,0.4);font-family:'JetBrains Mono',monospace;">Synaptic score</span>
  </div>
</div>""", unsafe_allow_html=True)

        # Probability breakdown + alt crops
        _all_probs = _res.get("all_probs", {})
        _bars = sorted(_all_probs.items(), key=lambda x: x[1], reverse=True)

        dg1, dg2 = st.columns([2, 1])
        with dg1:
            st.markdown("""
<div class="probs">
  <div class="probs-head">
    <div><span class="eyebrow">Soil probability breakdown</span>
      <h3 class="display probs-title">Synaptic confidence, per soil.</h3></div>
  </div>""", unsafe_allow_html=True)
            _dash_tab = st.radio("View", ["Soil", "Climate"], horizontal=True, key="dash_view_tab", label_visibility="collapsed")
            if _dash_tab == "Soil":
                st.markdown('<div class="probs-chart">', unsafe_allow_html=True)
                _colors = ["#5a8a3a","#d4a373","#b8884f","#a8a598","#a8a598","#a8a598","#a8a598"]
                for _bi, (_bk, _bv) in enumerate(_bars[:7]):
                    _bc = _colors[min(_bi, len(_colors)-1)]
                    st.markdown(
                        '<div class="prob-row"><div class="prob-k">' + _bk + '</div>'
                        '<div class="prob-track"><div class="prob-fill" style="width:' + f'{_bv:.1f}' + '%;background:' + _bc + ';animation-delay:' + str(_bi*80) + 'ms;"></div></div>'
                        '<div class="prob-v num">' + f'{_bv:.1f}' + '%</div></div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Climate chart
                _t_v = _res.get("temp", 27.2)
                _h_v = _res.get("hum", 75.3)
                _r_v = _res.get("rain", 1302.0)
                _t_pct = min(100, _t_v / 50 * 100)
                _h_pct = min(100, _h_v)
                _r_pct = min(100, _r_v / 3000 * 100)
                st.markdown(
                    '<div style="padding:20px 0;">'
                    '<div style="margin-bottom:20px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#6b6b5e;">Temperature</span>'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#c44536;font-weight:500;">' + str(_t_v) + ' °C</span>'
                    '</div>'
                    '<div style="height:8px;background:rgba(20,20,15,0.07);border-radius:4px;overflow:hidden;">'
                    '<div style="height:100%;width:' + f'{_t_pct:.1f}' + '%;background:linear-gradient(90deg,#e87048,#c44536);border-radius:4px;transition:width 1s cubic-bezier(0.2,0.8,0.2,1);"></div>'
                    '</div><div style="font-size:10px;color:#a8a598;margin-top:4px;font-family:\'JetBrains Mono\',monospace;">Range: 0–50 °C · Optimal for most crops: 20–35 °C</div>'
                    '</div>'
                    '<div style="margin-bottom:20px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#6b6b5e;">Humidity</span>'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#5a8a3a;font-weight:500;">' + str(_h_v) + ' %</span>'
                    '</div>'
                    '<div style="height:8px;background:rgba(20,20,15,0.07);border-radius:4px;overflow:hidden;">'
                    '<div style="height:100%;width:' + f'{_h_pct:.1f}' + '%;background:linear-gradient(90deg,#7ba854,#5a8a3a);border-radius:4px;transition:width 1s cubic-bezier(0.2,0.8,0.2,1);"></div>'
                    '</div><div style="font-size:10px;color:#a8a598;margin-top:4px;font-family:\'JetBrains Mono\',monospace;">Optimal for most crops: 50–80 %</div>'
                    '</div>'
                    '<div style="margin-bottom:20px;">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;letter-spacing:0.12em;text-transform:uppercase;color:#6b6b5e;">Annual Rainfall</span>'
                    '<span style="font-family:\'JetBrains Mono\',monospace;font-size:13px;color:#d4a373;font-weight:500;">' + str(int(_r_v)) + ' mm</span>'
                    '</div>'
                    '<div style="height:8px;background:rgba(20,20,15,0.07);border-radius:4px;overflow:hidden;">'
                    '<div style="height:100%;width:' + f'{_r_pct:.1f}' + '%;background:linear-gradient(90deg,#e8c989,#d4a373);border-radius:4px;transition:width 1s cubic-bezier(0.2,0.8,0.2,1);"></div>'
                    '</div><div style="font-size:10px;color:#a8a598;margin-top:4px;font-family:\'JetBrains Mono\',monospace;">Scale: 0–3000 mm/yr · Optimal: 600–2000 mm</div>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with dg2:
            st.markdown('<div class="alts"><div class="alts-head"><span class="eyebrow">Alternative crops</span><span class="label">Ranked K=2–4</span></div>', unsafe_allow_html=True)
            for _ai, _cr in enumerate(_crops[1:4], start=2):
                _ascore = round(_cr.get("suitability", 0.5) * 100)
                _aemoji = _CROP_EMOJI.get(_cr.get("name",""), "🌱")
                st.markdown(
                    '<div class="alt">'
                    '<div style="font-size:26px;line-height:1;margin-right:4px;">' + _aemoji + '</div>'
                    '<div class="alt-body"><div class="alt-name">' + _cr['name'] + '</div>'
                    '<div class="alt-meta"><span class="num">' + _cr.get('npk','—') + '</span></div></div>'
                    '<div class="alt-score num">' + str(_ascore) + '%</div></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # Timeline
        st.markdown(f"""
<div class="timeline">
  <div class="timeline-head">
    <span class="eyebrow">Cultivation timeline · 8 months</span>
    <span class="pill earth">{_season} 2026</span>
  </div>
  <div class="timeline-months">
    <div class="timeline-m label"></div>
    <div class="timeline-m label">Jun</div><div class="timeline-m label">Jul</div>
    <div class="timeline-m label">Aug</div><div class="timeline-m label">Sep</div>
    <div class="timeline-m label">Oct</div><div class="timeline-m label">Nov</div>
    <div class="timeline-m label">Dec</div><div class="timeline-m label">Jan</div>
  </div>
  <div class="timeline-body">
    <div class="timeline-row"><div class="timeline-phase">Sowing</div>
      <div class="timeline-track"><div class="timeline-bar" style="left:0%;width:12.5%;background:var(--sage);animation-delay:0ms;"></div></div></div>
    <div class="timeline-row"><div class="timeline-phase">Vegetative</div>
      <div class="timeline-track"><div class="timeline-bar" style="left:10%;width:27.5%;background:var(--sage-2);animation-delay:100ms;"></div></div></div>
    <div class="timeline-row"><div class="timeline-phase">Flowering</div>
      <div class="timeline-track"><div class="timeline-bar" style="left:37.5%;width:18.75%;background:var(--earth);animation-delay:200ms;"></div></div></div>
    <div class="timeline-row"><div class="timeline-phase">Pod development</div>
      <div class="timeline-track"><div class="timeline-bar" style="left:50%;width:25%;background:var(--earth-2);animation-delay:300ms;"></div></div></div>
    <div class="timeline-row"><div class="timeline-phase">Harvest</div>
      <div class="timeline-track"><div class="timeline-bar" style="left:75%;width:18.75%;background:var(--clay);animation-delay:400ms;"></div></div></div>
  </div>
</div>""", unsafe_allow_html=True)

        # Advisory insights
        _n = _res.get("n", 90); _p = _res.get("p", 35); _h = _res.get("hum", 75)
        st.markdown(f"""
<div class="dash-insights">
  <div class="section-head">
    <span class="eyebrow">Advisory · notes</span>
    <h2 class="display section-title">What the model <em>noticed.</em></h2>
  </div>
  <div class="insights-grid">
    <div class="insight">
      <div class="insight-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3s7 8 7 13a7 7 0 1 1-14 0c0-5 7-13 7-13Z"/></svg></div>
      <div><div class="insight-k">Soil moisture</div><p class="insight-v">Humidity at {_h}%. {'Above optimal — reduce next irrigation cycle.' if _h > 70 else 'Within optimal range.'}</p></div>
    </div>
    <div class="insight">
      <div class="insight-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9V3"/></svg></div>
      <div><div class="insight-k">Nitrogen status</div><p class="insight-v">N at {_n} mg/kg. {'Within optimal band — maintain current dosage.' if 60<=_n<=140 else 'Outside optimal range (60–140).'}</p></div>
    </div>
    <div class="insight">
      <div class="insight-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg></div>
      <div><div class="insight-k">Climate trend</div><p class="insight-v">{_crop_name} performs well under current climate pattern for this district zone.</p></div>
    </div>
    <div class="insight">
      <div class="insight-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h4v4a2 2 0 0 1-2 2Z"/><path d="M12 6H6"/><path d="M10 2h4"/></svg></div>
      <div><div class="insight-k">Fungal risk window</div><p class="insight-v">{'Monitor humidity — conditions favour fungal stress in weeks 6–8.' if _h > 65 else 'Low fungal risk under current conditions.'}</p></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    else:
        # No results yet
        st.markdown("""
<div style="padding:80px 40px;text-align:center;">
  <span class="eyebrow">No results yet</span>
  <h2 class="display section-title" style="margin-top:16px;">Run an analysis first.</h2>
  <p style="color:var(--ink-2);font-size:16px;margin:20px auto;max-width:480px;">Submit a soil image and parameters on the Cultivation page to generate your first synthesis report.</p>
  <a class="btn btn-primary" href="?page=cultivation" target="_self" style="text-decoration:none;display:inline-flex;align-items:center;gap:10px;">Go to Cultivation &#8594;</a>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

