# AgroSynapse AI — Claude Code Handoff Prompt
# ============================================================
# PASTE THIS ENTIRE FILE INTO CLAUDE CODE AS YOUR FIRST MESSAGE
# ============================================================

## STEP 0 — DO THIS FIRST (before writing a single line of code)

1. Open `AgroSynapse Redesign.html` in your browser right now.
2. Click through ALL FOUR pages: Home, Cultivation, Diagnostic, Dashboard.
3. Scroll through each page completely.
4. Read screenshots/README.md for section-by-section descriptions.
5. Read styles.css, styles-home.css, styles-tool.css LINE BY LINE.
6. Read every .jsx file in components/ LINE BY LINE.

Only after completing all 6 steps above, start implementing.

---

## YOUR MISSION

Rebuild the AgroSynapse Streamlit app UI (`https://github.com/manojanumolu/AgroSynapse`) to **pixel-perfectly match** `AgroSynapse Redesign.html`. Every color, font size, spacing, border radius, animation, and interaction must match exactly.

---

## DESIGN TOKENS — USE THESE EXACT VALUES, NO SUBSTITUTIONS

```css
/* Colors */
--paper:    #faf8f3   /* page background — NOT white, NOT #fff */
--paper-2:  #f2ede2   /* card/subtle bg */
--paper-3:  #e8e2d4   /* inset bg */
--ink:      #14140f   /* primary text */
--ink-2:    #3a3a32   /* secondary text */
--ink-3:    #6b6b5e   /* labels/tertiary */
--ink-4:    #a8a598   /* muted */
--forest:   #0f2818   /* PRIMARY dark green — rail bg, CTA bg, module cards */
--forest-2: #1a3a2a
--sage:     #5a8a3a   /* accent green */
--sage-2:   #7ba854   /* lighter sage */
--earth:    #d4a373   /* warm amber accent */
--earth-2:  #b8884f
--clay:     #c44536   /* warning red */
--gold:     #e8c989   /* hero accent — "acre." text, brand "SYNAPSE" */

/* Typography — load from Google Fonts */
--serif: "Instrument Serif", "Times New Roman", serif
--sans:  "Inter Tight", -apple-system, system-ui, sans-serif
--mono:  "JetBrains Mono", ui-monospace, monospace

/* Google Fonts import */
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Motion */
--ease:     cubic-bezier(0.2, 0.8, 0.2, 1)
--ease-out: cubic-bezier(0.16, 1, 0.3, 1)
```

---

## CHROME — PERSISTENT ON ALL PAGES

### Left Rail (72px wide, `#0f2818` bg, fixed/sticky)
```
Logo: 36×36px rounded-10 box, sage bg (#5a8a3a), leaf SVG icon in forest color
4 nav icons (18px SVG, stroke-only, no fill):
  • Home icon
  • Seedling icon  
  • Scan/crosshair icon
  • Line-chart icon
Active state: 2px sage left border accent + sage icon + rgba(122,168,84,0.12) bg pill
Inactive: rgba(250,248,243,0.5) color, transparent bg
Spacer fills remaining height
Bottom: Settings gear icon + "MA" avatar (36px circle, earth #d4a373 bg, forest text, JetBrains Mono)
```

### Top Bar (sticky, 69px tall, blurred glass)
```
bg: rgba(250,248,243,0.85), backdrop-filter: blur(16px)
border-bottom: 1px solid rgba(20,20,15,0.06)

Left: green dot (6px, sage with 3px sage/18% ring) + "AGROSYNAPSE" mono uppercase 11px
      On non-home pages: "AGROSYNAPSE · PAGE NAME"

Center: pill nav container (paper-2 bg, 1px border, 4px padding, border-radius 999px)
  Pills: padding 7px 16px, 13px Inter Tight 500, border-radius 999px
  Active: forest bg (#0f2818) + paper text (#faf8f3), box-shadow

Right: 3 icon buttons (36px circle, 1px border, 15px icons) — Search, Bell, Tweaks
```

---

## PAGE 1 — HOME (see screenshots/README.md §01)

### Hero Section
```
Full viewport height minus topbar (calc(100vh - 69px))
Background: ANIMATED SVG LANDSCAPE — NOT a CSS gradient, NOT a stock photo
  SVG must contain:
  - Sky: linearGradient top #f5e8cf → mid #e8c989 → lower #c68f4c → bottom #5a3a1a
  - Sun radial glow at ~72% from left, 38% from top: #fff3d1 center → #fcd982 → transparent
  - Distance ridge: muted brown path Q-curves
  - Foreground field: darker gradient path
  - 22 wheat stalks: vertical paths with Q-curve sway + ellipse grain heads
    Each stalk animates with sin(time/period + phase) — phase offset per stalk
  - 8 large blurred foreground stalks (filter:blur(2px), strokeWidth 4, opacity 0.8)
  - Grain filter overlay (feTurbulence baseFrequency 0.9 + feColorMatrix)
  - Vignette radial overlay
  Outer animation: scale 1.05→1.08 + translate, 16s loop ease-in-out

Overlay gradient:
  linear-gradient(180deg, transparent 0%, near-transparent 40%, rgba(250,248,243,0.88) 95%, #faf8f3 100%)
  + linear-gradient(90deg, rgba(10,15,10,0.55) 0%, rgba(10,15,10,0.25) 60%, rgba(10,15,10,0.45) 100%)

Content (z-index: 2, max-width 920px, padding-top 80px):
  1. Brand label: "AGRO" + "SYNAPSE" — Instrument Serif uppercase, clamp(18px,2.2vw,28px)
     letter-spacing 0.18em. "AGRO" color rgba(250,248,243,0.55). "SYNAPSE" color #e8c989.
  2. H1: "Laboratory-grade / agronomy, delivered / to every acre."
     Font: Instrument Serif 400, clamp(52px,8vw,120px), line-height 0.93, letter-spacing -0.04em
     Color: #faf8f3. text-shadow: 0 4px 40px rgba(0,0,0,0.25)
     "acre." is <em> italic, color #e8c989
     ANIMATION: each line slides up from translateY(110%) clip — staggered 180ms, 300ms, 420ms
  3. Lede: max-width 560px, 17px Inter Tight, line-height 1.6, rgba(250,248,243,0.85)
  4. Two buttons:
     Primary: paper bg (#faf8f3) + forest text (#0f2818), pill shape. Hover: #e8c989 bg.
     Ghost: rgba(250,248,243,0.08) bg, rgba(250,248,243,0.3) border, blur(8px). Paper text.
  5. Three badges: "Predictive Cultivation" / "Phyto-Diagnostic" / "AI Dashboard"
     Pill: rgba(250,248,243,0.08) bg, rgba(250,248,243,0.18) border, blur(8px), 12px, 500 weight
"SCROLL ↓" label bottom-left: rgba(250,248,243,0.5), floatY animation 2.4s
```

### Module Cards
```
Two cards side by side (1fr 1fr grid, 24px gap)
Left card bg: #0f2818 (forest). Right card bg: #2a2014 (dark earth).
Each card: border-radius 24px, padding 40px, min-height 360px, cursor pointer
Cursor-tracking radial gradient glow on hover (::before pseudo, opacity 0→1)
Hover: translateY(-6px) + box-shadow 0 32px 80px rgba(15,40,24,0.25)

Header row: 52×52 icon box (rgba(250,248,243,0.08) bg, 14px radius, rgba border) + eyebrow label
Title: Instrument Serif, clamp(36px,4vw,52px), color paper
Description: 15px, rgba(250,248,243,0.7), flex:1
Bottom: "Open module →" + 36px circle arrow button
  Arrow circle: rgba(250,248,243,0.1). On hover: sage bg + translateX(4px)
```

### Soil Pipeline (dark section, #0f2818 bg)
```
Eyebrow: "PIPELINE · SOIL TO CROP" — rgba(250,248,243,0.55)
Title: "From soil, a crop." — Instrument Serif, clamp(36px,4.5vw,64px), color paper
Sub text: rgba(250,248,243,0.7)
4 step cards (rgba(250,248,243,0.07) bg, rgba(250,248,243,0.12) border, 18px radius):
  Icon box: rgba(122,168,84,0.15) bg, rgba(122,168,84,0.3) border, #a8d080 icon color
  Step num: rgba(250,248,243,0.35)
  Title: paper color. Desc: rgba(250,248,243,0.65)
  Hover: rgba(250,248,243,0.12) bg + translateY(-4px) + rgba(122,168,84,0.5) border
Arrow connectors: rgba(250,248,243,0.08) bg, rgba(250,248,243,0.15) border
Steps: Specimen vision / Chemical profile / Climate synthesis / Crop recommendation
```

### Leaf Pipeline (paper-2 bg, #f2ede2)
```
4 step cards (white #faf8f3 bg, 1px solid var(--line) border, box-shadow-sm):
  Icon box: rgba(212,163,115,0.1) bg, rgba(212,163,115,0.25) border, earth-2 icon
  Step num: ink-4 color
  Hover: translateY(-4px) + shadow-lg + earth border
Steps: Leaf specimen / Neural pathology / Dosage synthesis / Action report
```

### CTA Block + Footer
```
CTA: #0f2818 bg, "Put a lab behind every field." — same serif treatment
Footer: #0f2818 bg, 2-column layout
```

---

## PAGE 2 — PREDICTIVE CULTIVATION

### Header
```
Eyebrow: "MODULE · AGRICULTURAL CORE" — mono 11px ink-3
Title: "Predictive Cultivation" — Instrument Serif clamp(48px,6vw,88px)
Sub: 16px Inter Tight, max-width 640px, ink-2 color
Right: Farmer Unit Guide card (paper-2 bg, 18px radius, 280px min-width)
  Rows: Yield/NPK/Area/Temp/Rainfall with mono values right-aligned
  Footer: "Open glossary →" in sage
```

### Form Blocks (2-column grid)
**Soil Specimen block:**
```
SVG soil image (aspect-ratio 4/3, dark brown #2a1608 bg):
  - radialGradient: center #6b3a1a → mid #4a2510 → edge #2a1608
  - feTurbulence grain filter  
  - ~50 scattered particles (circle elements, opacity 0.2-0.8)
  - Green dashed analysis grid (rgba 7ba854, 0.5 opacity, strokeDasharray 2 4)
  - Two annotation circles: radius 20 and 16, stroke #7ba854 1.2px
  - Labels: "TEX · Sandy-loam" and "AGG · 0.82" in #c5e6a0, 9px monospace
Chip overlay: "✓ Valid specimen · ready for synthesis" — paper bg blur, sage text, bottom-left
File row below: dark thumb + filename "specimen_T047.jpg" + "2.4 MB · 3024×4032"
```

**Chemical Profile block:**
```
2×2 grid of numeric steppers:
  Each stepper: border 1px solid var(--line), paper bg, 10px radius
  Input: mono 14px, left-aligned
  ± buttons: 38×42px, border-left, hover paper-2 bg
  Range bar: 3px height, paper-3 track, sage fill (or earth-2 if outside optimal)
  Optimal zone: sage-2/0.3 overlay on range bar
  Hint text: "● Optimal · X–Y" in sage OR "○ Outside optimal" in earth-2 (mono 11px)
  
  ⓘ tooltip on each label:
    N: "Available nitrogen in soil. Low N causes yellowing; high N causes excessive leaf growth. Optimal: 60–140 mg/kg."
    P: "Supports root development and flowering. Deficiency shows as purple leaf undersides. Optimal: 20–60 mg/kg."
    K: "Regulates water uptake and disease resistance. Low K causes leaf-edge browning. Optimal: 30–120 mg/kg."
    pH: "Acidity/alkalinity scale. Most crops thrive at 6.0–7.2. Outside this range, nutrients become unavailable."
  Tooltip: 240px wide, ink bg, paper text, 10px radius, fade+slide on hover, arrow pointer
```

**Climate Synthesis block (full-width):**
```
4-column row: State select / District select / Village input / "✦ Fetch vectors" button (sage bg)
Below: 3 tiles in row (paper-2 bg, 12px radius):
  Each tile: icon box (34px, paper bg, 8px radius) + label + large serif value + sparkline SVG
  Temperature: #c44536 sparkline. Humidity: #5a8a3a sparkline. Rainfall: #d4a373 sparkline.
```

**Farm Context block:**
```
2×2 grid: yield stepper + fertilizer stepper (both with ⓘ tooltips), then 4 selects
  Yield tooltip: "Your previous crop's yield in kg/ha. 1 acre ≈ 0.4 ha."
  Fertilizer tooltip: "Total fertilizer applied last season. Typical range: 60–200 kg/ha."
```

### Analyze Bar
```
Full-width, #0f2818 bg, border-radius 20px
Left: "Ready to synthesize." serif 32px + sub text rgba(250,248,243,0.7)
Right: Large pill button (padding 16px 28px)
On click: spinner animation + progress bar (3px, sage-2, animates 0→100% over 2.4s) → navigate to Dashboard
```

---

## PAGE 3 — PHYTO-DIAGNOSTIC SUITE

### Layout (2-column: 1.5fr + 1fr)

**Left: Plant Specimen card**
```
SVG leaf image (4:3 aspect ratio, dark green #14240f bg):
  - radialGradient: center #5a8a3a → mid #2d4a2b → edge #14240f
  - Ellipse leaf shape (#4a7a2a fill)
  - feTurbulence leaf texture
  - Vein paths (stroke #2a4a1a)
  - 5 disease spots: dark brown circles (#3a2014 outer, #5a3a20 inner)
  - Two bounding boxes (rect, stroke #e8c989, strokeDasharray 3 3):
    "Lesion · 0.91" and "Lesion · 0.88" labels in #e8c989 9px mono
Chip: "✓ Valid leaf detected · ready for diagnosis"
File row: green thumb + "apple_leaf_sample.jpg" + "1.8 MB · 2048×1536"
```

**Right column (stacked cards):**
```
Detection Result card:
  Header: "DETECTION RESULT" eyebrow + "PATHOGEN DETECTED" pill (clay/red bg rgba(196,69,54,0.08))
  "Apple Scab" — Instrument Serif 44px
  "Venturia inaequalis · fungal" — mono 12px ink-3
  Confidence: "CONFIDENCE LEVEL" label + "96.8%" sage serif right + 3px range bar full sage

Treatment Plan card:
  3 numbered steps (01/02/03) each with label + description
  01: Mancozeb 75% WP @ 2.5g/L or Captan 50% WP @ 2g/L
  02: NPK 12:32:16 @ 8g/L + Ca(NO₃)₂ @ 3g/L
  03: Prune infected twigs, rake leaves, ensure airflow

Top-5 Predictions card:
  5 rows: crop name / horizontal bar / percentage
  "Apple — Apple scab" 96.8% sage fill; others ink-4 fill
```

**Bottom:** "Run neural diagnosis →" (sage btn) + "↓ Export report" (ghost btn)

---

## PAGE 4 — DASHBOARD / ANALYTICS

### Primary Crop Card (full-width, #0f2818 bg, 3-column: 280px image + body + ring)
```
Left image: SVG 280px × full height, terracotta bg (#d4533f→#8a2a1a radialGradient)
  6 groundnut pod ellipses (cx varies, cy 160+50i, rx22 ry16, fill #d4a373, stroke #8b6f47)
  Bottom text: "GROUNDNUT · ARACHIS HYPOGAEA" rgba(255,255,255,0.4) 10px mono

Center body (padding 32px 40px):
  Eyebrow: "PRIMARY RECOMMENDATION" rgba(250,248,243,0.6)
  "Groundnut" — Instrument Serif clamp(48px,5vw,72px), paper color
  Description paragraph rgba(250,248,243,0.75)
  Divider: rgba(250,248,243,0.12)
  3-column metadata: NPK "18:46:32" / Yield "3,140 kg/ha" / Confidence "93%"
  Values: Instrument Serif 22px, letter-spacing -0.015em

Right ring (padding 32px 40px, border-left rgba(250,248,243,0.08)):
  SVG ring: 140px, stroke-width 6
  Track: rgba(250,248,243,0.15). Fill: #e8c989 (gold)
  Value "93%" centered in Instrument Serif 32px paper color
  "SYNAPTIC SCORE" label below
  Animation: stroke-dashoffset from full→target on viewport enter, 2s ease-out
```

### Probability Breakdown (2-column: 2fr + 1fr grid)
```
Left card: "Synaptic confidence, per crop."
  Segmented control: Soil / Climate / Combined
  7 horizontal bars: Groundnut 93% sage, Maize 68% earth, Chilli 54% earth-2, rest ink-4
  Each bar: 10px height, paper-2 track, border-radius 999px
  CSS animation: grow from 0 width on load

Right card: Alternative crops (ranked list)
  Rank num / crop name serif 22px / meta mono / tag pill / score% serif sage
  Row hover: background paper-2 with negative margin trick
```

### 8-Month Timeline (full-width card)
```
Month labels: Jun Jul Aug Sep Oct Nov Dec Jan (mono 10px ink-3)
5 phases in 2-column grid (160px label + 1fr track):
  Sowing / Vegetative / Flowering / Pod development / Harvest
  Bars: absolute positioned, height 26px, border-radius 6px
  Colors: sage / sage-2 / earth / earth-2 / clay
  Animation: opacity 0 → 0.85 on load
```

### Advisory Insight Cards (2×2 grid)
```
Each: icon box (38px, paper-2 bg, 10px radius, sage icon) + title bold 14px + desc ink-2 13px
Border 1px line-2. Hover: border-color line + shadow-sm.
Topics: Soil moisture elevated / Nitrogen balanced / Climate trend / Fungal window alert
```

---

## ANIMATIONS — ALL REQUIRED

```javascript
// 1. Hero text mask reveal
// Each line wrapped in .reveal-mask > span
// On page load: span transitions from translateY(110%) → translateY(0)
// Staggered delays: 180ms, 300ms, 420ms
// transition: transform 0.9s cubic-bezier(0.16, 1, 0.3, 1)

// 2. Scroll reveal (ALL content blocks)
// IntersectionObserver threshold 0.15
// Initial: opacity:0, translateY(20px)
// On intersect: opacity:1, translateY(0)
// transition: opacity 0.9s ease-out, transform 0.9s ease-out
// Stagger via delay param (80-200ms increments per sibling)

// 3. Hero landscape float
// @keyframes: transform scale(1.05)→scale(1.08) + translate, 16s ease-in-out infinite

// 4. Module card cursor glow
// mousemove listener → CSS vars --mx --my → ::before radial-gradient follows cursor

// 5. Number counters (dashboard + metrics)
// IntersectionObserver → requestAnimationFrame loop
// Value counts from 0 to target, cubic ease-out ~1.6s

// 6. Progress ring (dashboard 93%)  
// SVG strokeDashoffset: full→target on viewport entry, 2s ease-out

// 7. Probability bars
// @keyframes probGrow: from {width: 0} to {width: declared}
// CSS animation on .prob-fill, 1.4s ease-out

// 8. Timeline bars
// @keyframes tbarIn: opacity 0 → 0.85, staggered animationDelay per row

// 9. Analyze button progress bar  
// @keyframes progBar: width 0→100% over 2.4s ease-out
// Shows while analyzing, then navigates to dashboard

// 10. Card hover lifts
// transition: transform 0.4s ease, box-shadow 0.4s ease
// hover: translateY(-4 to -6px)

// 11. Pill animation (floatY)
// @keyframes: translateY(0→-6px→0), 2.4s ease-in-out infinite

// 12. Tooltips
// opacity: 0, translateY(4px) → opacity: 1, translateY(0) on hover, 0.2s
```

---

## STREAMLIT IMPLEMENTATION

### Hide all default Streamlit chrome
```python
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.main .block-container { padding-top: 0 !important; }
</style>""", unsafe_allow_html=True)
```

### Inject all CSS
```python
import pathlib
css = (pathlib.Path('styles.css').read_text() + 
       pathlib.Path('styles-home.css').read_text() + 
       pathlib.Path('styles-tool.css').read_text())
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{css}</style>""", unsafe_allow_html=True)
```

### Page routing
```python
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Inject rail + topbar as fixed HTML (use st.components.v1.html for the chrome)
# Use st.session_state.page to control which page content to show
```

### For animated sections use st.components.v1.html()
```python
import streamlit.components.v1 as components
components.html("""
<div class="hero">
  <!-- Full hero HTML from HomePage.jsx -->
</div>
<script>/* animation JS */</script>
""", height=600)
```

### For form sections use st.columns() + custom CSS on native widgets
```python
col1, col2 = st.columns(2)
with col1:
    # Soil specimen upload
    uploaded = st.file_uploader("", type=['jpg','png'])
    # Wrap with custom HTML card
```

---

## ABSOLUTE RULES — VIOLATIONS WILL BREAK THE DESIGN

❌ NEVER use CSS gradients as the hero background — it MUST be the animated SVG landscape  
❌ NEVER use #fff or #ffffff — background is #faf8f3  
❌ NEVER use dark grey or black for the module cards — they are #0f2818 (forest) and #2a2014  
❌ NEVER use Inter, Roboto, or system fonts — must load Instrument Serif + Inter Tight + JetBrains Mono  
❌ NEVER add emojis anywhere in the UI  
❌ NEVER break existing ML inference code (crop prediction, plant disease detection)  
❌ NEVER use stock images or external image URLs — all visuals are inline SVGs  
❌ NEVER use border-radius > 24px on main cards, or < 10px on form inputs  
❌ NEVER omit the ⓘ tooltips on N, P, K, pH, Yield, Fertilizer fields  
❌ NEVER show Streamlit's default sidebar, header, or footer  
❌ NEVER skip animations — if Streamlit can't do it natively, use st.components.v1.html()  

---

## DELIVERABLES

1. `app.py` or multi-page Streamlit structure (pages/ folder)  
2. `static/styles.css` (combined all CSS)  
3. Brief change summary (what files changed, what didn't)  
4. DO NOT change: model inference functions, data loading, preprocessing, session state keys used by ML pipeline
