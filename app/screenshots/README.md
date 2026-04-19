# AgroSynapse Design Screenshots

## IMPORTANT: How to view the full reference design
The screenshots in this folder are viewport-level captures. For the FULL reference design:

1. Open `AgroSynapse Redesign.html` in Chrome/Firefox
2. Navigate each page using the top nav or left rail
3. Scroll through each page to see all sections

## What each page looks like (viewport captures):

### 01 — Home Page
- Full-viewport hero: animated SVG wheat field landscape (sunset sky, swaying stalks)
- Brand name "AGROSYNAPSE" in small uppercase + "SYNAPSE" in gold (#e8c989) above headline
- Large Instrument Serif headline: "Laboratory-grade / agronomy, delivered / to every acre." 
  — "acre." is italic gold
- Two CTA buttons + three capability badge pills
- Scrolling down: Two dark module cards (Predictive Cultivation on #0f2818 forest green, 
  Phyto-Diagnostic on #2a2014 dark brown)
- Soil pipeline section: dark forest bg (#0f2818), 4 glass-style step cards with sage icons
- Leaf pipeline section: paper-2 bg (#f2ede2), 4 white cards with earth-colored icons + shadow
- CTA block (forest bg) + footer

### 02 — Predictive Cultivation
- Large serif title "Predictive Cultivation", eyebrow "MODULE · AGRICULTURAL CORE"
- Farmer Unit Guide card (top right): paper-2 bg, mono labels
- 2-column grid of 4 form blocks:
  • Soil Specimen: dark soil SVG with green analysis circles + "TEX · Sandy-loam", "AGG · 0.82"
  • Chemical Profile: NPK + pH steppers with green range bars + optimal zone hint + ⓘ tooltips
  • Climate Synthesis: State/District/Village dropdowns + sparkline tiles (Temperature/Humidity/Rainfall)
  • Farm Context: yield + fertilizer steppers with tooltips + season/irrigation/crop/zone dropdowns
- Full-width forest-bg bar at bottom: "Analyze & predict crop →"

### 03 — Phyto-Diagnostic Suite  
- Large serif title "Phyto-Diagnostic Suite", eyebrow "MODULE · NEURAL VISION"
- Left: Plant Specimen card — dark green leaf SVG with yellow bounding boxes "Lesion · 0.91"
- Right column: Detection Result ("Apple Scab", Venturia inaequalis, 96.8% confidence bar)
  Treatment Plan (3 numbered steps), Top-5 predictions (horizontal probability bars)
- "Run neural diagnosis" + "Export report" buttons at bottom

### 04 — Dashboard (Analytics)
- "Synthesis complete." serif headline + Export PDF / Share report buttons
- Primary crop card (forest bg, full width): terracotta groundnut SVG image, "Groundnut" 
  large serif, NPK/Yield/Confidence metadata, gold ring progress at 93%
- Probability breakdown: horizontal bars per crop (sage = groundnut, earth = maize etc)
- Alternative crops panel (ranked list with scores)
- 8-month Gantt-style cultivation timeline
- 4 advisory insight cards in 2-col grid

## Design tokens (from styles.css):
--paper: #faf8f3 | --forest: #0f2818 | --sage: #5a8a3a | --earth: #d4a373
Font: Instrument Serif (display) + Inter Tight (UI) + JetBrains Mono (labels/numbers)
