# Design System Specification: Atmospheric Intelligence

## 1. Overview & Creative North Star: "The Ethereal Harvest"
This design system rejects the "dashboard-in-a-box" aesthetic in favor of **The Ethereal Harvest**. Our North Star is a high-end, editorial experience that feels as quiet and vast as an agricultural field at dawn. By blending high-precision AI data with organic, atmospheric glassmorphism, we create a UI that feels less like a tool and more like an expansive window into the future of farming.

The system breaks the rigid "flat-web" template through:
*   **Layered Transparency:** Using depth to represent the complexity of neural synapses.
*   **Atmospheric Immersion:** Allowing the high-quality dawn photography to bleed through the UI, grounding the technology in the physical earth.
*   **Intentional Asymmetry:** Breaking the grid with overlapping glass panes and varied typographic scales to mirror the natural irregularity of organic growth.

---

## 2. Colors & Surface Philosophy

### The Palette
We utilize a dark-mode foundation where "Black" is never pure, but rather a deep, subterranean green, and "White" is a soft, dew-kissed mist.

*   **Primary Accent:** `#acf3ba` (Soft Green Glow). This is our "Life Force." Use it sparingly for critical data points, active states, and primary CTAs.
*   **Surfaces:** Use `surface` (`#0d0f0d`) as the base. Use `surface-container` tiers to build depth.
*   **Accents:** `tertiary` (`#ebfdfc`) provides a crisp, sterile contrast to the organic greens, ideal for technical readouts.

### The "No-Line" Rule
Standard 1px solid borders are strictly prohibited for layout sectioning. Separation must be achieved through:
1.  **Background Shifts:** Placing a `surface-container-high` card over a `surface-dim` background.
2.  **Backdrop Blurs:** Using the glass effect to create a perceived boundary.

### Glass & Gradient Rule
To move beyond a "generic" look, all primary containers must utilize **Glassmorphism**:
*   **Fill:** `surface-container` at 40%–60% opacity.
*   **Blur:** 20px to 40px Backdrop Blur.
*   **Signature Glow:** Apply a subtle linear gradient to the `primary` accent (from `#b4fcc2` to `#1e6337`) for main CTAs to give them "soul" and weight.

---

## 3. Typography: Modern Editorial
We pair **Manrope** for its technical precision with **Plus Jakarta Sans** for metadata. This conveys an authoritative yet human-centric AI presence.

*   **Display (L/M/S):** Manrope. Use for "Hero" stats or agricultural insights. Keep tracking tight (-0.02em) to maintain an editorial feel.
*   **Headlines:** Manrope SemiBold. These are your anchors. Ensure high contrast against the blurred backgrounds.
*   **Labels:** Plus Jakarta Sans. Use for technical units (e.g., *kg/ha*, *Soil PH*). The slight geometric nature of Jakarta conveys the "Synapse" / AI aspect of the brand.
*   **Legibility:** All text defaults to `on-surface` (`#faf9f6`). Never use pure grey on glass; instead, use `on-surface-variant` (`#ababa8`) to maintain the misty, atmospheric tone.

---

## 4. Elevation & Depth: Tonal Layering

### The Layering Principle
Depth is achieved by stacking frosted panes.
*   **Level 0 (Earth):** The high-quality field background image.
*   **Level 1 (Mist):** `surface-container-lowest` at 30% opacity with heavy blur. Large layout areas.
*   **Level 2 (Frost):** `surface-container-low`. Individual cards or interactive panes.
*   **Level 3 (Light):** `surface-container-highest`. Tooltips or active modals.

### The "Ghost Border" Fallback
While solid lines are banned, "Ghost Borders" are permitted for definition.
*   **Spec:** 1px width, `outline-variant` (`#474846`) at **15% opacity**.
*   **Implementation:** This border should only exist to catch the "light" at the edge of a glass pane, mimicking the physical edge of a glass sheet.

### Ambient Shadows
Avoid black shadows. Use `on-primary-container` at 8% opacity with a 32px blur to create a soft, green-tinted ambient lift that feels like a glow rather than a shadow.

---

## 5. Components

### Glass Cards (The Signature Component)
*   **Style:** `surface-container-low` with 16px corner radius (`xl`).
*   **Border:** Ghost Border (15% opacity).
*   **Content:** No dividers. Use 24px vertical padding (from spacing scale) to separate headers from body text.

### Buttons
*   **Primary:** Fill with `primary` (`#b4fcc2`), text in `on-primary` (`#1e6337`). Apply a soft outer glow using the primary color at 20% opacity.
*   **Secondary:** Ghost Border style. No fill, `on-surface` text.
*   **Tertiary:** Transparent background, `primary` text, no border.

### Inputs & Fields
*   **State:** Background should be `surface-container-highest` at 20% opacity.
*   **Active:** The "Ghost Border" transitions to 100% opacity `primary` glow.

### Precision Chips
*   **Usage:** Used for crop types (Tomato, Potato) or sensor status.
*   **Visuals:** Pill-shaped (`full` roundedness), `surface-variant` background, subtle `label-md` typography.

---

## 6. Do’s and Don’ts

### Do:
*   **Do** embrace negative space. Let the dawn background "breathe" between glass panes.
*   **Do** use the `xl` (16px) radius consistently for all primary containers to maintain the "soft tech" feel.
*   **Do** use asymmetrical layouts (e.g., a large glass pane offset against a smaller, high-density data pane).

### Don’t:
*   **Don't** use 100% opaque backgrounds for cards. It breaks the "Ethereal" immersion.
*   **Don't** use traditional dividers or lines to separate list items. Use tonal shifts or 12px/16px gaps.
*   **Don't** use high-saturation reds for errors. Use `error_dim` (`#d7383b`) to ensure it fits within the muted, atmospheric dawn palette.