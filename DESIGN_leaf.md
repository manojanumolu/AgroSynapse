# Design System Strategy: The Digital Arboretum

## 1. Overview & Creative North Star
The Creative North Star for this design system is **"The Living Archive."** 

Agriculture is a discipline of patience, depth, and organic growth. To reflect this, we are moving away from the "SaaS-dashboard-in-a-box" aesthetic. Instead, we aim for a **High-End Editorial** feel that balances the technical precision of laboratory science with the tactile warmth of a botanical garden. 

We break the "template" look by utilizing **intentional asymmetry** and **tonal depth**. Rather than rigid, boxed-in grids, we use generous white space (Scale 12-24) and overlapping "specimen" cards to create a sense of discovery. Large, bold headlines provide an authoritative, editorial anchor, while light-weight body text creates an airy, sophisticated atmosphere.

---

## 2. Colors & Surface Logic
Our palette is rooted in the earth, using deep forest greens and soft creams to reduce eye strain and establish a premium, "naturalist" tone.

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to define sections. Boundaries must be defined solely through background color shifts. For example:
*   Place a `surface-container-low` (#f5f4f0) section directly against a `surface` (#faf9f5) background. 
*   Use the `surface-dim` (#dbdad6) token for subtle grouping within a larger layout.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers—like stacked sheets of fine, heavy-weight paper.
*   **Base Layer:** `surface` (#faf9f5) for the main viewport.
*   **Nesting Level 1:** Use `surface-container-low` (#f5f4f0) for large structural areas (like the main content well).
*   **Nesting Level 2:** Use `surface-container-highest` (#e3e2df) for high-priority interactive cards or data clusters.

### Signature Textures & Glass
To elevate the platform beyond a flat digital interface:
*   **The Glass Rule:** Use `surface-container-lowest` (#ffffff) with 70% opacity and a `backdrop-blur` of 20px for floating elements like modals or hover-states. This allows the organic greens of the background to bleed through softly.
*   **Editorial Gradients:** For primary CTAs or data visualization backgrounds, use a subtle linear gradient from `primary` (#004425) to `primary-container` (#1e5c3a) at a 135-degree angle. This adds "visual soul" and depth.

---

## 3. Typography: The Scientific Narrative
We use a high-contrast pairing of **Manrope** and **Work Sans** to balance technical clarity with modern elegance.

*   **Display & Headlines (Manrope):** Bold, geometric, and authoritative. These should be treated as "titles of a scientific journal." Use `headline-lg` for dashboard titles to ground the user immediately.
*   **Body & Labels (Work Sans):** Light and modern. The light-weight body text (Weight 300/400) creates a sense of high-end precision.
*   **Hierarchy Note:** Use `on-surface-variant` (#404942) for body text to maintain a softer, more organic reading experience, reserving `on-surface` (#1b1c1a) strictly for high-impact headlines and active labels.

---

## 4. Elevation & Depth: Tonal Layering
We eschew traditional "box shadows" in favor of **Tonal Layering**.

*   **The Layering Principle:** Depth is achieved by "stacking." A `surface-container-lowest` card placed on a `surface-container-low` section provides enough natural contrast to signify elevation without visual noise.
*   **Ambient Shadows:** If a "floating" effect is required (e.g., a detail tooltip or active card), use an extra-diffused shadow: `box-shadow: 0 12px 48px rgba(27, 28, 26, 0.06);`. The shadow color is a tint of our `on-surface` color, mimicking natural forest light.
*   **The "Ghost Border":** If a border is required for accessibility in complex forms, use the `outline-variant` (#c0c9bf) at **15% opacity**. Never use a 100% opaque border.

---

## 5. Components

### Sidebar Navigation
The sidebar uses `tertiary` (#214130) as the base. Navigation items should have a "pill" active state using `secondary-container` (#acf3ba) with `on-secondary-container` (#2f7144) text. This high contrast against the dark background ensures the user’s location is never in doubt.

### Rich Result Cards
*   **Structure:** No internal dividers. Use Spacing Scale `4` (1.4rem) to separate the title from the data visualization.
*   **Progress Bars:** Use a thick (8px) track with a `primary` fill and a `surface-container-highest` background. Round the ends completely (`rounded-full`).
*   **Charts:** Use a palette of `primary`, `secondary`, and `tertiary_fixed` to represent data. Ensure lines are 2px thick with smoothed (spline) curves to feel more organic.

### Form Inputs & Range Indicators
*   **Inputs:** Use `surface-container-low` backgrounds. On focus, shift to `surface-container-lowest` with a "Ghost Border" of `primary`.
*   **Range Indicators:** The track should be `surface-container-highest`. The thumb (handle) should be a large, 24px circle in `primary` to provide a tactile, high-end feel.

### Data Badges (Pills)
Use `rounded-md` (12px) for all badges.
*   **Status Badges:** Use `secondary-fixed` for positive states and `error-container` for alerts. Keep text uppercase and tracked out (+0.05em) using `label-sm` for an editorial look.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use asymmetrical margins. For example, give the right side of a card more breathing room than the left to create an editorial flow.
*   **Do** use `rounded-md` (0.75rem / 12px) consistently across all cards and buttons to maintain the "Soft Minimalist" aesthetic.
*   **Do** lean into `surface-container` shifts for content organization instead of lines.

### Don’t
*   **Don't** use pure black (#000000) for text. It breaks the organic feel. Use `on-surface`.
*   **Don't** use standard 1px borders. If a boundary feels missing, increase the background color contrast or the spacing.
*   **Don't** clutter the "Rich Result Cards." If a card has more than 5 data points, it must be broken into nested tonal surfaces.
*   **Don't** use sharp corners. Everything—from tooltips to checkboxes—must adhere to the roundedness scale to feel approachable and high-end.