# STYLESEED.md — Design Lock

Domain: Academic personal homepage (static, 2-page: home + CV)
Skin: Custom (StyleSeed-derived, Toss-influenced color revival — see Round 2 below)
Locked date: 2026-06-23 (Round 1) · updated 2026-06-23 (Round 2: Toss color pass)

## Key Color
`--brand: #3182F6` (Toss Blue, replaced Academic Blue `#1A73E8` in Round 2)
Tint: `--brand-tint: #EAF2FE` · Hover: `#1B6FE0` · Active: `#1559B8`
Usage: links (hover/active), primary buttons, the `NEW` tag, focus rings, selected/active nav state, `.new-publication` card border. This is the **interaction** accent — reserved for links/buttons/selection, not used for categorical labeling (see Venue taxonomy palette below).

## Venue taxonomy palette (Round 2 — intentional exception to "single accent")
Round 1 flattened all venue badges to one neutral grey pill, per StyleSeed's strict "single accent, no rainbow" rule. User feedback: this made papers indistinguishable at a glance. Round 2 reintroduces color, but as **soft pastel category chips** (Toss spending-category-tag style) rather than the original saturated rainbow fills — two co-existing, internally-consistent color systems:
- **Accent (`--brand`)** = interaction/selection (links, buttons, NEW, focus).
- **Venue palette** = categorical taxonomy (which conference), pastel bg + matching darker text, picked for ~AA contrast at badge text size:

| Venue | Background | Text |
|---|---|---|
| CVPR | `#E3EEFD` | `#1454B8` |
| ICCV / ICCVW | `#F0E9FB` | `#6B3FA0` |
| ECCV | `#DFF5F1` | `#0E7A68` |
| NeurIPS | `#FCEEDC` | `#A35A0A` |
| ICLR / ICLRW | `#FCE4ED` | `#B22D68` |
| ICML | `#E8EAFB` | `#3F4FA0` |
| AAAI | `#FBE6E6` | `#B23A3A` |
| WACV | `#E1F5E9` | `#1E7A45` |
| IPIU | `#EAEDEF` | `#51606B` |
| arXiv | `#F7E6E2` | `#9C3F26` |
| **Under Review** (no modifier) | `var(--surface-inactive)` | `var(--gray-700)` |

**"Under Review" deliberately stays neutral grey** — a confirmed venue is a *category* and earns a color identity; "Under Review" is a *pending status*, not yet a category, consistent with StyleSeed's "status color = severity, normal/pending stays grey" principle even as confirmed venues get color back. `.award-badge` (meta-annotations like "Work done at Kakao", "350+ Stars") also stays neutral — it's not a venue category either.

News-list icon badges (`.news-icon-badge--{blue,amber,violet}`) reuse this same closed palette (CVPR-blue / NeurIPS-amber / ICML-indigo) for generic milestone categories (career/achievement/education) rather than introducing new hues.

Card backgrounds themselves stay plain white (`--surface-card`) — color is concentrated in icons/badges/buttons only, never in tinting whole card surfaces, to keep the Round 1 card hierarchy intact.

## Grayscale
- `--gray-900 #2A2A2A` — primary text (titles, names)
- `--gray-700 #3C3C3C` — secondary emphasis (descriptions, badges)
- `--gray-500 #6A6A6A` — meta/tertiary text (dates, captions, default nav/link color)
- `--gray-400 #7A7A7A` — decorative/disabled only; borderline AA contrast, never used for body text
- `--gray-300 #9B9B9B` — faint

## Surfaces
Page `#FAFAFA` (unused directly — page background stays browser default white) · Card `#FFFFFF` · Row `#FAFAF9` (unused, reserved) · Inactive `#E8E6E1` (venue badge base/"Under Review" fill) · Tint `#EAF2FE`
Borders: `--border-subtle #ECECEC`, `--border-default #E0E0E0`

## Radius personality: "Soft Card"
Cards/images: 16px (`--radius-card`) · Small controls: 10px (`--radius-sm`) · Large controls: 14px (`--radius-lg`) · Pills/badges: 999px (`--radius-pill`)

## Shadow language
Card: `0 1px 2px rgba(0,0,0,.04), 0 6px 16px rgba(0,0,0,.08)` · Hover: `0 10px 24px rgba(0,0,0,.12)` · Elevated: `0 12px 28px rgba(0,0,0,.12)` · Modal: `0 16px 40px rgba(0,0,0,.14)`
Rule: depth shadows never exceed ~15% opacity; escalates card < hover < elevated < modal. (Icon/text legibility scrims — the zoom-hint badge, the lightbox backdrop — are a separate category and intentionally darker; they're overlays for contrast, not depth cues.)
Cards also get a crisp `1px solid var(--border-subtle)` border (Round 3) — paired with the page now actually using `--surface-page` (`#FAFAFA`) as the `body` background (a Round 1 oversight: the token existed but was never applied, so cards were invisible against a white page). `.publication-card` additionally lifts (`translateY(-2px)` + `--shadow-hover`) on hover, "Snap"-timed, disabled under `prefers-reduced-motion`.

## Motion seed: "Snap"
`--dur-fast: 100ms`, `--dur-normal: 150ms`, `--ease-snap: ease-out`. Minimal translateY/scale (lightbox pop-in uses `scale(0.98→1)`, not a larger jump).
`prefers-reduced-motion: reduce` collapses all CSS transitions/animations globally; the jQuery-driven News "Show All" toggle is bypassed via a `matchMedia` check in `js/index.js` (jQuery's `slideDown`/`slideUp` aren't covered by the CSS media query).

## Type scale
Existing font sizes were left as-is (Noto Sans body, Google Sans 700 headings, Source Sans 3 nav) — only color/radius/shadow/motion were tokenized, to avoid regressing a content-dense page's line lengths/wraps.

## Density
Desktop-first responsive, single breakpoint at 768px (no 430px mobile-frame assumption). Card/hero padding scales down at the breakpoint (32px→20px-ish via existing rules) rather than adopting StyleSeed's mobile spacing numbers verbatim.

## Section types in use
- **A (Full Card):** each publication entry (`.publication-card`), Experience entry (folded into its own `.card-section`)
- **B (Grid/list container):** News list, CV Education/Experience/Extracurricular/Publications blocks (`.card-section`)
- **C (Carousel):** not used — no carousel content on this site
- **D (Hero Card):** profile hero (`index.html`, `.card-hero`), CV header (`cv.html`, `.card-hero` with compact padding override)
- **Chrome (exempt from the card rule):** `nav.site-nav`, `footer.footer` — same exemption StyleSeed gives its own TopBar/BottomNav

## Explicitly out of scope
KPI grids, donut/bar charts, bottom-nav tab bar, input-field rules (no `<form>`/`<input>` anywhere), the 430px mobile-frame spacing assumption, loading skeletons (fully static site, no async data fetching — the one realistic failure mode, image/video load failure, gets an `onerror`/`error` fallback instead).

## Content invariant
Publication list content (titles/authors/links/venues) is byte-for-byte equivalent across `index.html` and `cv.html` through this redesign — verified by diffing stripped text content before/after. Only CSS classes/markup wrappers changed. `CV_LATEX/CV.tex` / `JiwonCV.pdf` are untouched LaTeX artifacts with their own design language, out of scope for this restyle.

## Known intentional deviations from StyleSeed's letter
- Venue badges are pastel category chips, not grayscale (Round 2) — see "Venue taxonomy palette" above for the rationale (taxonomy vs. interaction are two separate color systems).
- `.publication-block` was renamed to `.publication-card` — a deliberate exception to `CLAUDE.md`'s existing "reuse these classes" guidance, required by the full structural card adoption (see `CLAUDE.md` note).
