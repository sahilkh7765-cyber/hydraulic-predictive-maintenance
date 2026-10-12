# Design direction: test-bench instrument panel

The app is the control panel of a hydraulic test rig. Its readers are
maintenance and ML engineers who want one answer per screen ("is the valve
failing?", "how good is the accumulator model, honestly?") and then the
evidence. Dark theme only.

## The one bold element

**The rig schematic.** A simplified ISO 1219-style circuit drawing of the
rig (motor, pump, valve, accumulator, cooler, filter, tank, two circuits) is
the identity of the app. On Overview it explains the system; on Predict the
same drawing becomes the status display, each component carrying a status
lamp. Everything else stays quiet so the schematic can carry the personality.

## Color (dark, validated)

| Token | Hex | Role |
|---|---|---|
| ground | `#12171C` | page plane: blue-grey panel steel, not neutral black |
| plate | `#1A2027` | panels and chart surface |
| raised | `#222932` | inputs, hovered rows, table headers |
| hairline | `#2C3540` | borders, gridlines (solid, never dashed) |
| ink / ink-2 / muted | `#E6EBF0` / `#AEB9C4` / `#8D99A6` | text: 13.7 / 8.2 / 5.7 : 1 on plate |
| signal | `#62B0EA` | interaction only: focus, selection, links (7.0 : 1) |

Status is reserved and always paired with an icon and a word:

| State | Hex | Icon | Word |
|---|---|---|---|
| Normal | `#3DBE8B` | circle-check | Normal |
| Degraded | `#E9A23B` | triangle-alert | Degraded |
| Critical | `#F0605A` | octagon-x | Critical |

Chart series (identity), fixed order, validated on the plate surface with the
dataviz validator (all checks pass, worst adjacent CVD ΔE 8.4):
`#3987e5`, `#d95926`, `#199e70`, `#c98500`. Model families always take the
same slot (LR 1, RF 2, XGB 3, SVM 4). Magnitude uses one blue ramp.

## Type

- **IBM Plex Sans** for everything: engineering heritage, open apertures,
  good tabular figures. Weights 400 / 500 / 600.
- **IBM Plex Sans Condensed** 500 for panel nameplates and axis titles. It is the
  engraved-label voice, in sentence case and never tracked-out caps.
- **IBM Plex Mono** only for things that are identifiers: feature names
  (`PS2_cv`), sensor codes, file names.

Scale (base 15 px, ratio 1.25): 12 · 13 · 15 · 19 · 23 · 29 · 36.
Large readouts use proportional figures; table columns and axis ticks use
`tabular-nums`.

## Layout

```
desktop ≥1024                         phone 390
┌──────┬──────────────────────────┐   ┌──────────────────┐
│ rail │ Page title               │   │ ≡  Rig monitor   │
│ nav  │ takeaway sentence        │   ├──────────────────┤
│      │ ┌ readout strip ───────┐ │   │ title            │
│ api  │ └──────────────────────┘ │   │ takeaway         │
│ lamp │ evidence panels (varied  │   │ readouts stack   │
│      │ widths, not a card grid) │   │ panels stack     │
└──────┴──────────────────────────┘   └──────────────────┘
```

- Content left-aligned, max line length about 70 characters for prose.
- Each page leads with its takeaway as a sentence, followed by the readout
  strip that proves it. Panels carry a condensed nameplate, sentence case.
- 4 px spacing grid. Radius 6 px for panels and 4 px for controls; lamps are round.
  There are no shadows: depth comes from plate against ground plus one hairline.
- Component focus (cooler, valve, pump, accumulator) is one segmented control
  in the page's filter row, mirrored in the URL (`?c=valve`).

## Charts

Hairline solid grid, muted 12 px tabular ticks, units in axis titles, 2 px
lines, bars with 4 px rounded data ends, a 2 px gap between adjacent fills.
Direct labels are selective, and every chart has a tooltip and a table view.
Axes start at zero unless the break is stated in the axis title.

## Motion

There is one orchestrated moment: after a prediction, the status lamps light in
sequence. Everything else changes instantly. Under `prefers-reduced-motion`,
all animation is disabled.

## Plan review

Checked against generic defaults and revised:
- *Mono for every number*: dropped. Mono is now limited to real identifiers.
- *Neutral near-black plus one neon accent*: the ground is tinted steel, and the
  accent is a calm signal blue reserved for interaction, so it never competes
  with status colors.
- *Card grid of KPI tiles*: replaced by one readout strip per page and
  panels of different widths sized to their content.
- *Uppercase eyebrow labels*: none. Nameplates are sentence-case condensed type.
