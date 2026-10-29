# Supported formats

Internal geometry is always millimetres. Source units are converted at import.

## SVG (MVP)

Supported when `usvg` can resolve them into paths:

- `path`, `line`, `polyline`, `polygon`, `rect`, `circle`, `ellipse`
- groups and nested transforms
- stroke/fill colors (solid)
- physical `width`/`height` in mm, cm, in, pt, pc, px
- `viewBox` combined with a physical viewport

Not imported (warning, not silent success):

- text (convert to outlines first)
- images/rasters
- filters, masks, clipping as visual effects
- gradient/pattern paints (geometry still imports; color match may fail)

If only a `viewBox` exists, import succeeds with **ambiguous units**. Export is blocked until the user confirms a unit.

Filled shapes are treated as boundary outlines. They are not hatched.

## DXF (MVP)

Imported:

- `LINE`
- `LWPOLYLINE` / `POLYLINE` (including bulge arcs)
- `CIRCLE`, `ARC`
- `ELLIPSE` (approximated with cubics)

Skipped with a report entry:

- `SPLINE`, `INSERT`/blocks, `HATCH`, dimensions, text, 3D, paper space

Units come from `$INSUNITS`. Unitless or unknown values are ambiguous until confirmed.

ByLayer colors use the layer table; ByBlock is treated as unmatched in the MVP.
