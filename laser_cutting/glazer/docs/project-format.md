# Project format

Files use JSON with extension `.lasercam.json`.

- `schemaVersion`: `1`
- `appVersion`: application version string
- `document`: canonical millimetre geometry, source metadata, import warnings, source-to-canonical transform
- `layers`: process layers including order, visibility, export enablement, speed, power, passes, operation kind. A layer is a **process container, not a colour**: it may hold paths of different source colours, and `sourceAssignment` (`manual` / `{ color: { hex, tolerance } }` / `unassigned`) only records how the layer was generated. The grey `unassigned` layer always exists and cannot be deleted; deleting another layer moves its paths there. Frame layers (`operation: "frame"`) are always first (`order: 0`), keep the laser off, and only allow `passes` and `speedMmMin` to change.
- `job`: scale, rotation, offset, origin mode, chord-error tolerance
- `machine`: rapid feed assumption, max S, optional work area
- `export`: selected firmware profile
- `display`: display unit, grid enable, snap, color-match tolerance, and `gridBackground` (`dark` | `offWhite`; absent in files written before this field existed and reads as `dark` via a serde default)

Geometry is stored in the file. Reopening a project does not need the original SVG/DXF.

Unknown future fields should be preserved by consumers that round-trip JSON objects; this first writer uses a closed struct and will need explicit migrations for schema 2+.
