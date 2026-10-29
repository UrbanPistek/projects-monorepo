# Project format

Files use JSON with extension `.lasercam.json`.

- `schemaVersion`: `1`
- `appVersion`: application version string
- `document`: canonical millimetre geometry, source metadata, import warnings, source-to-canonical transform
- `layers`: process layers including order, visibility, export enablement, speed, power, passes, operation kind
- `job`: scale, rotation, offset, origin mode, chord-error tolerance
- `machine`: rapid feed assumption, max S, optional work area
- `export`: selected firmware profile
- `display`: grid and color-match tolerance

Geometry is stored in the file. Reopening a project does not need the original SVG/DXF.

Unknown future fields should be preserved by consumers that round-trip JSON objects; this first writer uses a closed struct and will need explicit migrations for schema 2+.
