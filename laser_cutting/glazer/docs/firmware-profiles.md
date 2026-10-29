# Firmware profiles

Compatibility is a growing set of explicit, testable profiles. Generating a file is not the same as machine compatibility.

| Profile ID | Display name | Tested versions | Power mapping | Supported | Omitted |
|---|---|---|---|---|---|
| `grbl-1.1` | GRBL 1.1 | Documented against GRBL 1.1 laser mode (`M3`/`M4`/`M5`, `S`) | Percent × configured max S (default 1000, user-set to match `$30`) | Absolute mm (`G21`/`G90`), linear moves, laser-off rapids, frame with laser held off | Arcs (`G2`/`G3`), controller setting writes (`$30`/`$32`), jogging, streaming |
| `fluidnc` | FluidNC | Not implemented in this milestone | — | — | Entire profile |
| `grblhal` | grblHAL | Not implemented | — | — | Entire profile |
| `smoothieware` | Smoothieware | Not implemented | — | — | Entire profile |

GRBL laser mode notes used for the first profile:

- `M3` is constant power; `M4` is dynamic power when `$32` laser mode is enabled on the controller.
- This application never writes `$30` or `$32`. Confirm those on the machine before cutting.
- Frame operations emit `M5` and travel moves only. They must not emit `M3` or `M4`.

Hardware/test environment for `grbl-1.1`: unit and golden tests only in this repository revision. Physical-machine verification is a release checklist item, not a claim of this source tree.
