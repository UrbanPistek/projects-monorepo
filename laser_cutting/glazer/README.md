# Glazer

Linux-first desktop laser CAM: import SVG/DXF, assign paths to process layers, preview the job, estimate runtime, and export GRBL 1.1 G-code. The CAM engine is a Rust workspace independent of Tauri.

Designed to simple, effective and portable.

## Development (Linux)

System packages (Debian/Ubuntu), from the Tauri 2 prerequisites:

```bash
sudo apt update
sudo apt install \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl wget file \
  libxdo-dev libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev pkg-config
```

Then:

```bash
npm install
cargo test --workspace
npm run tauri dev
```

## Layout

- `crates/laser-core` — millimetre geometry, layers, project model
- `crates/laser-import*` — SVG (`usvg`) and DXF adapters
- `crates/laser-planner` — firmware-neutral motion plan
- `crates/laser-profile-grbl` — GRBL 1.1 postprocessor
- `crates/laser-estimator` — linear time estimate with disclosed assumptions
- `crates/laser-app` — import/export/project services (no Tauri)
- `src-tauri` — desktop shell and commands
- `src` — Svelte UI

## Safety

Generated G-code can move a laser. Verify output with the laser disabled, confirm origin and work area, and set controller `$30`/`$32` yourself. This app does not write controller settings.
