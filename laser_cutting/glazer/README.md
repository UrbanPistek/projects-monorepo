# Glazer

Linux-first desktop laser CAM: import SVG/DXF, assign paths to process layers, preview the job, estimate runtime, and export GRBL 1.1 G-code. The CAM engine is a Rust workspace independent of Tauri.

Designed to simple, effective and portable.

## Development

Clone, then install the system dependencies for your platform (below), then:

```bash
npm install
cargo test --workspace   # Rust engine tests
npm run check            # Svelte/TypeScript checks
npm run tauri dev        # run the desktop app in dev mode
```

### Linux — Debian / Ubuntu

System packages (from the Tauri 2 prerequisites):

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

### Linux — Fedora / RHEL / CentOS Stream

```bash
sudo dnf install \
  webkit2gtk4.1-devel \
  libsoup3-devel \
  libayatana-appindicator-gtk3-devel \
  librsvg2-devel \
  libxdo-devel \
  openssl-devel \
  gcc-c++ \
  curl wget file \
  pkg-config
```

### Linux — Arch / Manjaro

```bash
sudo pacman -S \
  webkit2gtk-4.1 \
  libsoup3 \
  ayatana-appindicator \
  librsvg \
  libxdo \
  openssl \
  base-devel \
  curl wget file \
  pkgconf
```

### Linux — openSUSE

```bash
sudo zypper install \
  libwebkit2gtk-4_1-devel \
  libsoup-3_0-devel \
  libayatana-appindicator3-devel \
  librsvg-devel \
  libxdo-devel \
  libopenssl-devel \
  gcc-c++ \
  curl wget file \
  pkg-config
```

On every Linux distro you also need Rust (via <https://rustup.rs>) and Node.js 20+.

### macOS

```bash
xcode-select --install   # Command Line Tools (required by Tauri)
brew install node        # or install Node.js from nodejs.org
```

No WebKitGTK packages are needed on macOS — Tauri uses the system WKWebView.

### Windows

1. Install Rust from <https://rustup.rs> (the `x86_64-pc-windows-msvc` toolchain, which needs the **Microsoft C++ Build Tools** — install "Desktop development with C++" in Visual Studio or the standalone Build Tools).
2. Install Node.js 20+ from <https://nodejs.org>.
3. WebView2 is preinstalled on Windows 11 and installed automatically on Windows 10 by the Tauri installer/bundler when needed.
4. For `.msi` bundles, install WiX Toolset v4+ (`winget install WiXToolset.WiXToolset` or from wixtoolset.org) and add it to `PATH`.

## Building an application package

With the dependencies above installed:

```bash
npm install
npm run tauri build
```

This runs `vite build` for the frontend and compiles the Rust backend, then bundles installers into `src-tauri/target/release/bundle/`:

| Platform | Output | Notes |
| --- | --- | --- |
| Linux (Debian/Ubuntu) | `.deb`, `.AppImage` | `.deb` needs `dpkg`; `.AppImage` needs `appimagetool` (auto-downloaded by default). Select with `npm run tauri build -- --bundles deb,appimage` |
| Linux (other distros) | `.AppImage` works everywhere; `.rpm` if `rpmbuild` is installed | AppImage is the most portable Linux format |
| macOS | `.dmg`, `.app` | On Apple Silicon also produces an `aarch64` bundle; `universal` builds need both toolchains |
| Windows | `.msi`, `.exe` (NSIS) | `.msi` needs WiX in `PATH`; `.exe` needs NSIS (`winget install NSIS.NSIS`) |

`tauri.conf.json` currently sets `"bundle": { "targets": "all" }`, so every bundle type available for your platform is built; override per-invocation with `--bundles` as shown above.

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
