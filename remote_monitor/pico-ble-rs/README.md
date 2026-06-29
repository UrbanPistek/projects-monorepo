# pico-wh-ble — BLE "Hello, World!" on the Raspberry Pi Pico WH

Turns the Pico WH into a BLE GATT Peripheral that:
- Advertises as **PicoHello**
- Exposes a custom service with one read-only characteristic
- Returns `Hello, World!` to any BLE Central that reads it

## Stack

```
Your firmware (src/main.rs)
  └── TrouBLE  (BLE host — GATT, ATT, SM layers)
       └── bt-hci ExternalController
            └── cyw43  (CYW43439 driver — HCI over PIO SPI)
                 └── cyw43-pio  (PIO state machine SPI transport)
                      └── Embassy RP HAL  (embassy-rp)
                           └── Embassy executor  (async runtime)
```

---

## Prerequisites

### 1. Rust toolchain

```bash
rustup target add thumbv6m-none-eabi
```

### 2. Tools

```bash
cargo install --locked probe-rs-tools   # flashing + defmt log streaming
cargo install flip-link                 # stack-overflow detection linker
```

### 3. CYW43 firmware blobs  ← **required before building**

The CYW43439 chip needs three firmware files baked into flash at compile time.
Download them from the Embassy repo and place them beside `src/`:

```bash
mkdir cyw43-firmware
# Download from:
# https://github.com/embassy-rs/embassy/tree/main/cyw43-firmware
#
# Files needed:
#   cyw43-firmware/43439A0.bin       (~220 KB)  Wi-Fi + BT firmware
#   cyw43-firmware/43439A0_clm.bin   (~5 KB)    Regulatory blob
#   cyw43-firmware/43439A0_btfw.bin  (~6 KB)    BT firmware patch
```

Direct download (requires `curl`):
```bash
BASE="https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware"
curl -L "$BASE/43439A0.bin"      -o cyw43-firmware/43439A0.bin
curl -L "$BASE/43439A0_clm.bin"  -o cyw43-firmware/43439A0_clm.bin
curl -L "$BASE/43439A0_btfw.bin" -o cyw43-firmware/43439A0_btfw.bin
```

---

## Project layout

```
pico-wh-ble/
├── .cargo/
│   └── config.toml      # build target + probe-rs runner
├── cyw43-firmware/      # ← you create this (see above)
│   ├── 43439A0.bin
│   ├── 43439A0_clm.bin
│   └── 43439A0_btfw.bin
├── src/
│   └── main.rs          # firmware
├── build.rs             # exposes memory.x to the linker
├── Cargo.toml
├── memory.x             # RP2040 flash/RAM layout
└── README.md
```

---

## Build and flash

Connect your Pico WH via a debug probe (Raspberry Pi Debug Probe, or a second
Pico running picoprobe firmware), then:

```bash
cargo run --release
```

`probe-rs` will flash the ELF and immediately start streaming `defmt` log output:

```
INFO  CYW43 initialised, BLE controller ready
INFO  BLE host stack built
INFO  GATT table built — service UUID ends in f0, char UUID ends in f1
INFO  Starting BLE advertising as 'PicoHello'
INFO  Advertising... waiting for a BLE Central to connect
```

> **No debug probe?**  You can still flash via UF2:
> ```bash
> cargo build --release
> cargo install elf2uf2-rs
> elf2uf2-rs target/thumbv6m-none-eabi/release/pico-wh-ble
> # Hold BOOTSEL, plug in USB, release — then copy the .uf2 file to the drive.
> ```
> You won't get log output, but the BLE peripheral will still work.

---

## GATT layout

| Layer | UUID | Value |
|---|---|---|
| Service | `12345678-1234-5678-1234-56789abcdef0` | — |
| └ Characteristic | `12345678-1234-5678-1234-56789abcdef1` | `Hello, World!` (read-only) |

---

## Verifying reception

### On Android

1. Install **nRF Connect** (Nordic Semiconductor) — free on the Play Store.
2. Open the **Scanner** tab → tap **Scan**.
3. Find **PicoHello** in the list → tap **Connect**.
4. Expand **Unknown Service** (UUID ending in `f0`).
5. Tap the **↓ Read** button on the characteristic (UUID ending in `f1`).
6. You will see: `48 65 6C 6C 6F 2C 20 57 6F 72 6C 64 21` = `Hello, World!`

### On iOS

Use **LightBlue** (Punchthrough) — same steps as above.

### On Linux (command line)

```bash
# Find the device
bluetoothctl scan on
# Look for [NEW] Device XX:XX:XX:XX:XX:XX PicoHello
bluetoothctl scan off

# Connect and read
bluetoothctl connect XX:XX:XX:XX:XX:XX

# List services (use the service UUID ending in f0)
bluetoothctl list-attributes XX:XX:XX:XX:XX:XX

# Read the characteristic (use the char UUID ending in f1)
bluetoothctl select-attribute /org/bluez/hci0/dev_XX_XX_XX_XX_XX_XX/service.../char...
bluetoothctl read
# Output: 48 65 6c 6c 6f 2c 20 57 6f 72 6c 64 21
# ASCII:  H  e  l  l  o  ,     W  o  r  l  d  !
```

### On macOS

Use **LightBlue** from the App Store (same as iOS).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails: "can't find `43439A0.bin`" | Firmware blobs missing | Run the `curl` commands above |
| Device not visible in scanner | CYW43 init failed | Check defmt logs; ensure CLM blob is correct |
| Read returns wrong data | UUID mismatch | Confirm characteristic UUID ends in `f1` |
| Connection drops immediately | BLE resources exhausted | Increase `task-arena-size` in Cargo.toml |
