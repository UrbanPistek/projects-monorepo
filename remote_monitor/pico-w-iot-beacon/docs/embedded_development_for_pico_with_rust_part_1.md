# Embedded Development for the Raspbaerry Pi Pico Using Rust

Writing embedded code using the Raspberry Pi Pico with the Rust ecosystem.

# Hardware

Before any code is written, first we need to be familar with the hardware. 
Knowing which board you are exactly using really matters - The Pico family are pretty similar but there are some key differences that may trip you up, plus you'll need to know which one to target when building the binary, these are the following options:

- [Pico](https://pip-assets.raspberrypi.com/categories/610-raspberry-pi-pico/documents/RP-008307-DS-2-pico-datasheet.pdf)
- [Pico W](https://pip-assets.raspberrypi.com/categories/686-raspberry-pi-pico-w/documents/RP-008312-DS-2-pico-w-datasheet.pdf)
- [Pico 2](https://pip-assets.raspberrypi.com/categories/1005-raspberry-pi-pico-2/documents/RP-008299-DS-3-pico-2-datasheet.pdf)
- [Pico 2 W](https://pip-assets.raspberrypi.com/categories/1088-raspberry-pi-pico-2-w/documents/RP-008304-DS-3-pico-2-w-datasheet.pdf)

Firstly, [this site](https://picow.pinout.xyz/) is a great source for checking the differences between the board layouts.
Here are some of the easiest was to tell which is which by just looking at the board:

| Board Differentiation | Notes | Image |
| ----------- | ----------- | ----------- | 
| Wireless (W) vs Non-Wireless Series | The Pico W series will contain the [CYW43439](https://www.infineon.com/part/CYW43439) wireless chip, which is visible on the board. | ![Pico W Layout](./images/pico_w_layout.png) |
| 1st vs 2nd Edition | The Pico 2 will have this additional IC along this what a 3 pin transistor [(Q1)](https://pcbsync.com/pcb-tombstone-marking-guide/) that is visible. | ![Pico 2 W Layout](./images/pico_w_2_layout.png) |

The main functionality difference between the wireless and non-wireless models is wifi / bluetooth support. From a programming standpoint the
wireless models require firmware for the CYW43439 chip to enable certain features.

**Note:** The H in either Pico H or Pico WH just tells you whether the board has the headers soldered on or not. 

[Documentation](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html)

# Rust Embedded Frameworks

For the firmware, save yourself the effort of building up everything from scratch (figuring out the pinout, etc) and use a existing framework - for most of the popular boards, a framework will give you all the basics to build custom firmware.

## Rust Embedded HAL with Pico Libraries

The first approach I found consisted of using the rust embedded-hal package along with the libraries for 
raspberry pi / pico devices:

1. [embedded-hal](https://github.com/rust-embedded/embedded-hal)l
2. [rp-pico](https://github.com/rp-rs/rp-hal-boards/tree/main/boards/rp-pico)
3. [rp-hal](https://github.com/rp-rs/rp-hal)

A sample project is here: [Rust Pico HAL Samples](../../prototyping/pico-w-rs).
Included is a basic blinky and a serial echo implementation. Outside of the library source files, I did not find too much support on additional functionality or guides. This approach is possible but also feels a bit more limited / has less features readily availible.

## Embassy-rs

The more supported and fully featured packge I found was Embassy-rs.

1. [Embassy-rs](https://github.com/embassy-rs/embassy)
2. [Embassy-rs Examples](https://github.com/embassy-rs/embassy/tree/main/examples/rp)

Here I found its better supported, more examples, more fully featured and more documentation and lots of support for other popular boards. This is the framework I'd recommend using, and the one I will expand on further.

# Environment & Setup

When working on the pico boards, there are a few specifics you need to ensure are setup correctly. As an example, something that might not be initially obvious is that the Pico W board does not haave a built in LED - so if you try running a [binkly sample](https://github.com/embassy-rs/embassy/blob/main/examples/rp/src/bin/blinky.rs) that is built for the Pico board, it will use GP25 for the built in LED, whereas on the Pico W board this pin is actually reserved for the SPI connection it needs to make with the CYW43439 chip - which has a built in LED (WL_GPIO0). So for [blinky on the Pico W](https://github.com/embassy-rs/embassy/blob/main/examples/rp/src/bin/wifi_blinky.rs) board you'll need to setup a SPI connection to the CYW43439 chip and then command it to turn on that LED. 

For a quick review of the differences between the boards, the documentation gives a simple list:
1. [Pico Boards](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html#layout_non-wireless)
2. [Pico Wireless Boards](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html#layout_wireless)

First, you'll need to ensure you have a [memory.x](../memory.x) file to define the RAM layout. Then accompanying that you need a [build.rs](../build.rs) script that copies the memory.x into a directory where the linker can always find it. You can reference the examples linked.

Next you'll want to specify the target and runner in a `config.toml`, you can see and example [here](../.cargo/config.toml): 

```
[build]
# Set the default target to match the Cortex-M0+ in the RP2040
target = "thumbv6m-none-eabi"

[env]
DEFMT_LOG = "debug"

# This runner will make a UF2 file and then copy it to a mounted RP2040 in USB
# Bootloader mode:
[target.'cfg(all(target_arch = "arm", target_os = "none"))']
runner = "elf2uf2-rs -d"
```

Next, particularily for the Pico W you need to download the CYW43439 firmware then load it at the start of the program for the CYW43439 chip. Depending on which functionalities you desire, you will need to load additional binaries. For example for BLE capabilities you'll also need the `43439A0_btfw.bin` binary. 

You can download the firmware as follows, there is also a [bash script](../install_firmware.sh) in my sample project for it.

```
mkdir cyw43-firmware
curl -L "https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware/43439A0.bin"      -o cyw43-firmware/43439A0.bin
curl -L "https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware/43439A0_clm.bin"  -o cyw43-firmware/43439A0_clm.bin
curl -L "https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware/43439A0_btfw.bin" -o cyw43-firmware/43439A0_btfw.bin
curl -L "https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware/nvram_rp2040.bin" -o cyw43-firmware/nvram_rp2040.bin
```

Then you'll need to load the binaries as shown [here](../src/lib/cyw43.rs):

```rs
// Embed firmware at compile time. Paths are relative to this source file.
let fw = aligned_bytes!("../../cyw43-firmware/43439A0.bin");
let clm = aligned_bytes!("../../cyw43-firmware/43439A0_clm.bin");
let btfw = aligned_bytes!("../../cyw43-firmware/43439A0_btfw.bin");
let nvram = aligned_bytes!("../../cyw43-firmware/nvram_rp2040.bin");
```

### Build & Flash

1. Hold BOOTSEL on the pico board - while holding it down plug the USB connector into the computer
2. Wait until the device shows up on the computer (similar to how a USB device shows up), BOOTSEL can be released, then run `cargo run --release` to build & flash
3. Baord should now be flashed - it will no longer show on the computer

# Sample Projects

Here are some linked sample projects to demonstrate some basic functionalities & features.

[Pico W Embassy Sample Projects](../../prototyping/pico-w-embassy-rs/) There are 3 different samples in here you can target & build for:

1. Blinky (Using the CYW43439 built in LED): `cargo build --bin blinky --release` or `cargo build --release`
2. Sleep with GPIO Awake Interrupt: `cargo build --bin gpio --release`
3. Periodic BLE Beacon: `cargo build --bin ble --release`

# Sources

1. [rp-hal-boards](https://github.com/rp-rs/rp-hal-boards/tree/main/boards/rp-pico)
2. [pico pinout](https://picow.pinout.xyz/)
3. [pico programming guide](https://raspberrytips.com/getting-started-with-raspberry-pi-pico/)
4. [rp2040 rust template](https://github.com/rp-rs/rp2040-project-template)
5. [rp serial demo](https://pico.implrust.com/usb-serial/code-with-rp-hal.html)
6. [Embassy-rs](https://github.com/embassy-rs/embassy/tree/main/examples/rp)
7. [struct chars](https://docs.python.org/3/library/struct.html#format-characters)
8. [raspberry pico docs](https://pip-assets.raspberrypi.com/categories/610-raspberry-pi-pico/documents/RP-008307-DS-2-pico-datasheet.pdf)
9. [beacon ble example](https://github.com/embassy-rs/trouble/blob/main/examples/apps/src/ble_beacon.rs)
10. [embassy examples](https://github.com/embassy-rs/embassy/blob/main/examples/rp/src/bin/gpio_async.rs)
11. [rust cyw43 driver](https://docs.rs/cyw43/latest/cyw43/)
