//! CYW43439 wireless chip setup for the Raspberry Pi Pico W.
//!
//! The Pico W routes Wi-Fi and Bluetooth through a CYW43439 module. Embassy talks
//! to that chip over a non-standard half-duplex SPI bus implemented with PIO.
//! This module owns that hardware bring-up: firmware loading, the background
//! runner task, and the `Control` handle used for the onboard LED and power modes.

use ::cyw43::aligned_bytes;
use ::cyw43::bluetooth::BtDriver;
use ::cyw43::Cyw43439;
use cyw43_pio::{DEFAULT_CLOCK_DIVIDER, PioSpi};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::peripherals::PIO0;
use embassy_rp::pio::Pio;
use embassy_rp::{dma, Peripherals};

use crate::Irqs;
use static_cell::StaticCell;

/// Long-lived driver state for the CYW43439.
///
/// Must live in `'static` storage because the spawned `cyw43_task` runs for the
/// entire firmware lifetime.
static CYW43_STATE: StaticCell<::cyw43::State> = StaticCell::new();

/// Handles returned by [`setup`].
///
/// `bt_device` is handed to the BLE stack. `control` drives the WL-LED and
/// selects how aggressively the radio sleeps between advertise windows.
pub struct Cyw43Handles {
    pub bt_device: BtDriver<'static>,
    pub control: ::cyw43::Control<'static>,
}

/// Boots the CYW43439 with Bluetooth enabled and spawns its event-loop task.
///
/// # Firmware blobs
///
/// Place these files in `cyw43-firmware/` next to `Cargo.toml` before building:
///
/// - `43439A0.bin` — main Wi-Fi / Bluetooth firmware
/// - `43439A0_clm.bin` — regulatory (CLM) blob, uploaded after boot
/// - `43439A0_btfw.bin` — Bluetooth patch firmware (required for BLE)
/// - `nvram_rp2040.bin` — board NVRAM calibration
///
/// Download from: <https://github.com/embassy-rs/embassy/tree/main/cyw43-firmware>
pub async fn setup(spawner: Spawner, p: Peripherals) -> Cyw43Handles {
    // Embed firmware at compile time. Paths are relative to this source file.
    let fw = aligned_bytes!("../../cyw43-firmware/43439A0.bin");
    let clm = aligned_bytes!("../../cyw43-firmware/43439A0_clm.bin");
    let btfw = aligned_bytes!("../../cyw43-firmware/43439A0_btfw.bin");
    let nvram = aligned_bytes!("../../cyw43-firmware/nvram_rp2040.bin");

    // WL_ON on GPIO 23 powers the CYW43439. CS on GPIO 25 frames SPI transactions.
    let pwr = Output::new(p.PIN_23, Level::Low);
    let cs = Output::new(p.PIN_25, Level::High);

    // PIO bit-bangs the CYW43-specific SPI protocol. Two DMA channels feed the
    // state machine so the CPU is not busy shifting every byte.
    let mut pio = Pio::new(p.PIO0, Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        DEFAULT_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        p.PIN_24,
        p.PIN_29,
        dma::Channel::new(p.DMA_CH0, Irqs),
        dma::Channel::new(p.DMA_CH1, Irqs),
    );

    let state = CYW43_STATE.init(::cyw43::State::new());

    // `new_with_bluetooth` loads Wi-Fi, NVRAM, and the BT patch, then exposes
    // an HCI controller (`bt_device`) for the TrouBLE host stack.
    let (_net_device, bt_device, mut control, runner) =
        ::cyw43::new_with_bluetooth(state, pwr, spi, fw, btfw, nvram).await;

    // The runner polls the chip for IRQs and SDIO events. If it stops, BLE stalls.
    spawner.spawn(task(runner).unwrap());

    // CLM must be uploaded before the chip is allowed to transmit on air.
    control.init(clm).await;

    Cyw43Handles { bt_device, control }
}

/// Background task that drives the CYW43439 event loop.
#[embassy_executor::task]
async fn task(
    runner: ::cyw43::Runner<
        'static,
        ::cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>,
        Cyw43439,
    >,
) -> ! {
    runner.run().await
}

/// Applies a CYW43 Wi-Fi power-management profile.
///
/// During the long sleep phase we use `SuperSave` so the radio dozes deeply.
/// During advertising we switch to `PowerSave` for reliable beacon timing.
pub async fn set_power_mode(control: &mut ::cyw43::Control<'_>, mode: ::cyw43::PowerManagementMode) {
    control.set_power_management(mode).await;
}
