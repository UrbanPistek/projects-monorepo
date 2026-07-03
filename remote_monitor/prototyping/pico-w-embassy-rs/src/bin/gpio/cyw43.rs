//! CYW43439 setup for the onboard LED (Wi-Fi only, no Bluetooth).
//!
//! The Pico W status LED is wired through the CYW43439, so we still need the
//! wireless chip running even though this firmware never uses Wi-Fi or BLE.
//! This module handles the minimum bring-up: firmware load, background runner,
//! and the `Control` handle used to blink the LED.

use ::cyw43::aligned_bytes;
use ::cyw43::Cyw43439;
use cyw43_pio::{DEFAULT_CLOCK_DIVIDER, PioSpi};
use embassy_executor::Spawner;
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::peripherals::PIO0;
use embassy_rp::pio::Pio;
use embassy_rp::{dma, Peripherals};
use static_cell::StaticCell;

use crate::Irqs;

/// Long-lived driver state — the cyw43 runner task holds a `'static` reference.
static CYW43_STATE: StaticCell<::cyw43::State> = StaticCell::new();

/// CYW43 control handle for the onboard LED and power management.
pub struct Cyw43Handles {
    pub control: ::cyw43::Control<'static>,
}

/// Combined peripheral setup: CYW43439 + PWM input on GP22.
pub struct Setup {
    pub cyw: Cyw43Handles,
    pub pwm_input: Input<'static>,
}

/// Boots the CYW43439 and configures the PWM input pin.
///
/// # Firmware blobs
///
/// Place in `cyw43-firmware/` next to `Cargo.toml`:
/// `43439A0.bin`, `43439A0_clm.bin`, `nvram_rp2040.bin`
///
/// Download from: <https://github.com/embassy-rs/embassy/tree/main/cyw43-firmware>
pub async fn setup(spawner: Spawner, p: Peripherals) -> Setup {
    // Destructure upfront — Rust does not allow passing `p` by value after
    // moving individual fields out one at a time.
    let embassy_rp::Peripherals {
        PIN_22,
        PIN_23,
        PIN_24,
        PIN_25,
        PIN_29,
        PIO0,
        DMA_CH0,
        DMA_CH1,
        ..
    } = p;

    let pwm_input = Input::new(PIN_22, Pull::Down);
    // GP22 — free on the Pico W header (CYW43 uses GP23–25 and GP29).

    let fw = aligned_bytes!("../../../cyw43-firmware/43439A0.bin");
    let clm = aligned_bytes!("../../../cyw43-firmware/43439A0_clm.bin");
    let nvram = aligned_bytes!("../../../cyw43-firmware/nvram_rp2040.bin");

    let pwr = Output::new(PIN_23, Level::Low);
    let cs = Output::new(PIN_25, Level::High);

    let mut pio = Pio::new(PIO0, Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        DEFAULT_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        PIN_24,
        PIN_29,
        dma::Channel::new(DMA_CH0, Irqs),
        dma::Channel::new(DMA_CH1, Irqs),
    );

    let state = CYW43_STATE.init(::cyw43::State::new());

    let (_net_device, mut control, runner) = ::cyw43::new(state, pwr, spi, fw, nvram).await;

    spawner.spawn(task(runner).unwrap());
    control.init(clm).await;

    Setup {
        cyw: Cyw43Handles { control },
        pwm_input,
    }
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

/// Applies a CYW43 power-management profile.
pub async fn set_power_mode(control: &mut ::cyw43::Control<'_>, mode: ::cyw43::PowerManagementMode) {
    control.set_power_management(mode).await;
}
