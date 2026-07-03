//! PWM peak monitor for the Raspberry Pi Pico W.
//!
//! Duty cycle:
//!   1. **Sleep** — wait for any edge on the PWM input GPIO (interrupt-driven,
//!      RP2040 core idles). CYW43439 in deep power save (LED off).
//!   2. **Active** — once PWM activity is detected, sample the high-pulse width
//!      ("peak duration") every second and blink the onboard LED (1 s on/off).
//!      When no PWM edges arrive for 60 consecutive seconds, return to sleep.
//!
//! No Bluetooth or Wi-Fi networking — the CYW43439 is used only for the LED.

#![no_std]
#![no_main]

use {defmt_rtt as _, panic_probe as _};

use ::cyw43::PowerManagementMode;
use embassy_executor::Spawner;
use embassy_rp::peripherals::{DMA_CH0, DMA_CH1, PIO0};
use embassy_rp::pio::InterruptHandler;
use embassy_rp::{bind_interrupts, dma};

#[path = "gpio/cyw43.rs"]
mod cyw43;
#[path = "gpio/led.rs"]
mod led;
#[path = "gpio/pwm.rs"]
mod pwm;

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>, dma::InterruptHandler<DMA_CH1>;
});

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // ── One-time setup ────────────────────────────────────────────────────────
    let mut platform = cyw43::setup(spawner, p).await;

    // ── Sleep / active duty cycle ─────────────────────────────────────────────
    loop {
        // Sleep phase: deepest CYW43 power save, LED off, wait for PWM activity.
        //
        // `wait_for_any_edge` is interrupt-driven — the Embassy executor puts
        // the RP2040 into WFE while waiting, so we are not spinning.
        //
        // Note: the CYW43439 stays powered (WL_ON high). Fully power-gating it
        // would require re-init on every wake and is omitted for simplicity.
        cyw43::set_power_mode(&mut platform.cyw.control, PowerManagementMode::SuperSave).await;
        led::set(&mut platform.cyw.control, false).await;
        pwm::sleep_until_activity(&mut platform.pwm_input).await;

        // Active phase: sample peak durations every second, blink LED, until the
        // PWM signal is quiet for 60 seconds.
        cyw43::set_power_mode(&mut platform.cyw.control, PowerManagementMode::PowerSave).await;

        // `_last_peak` holds the most recent measurement for future use (e.g.
        // telemetry). Not logged here to keep the firmware simple.
        let _last_peak =
            pwm::monitor_until_quiet(&mut platform.pwm_input, &mut platform.cyw.control).await;
    }
}
