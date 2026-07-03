//! Periodic BLE beacon for the Raspberry Pi Pico W.
//!
//! Duty cycle:
//!   1. Sleep for 60 seconds (radio in deep power save, no advertising).
//!   2. Wake for 15 seconds: broadcast a non-connectable BLE beacon and blink the LED.
//!   3. Repeat.
//!
//! The CYW43439 runner and TrouBLE host stack run continuously in the background.
//! Power savings during sleep come from disabling advertisements, using aggressive
//! CYW43 power management, and letting Embassy idle the RP2040 core while waiting
//! on `Timer::after`.

#![no_std]
#![no_main]

use {defmt_rtt as _, panic_probe as _};

use ::cyw43::PowerManagementMode;
use embassy_executor::Spawner;
use embassy_futures::join::join;
use embassy_rp::peripherals::{DMA_CH0, DMA_CH1, PIO0};
use embassy_rp::pio::InterruptHandler;
use embassy_rp::{bind_interrupts, dma};
use embassy_time::{Duration, Timer};

#[path = "ble/advertise.rs"]
mod advertise;
#[path = "ble/cyw43.rs"]
mod cyw43;
#[path = "ble/led.rs"]
mod led;

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>, dma::InterruptHandler<DMA_CH1>;
});

/// How long the device rests between advertise windows.
const SLEEP_INTERVAL: Duration = Duration::from_secs(60);

/// How long each BLE beacon transmission window lasts.
const ADVERTISE_DURATION: Duration = Duration::from_secs(15);

/// LED on/off period during the advertise window (matches the original demo).
const LED_BLINK_PERIOD: Duration = Duration::from_secs(1);

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // ── One-time peripheral setup ───────────────────────────────────────────
    let mut cyw = cyw43::setup(spawner, p).await;
    let stack = advertise::setup(cyw.bt_device);
    let mut peripheral = stack.peripheral();
    let mut runner = stack.runner();

    // ── Duty-cycle loop (runs concurrently with the BLE host runner) ──────────
    let app = async {
        let mut wake_count: u32 = 0;

        loop {
            // Sleep phase: no advertising, radio in the deepest supported save mode.
            // `Timer::after` puts the executor to sleep (WFE on the RP2040).
            //
            // Note: the CYW43439 remains powered on WL_ON. Fully power-gating the
            // chip between cycles would save more energy but requires a full re-init
            // each wake — intentionally omitted here for simplicity.
            cyw43::set_power_mode(&mut cyw.control, PowerManagementMode::SuperSave).await;
            Timer::after(SLEEP_INTERVAL).await;

            // Advertise phase: bump the sequence number, relax power save for TX,
            // then run LED blink and BLE beacon concurrently for 15 seconds.
            wake_count = wake_count.wrapping_add(1);
            cyw43::set_power_mode(&mut cyw.control, PowerManagementMode::PowerSave).await;

            join(
                advertise::run_burst(&mut peripheral, wake_count, ADVERTISE_DURATION),
                led::run_for(&mut cyw.control, ADVERTISE_DURATION, LED_BLINK_PERIOD),
            )
            .await;
        }
    };

    // The host runner processes HCI events from the CYW43. Both futures must poll.
    join(runner.run(), app).await;
}
