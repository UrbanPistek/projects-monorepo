//! Raspberry Pi Pico W IoT Beacon
//! Conserves power by sleeping most of the time and only advertising when the PWM signal is quiet.

#![no_std]
#![no_main]

use {defmt_rtt as _, panic_probe as _};

use ::cyw43::PowerManagementMode;
use embassy_executor::Spawner;
use embassy_rp::peripherals::{DMA_CH0, DMA_CH1, PIO0};
use embassy_rp::pio::InterruptHandler;
use embassy_rp::{bind_interrupts, dma};
use embassy_time::{Duration, Timer};
use embassy_futures::join::join;

#[path = "lib/cyw43.rs"]
mod cyw43;
#[path = "lib/led.rs"]
mod led;
#[path = "lib/pwm.rs"]
mod pwm;
#[path = "lib/advertise.rs"]
mod advertise;

/// How long each BLE beacon transmission window lasts.
const ADVERTISE_DURATION: Duration = Duration::from_secs(15);

/// LED on/off period during the advertise window (matches the original demo).
const LED_BLINK_PERIOD: Duration = Duration::from_millis(250);

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>, dma::InterruptHandler<DMA_CH1>;
});

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // ── One-time setup ────────────────────────────────────────────────────────
    let mut platform = cyw43::setup(spawner, p).await;
    let stack = advertise::setup(platform.cyw.bt_device);
    let mut peripheral = stack.peripheral();
    let mut runner = stack.runner();

    // ── Sleep / active duty cycle ─────────────────────────────────────────────
    let app = async {
        let mut wake_count: u8 = 0;

        loop {
            // Sleep phase: deepest CYW43 power save, LED off, wait for PWM activity.
            //
            // `wait_for_any_edge` is interrupt-driven — the Embassy executor puts
            // the RP2040 into WFE while waiting, so we are not spinning.
            //
            // Note: the CYW43439 stays powered (WL_ON high). Fully power-gating it
            // would require re-init on every wake and is omitted for simplicity.
            cyw43::set_power_mode(&mut platform.cyw.control, PowerManagementMode::SuperSave).await;
            led::set(&mut platform.cyw.control, false).await; // Turn OFF
            pwm::sleep_until_activity(&mut platform.pwm_input).await;
            
            // Active phase: sample peak durations every second, LED ON
            led::set(&mut platform.cyw.control, true).await; // Turn ON
            wake_count = wake_count.wrapping_add(1);
            cyw43::set_power_mode(&mut platform.cyw.control, PowerManagementMode::PowerSave).await;

            // // `_last_peak` holds the most recent measurement for future use (e.g.
            // // telemetry). Not logged here to keep the firmware simple.
            // let measurements: pwm::FlowMeasurements = pwm::monitor_signal_until_quiet(&mut platform.pwm_input).await;
            // led::set(&mut platform.cyw.control, false).await; // Turn OFF

            let measurements = pwm::FlowMeasurements {
                avg_flow_rate_litres_per_min: 0.423,
                total_volumne_litres: 5.3,
            };
            Timer::after(Duration::from_secs(3)).await;
            led::set(&mut platform.cyw.control, false).await; // Turn OFF

            // Advertise after the PWM signal is quiet.
            join(
                advertise::run_burst(&mut peripheral, wake_count, measurements, ADVERTISE_DURATION),
                led::run_for(&mut platform.cyw.control, ADVERTISE_DURATION, LED_BLINK_PERIOD),
            )
            .await;
        }
    };

    // The host runner processes HCI events from the CYW43. Both futures must poll.
    join(runner.run(), app).await;
}