//! PWM input monitoring on an RP2040 GPIO.
//!
//! A typical PWM source toggles a line high and low at a fixed rate. We treat
//! each **high pulse width** as a "peak duration" — the time from rising edge
//! to falling edge.
//!
//! Sleep / wake:
//!   - **Sleep**: block on `wait_for_any_edge()` — interrupt-driven, no polling.
//!   - **Active**: sample one peak per second; 60 quiet seconds → back to sleep.
//!
//! The input pin (GP22) is configured in [`cyw43::setup`](crate::cyw43::setup).

use embassy_rp::gpio::Input;
use embassy_time::{Duration, Instant, with_timeout};

/// How long with no PWM edges before we consider the signal gone.
const QUIET_SECONDS: u32 = 5;

/// Sample once per second while actively monitoring.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Blocks until the PWM line toggles — used to wake from sleep.
///
/// Any edge means "activity started". The Embassy executor sleeps the RP2040
/// (WFE) while this future is pending.
pub async fn sleep_until_activity(input: &mut Input<'_>) {
    input.wait_for_any_edge().await;
}

/// Monitors PWM peaks once per second, blinking the LED concurrently, until the
/// line is quiet for [`QUIET_SECONDS`].
///
/// Returns the last measured peak width (if any was captured this session).
pub async fn monitor_until_quiet(input: &mut Input<'_>) -> Option<Duration> {
    let mut quiet_seconds: u32 = 0;
    let mut last_peak: Option<Duration> = None;

    while quiet_seconds < QUIET_SECONDS {
        // ── Sample one PWM peak within a 1-second window ─────────────────────
        match sample_peak(input).await {
            Some(peak) => {
                last_peak = Some(peak);
                quiet_seconds = 0;
            }
            None => {
                quiet_seconds += 1;
            }
        }
    }


    last_peak
}

/// Samples once per second: waits for PWM activity, then measures one high pulse.
///
/// Returns `None` if no edge arrives within one second — a "quiet" second.
async fn sample_peak(input: &mut Input<'_>) -> Option<Duration> {
    with_timeout(SAMPLE_INTERVAL, async {
        // Any edge means the signal is alive this second.
        input.wait_for_any_edge().await;

        // Align to the start of a high pulse so we measure a full peak width.
        if input.is_low() {
            input.wait_for_rising_edge().await;
        }

        let rise = Instant::now();
        input.wait_for_falling_edge().await;
        rise.elapsed()
    })
    .await
    .ok()
}
