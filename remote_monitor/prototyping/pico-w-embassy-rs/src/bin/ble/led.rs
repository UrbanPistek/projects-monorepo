//! Onboard LED control via the CYW43439 GPIO.
//!
//! The Pico W routes its status LED through the wireless chip, not a direct RP2040
//! GPIO. We toggle it with `control.gpio_set(0, …)` on the cyw43 `Control` handle.

use cyw43::Control;
use embassy_time::{Duration, Instant, Timer};

/// Blinks the onboard LED for `duration`, toggling every `blink_period`.
///
/// Uses the same 1-second on / 1-second off pattern as the original blinky demo.
pub async fn run_for(control: &mut Control<'_>, duration: Duration, blink_period: Duration) {
    let deadline = Instant::now() + duration;

    while Instant::now() < deadline {
        control.gpio_set(0, true).await;
        Timer::after(blink_period).await;

        if Instant::now() >= deadline {
            break;
        }

        control.gpio_set(0, false).await;
        Timer::after(blink_period).await;
    }

    // Leave the LED off when the advertise window ends.
    control.gpio_set(0, false).await;
}
