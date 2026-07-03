//! Onboard LED control via the CYW43439 GPIO.
//!
//! The Pico W routes its status LED through the wireless chip, not a direct
//! RP2040 GPIO. We toggle it with `control.gpio_set(0, …)`.

use ::cyw43::Control;
use embassy_time::Duration;

/// LED on/off period — matches the BLE beacon firmware.
pub const BLINK_PERIOD: Duration = Duration::from_secs(1);

/// Sets the onboard LED on or off.
pub async fn set(control: &mut Control<'_>, on: bool) {
    control.gpio_set(0, on).await;
}
