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

use heapless::Vec;
use embassy_rp::gpio::Input;
use embassy_time::{Duration, Instant};

/// How long with no PWM edges before we consider the signal gone.
const QUIET_SECONDS: u32 = 5;
const FLOW_SENSOR_K_FACTOR: f32 = 5.5;

/// Sample once per second while actively monitoring.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

// Track flow measurements
pub struct FlowMeasurements {
    pub avg_flow_rate_litres_per_min: f32,
    pub total_volumne_litres: f32,
}

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
/// F = (5.5 × Q)
/// Q = Flow rate in L/min
/// 5.5 = Pulses per litre (the sensor's K-factor, in pulses/L/min → Hz conversion)
/// F = The resulting pulse frequency in Hz
pub async fn monitor_signal_until_quiet(input: &mut Input<'_>) -> FlowMeasurements {
    
    // Track how long the signal is active for
    let start = Instant::now();

    // Using a max of 16 sample points to limit size reserved on the stack
    let mut current_frequency = 100.0 as f32; // Hertz
    let mut flow_rates: Vec<f32, 16> = Vec::new();

    // When the calculated frequency doops below a certain point we can stop sampling
    // while current_frequency > 1.0 {
    while start.elapsed() < Duration::from_secs(5) {
        
        // Determine frequency
        current_frequency = determine_pwm_frequency(input).await;
        let flow_rate = current_frequency / FLOW_SENSOR_K_FACTOR;
        flow_rates.push(flow_rate);
    }

    // Determine final values
    let elapsed = start.elapsed();
    let avg_flow_rate = average(&flow_rates).unwrap(); // L / min
    let total_volumne = avg_flow_rate * (elapsed.as_secs() * 60) as f32; // L

    FlowMeasurements {
        avg_flow_rate_litres_per_min: avg_flow_rate,
        total_volumne_litres: total_volumne
    }
}

/// Samples once per second: waits for PWM activity, then measures one high pulse.
///
/// Returns `None` if no edge arrives within one second — a "quiet" second.
async fn determine_pwm_frequency(input: &mut Input<'_>) -> f32 {
    
    let start = Instant::now();
    let mut elapsed = start.elapsed();
    let mut pulse_count = 0;

    // determine the pulse frequency
    while elapsed < SAMPLE_INTERVAL {

        // Align to the start of a high pulse so we measure a full peak width.
        // Count the number of pulses in the interval
        if input.is_low() {
            input.wait_for_rising_edge().await;
            pulse_count += 1;
        }

        // Determine elapsed time
        elapsed = start.elapsed();
    }

    // Calculate the frequency in hertz
    let frequency: f32 = (pulse_count / elapsed.as_millis()) as f32 * 1000.0; // uses millis so it only rounds down to the millisecond
    return frequency;
}

fn average<const N: usize>(data: &Vec<f32, N>) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    Some(data.iter().sum::<f32>() / data.len() as f32)
}
