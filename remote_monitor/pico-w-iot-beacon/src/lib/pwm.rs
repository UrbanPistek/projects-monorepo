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
use embassy_time::with_timeout;
use embassy_time::{Duration, Instant};

/// How long with no PWM edges before we consider the signal gone.
const QUIET_SECONDS: Duration = Duration::from_secs(5);
const FLOW_SENSOR_K_FACTOR: f32 = 27.5; // original = 5.5

/// Sample once per second while actively monitoring.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const HARD_MAX_DURATION_LIMIT: Duration = Duration::from_secs(60*60*2); // Max 2 hour hard time limit to prevent an infinite sample loop

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
    let mut signal_is_quiet = false;

    // Using a max of 16 sample points to limit size reserved on the stack
    let mut current_frequency = 100.0 as f32; // Hertz
    let mut flow_rates: Vec<f32, 16> = Vec::new();

    // When the calculated frequency doops below a certain point we can stop sampling
    while !signal_is_quiet && start.elapsed() < HARD_MAX_DURATION_LIMIT {
        
        // Determine frequency
        current_frequency = determine_pwm_frequency(input).await;
        let flow_rate = current_frequency / FLOW_SENSOR_K_FACTOR;
        let _ = flow_rates.push(flow_rate);

        // Check if input has stopped for a specific amount of time
        // After each frequency sample: if no edge within QUIET_SECONDS, signal is done.
        signal_is_quiet = with_timeout(QUIET_SECONDS, input.wait_for_any_edge())
            .await
            .is_err(); // timeout = quiet, Ok(_) = still active
    }

    // Determine final values
    let elapsed_seconds = ((start.elapsed().as_secs()) - (QUIET_SECONDS.as_secs())) as f32; // Subtract out quiet period threshold
    let avg_flow_rate = average(&flow_rates).unwrap(); // L / min
    let total_volumne = avg_flow_rate * (elapsed_seconds / 60.0); // L

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
    let mut pulse_count = 0;

    // determine the pulse frequency
    // within a specific sample interval
    while start.elapsed() < SAMPLE_INTERVAL {

        // Get remaining to keep true to sample interval
        let remaining = SAMPLE_INTERVAL - start.elapsed();

        // Align to the start of a high pulse so we measure a full peak width.
        // Count the number of pulses in the interval
        // Have a timeout so if signal stops we do not wait forever
        match with_timeout(remaining, input.wait_for_rising_edge()).await {
            Ok(()) => pulse_count += 1,
            Err(_) => break, // no rising edge before sample window ends
        }
    }

    // Determine elapsed time
    let elapsed = start.elapsed();

    // Divide by 0 prevention
    if elapsed.as_millis() < 1 {
        return 0.0;
    }

    // Calculate the frequency in hertz
    let frequency: f32 = (pulse_count as f32 / elapsed.as_millis() as f32) * 1000.0; // uses millis so it only rounds down to the millisecond
    return frequency;
}

fn average<const N: usize>(data: &Vec<f32, N>) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    Some(data.iter().sum::<f32>() / data.len() as f32)
}
