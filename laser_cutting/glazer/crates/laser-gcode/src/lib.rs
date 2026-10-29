use laser_core::{MachineSettings, PointMm};
use laser_planner::MotionPlan;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PowerMode {
    ConstantM3,
    DynamicM4,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FirmwareCapabilities {
    pub profile_id: String,
    pub display_name: String,
    pub tested_versions: Vec<String>,
    pub documentation_url: String,
    pub units_mm: bool,
    pub g20_g21: bool,
    pub absolute_positioning: bool,
    pub coordinate_decimals: u32,
    pub rapid_is_g0: bool,
    pub rapid_guarantees_laser_off: bool,
    pub supports_m3: bool,
    pub supports_m4: bool,
    pub supports_m5: bool,
    pub supports_arcs: bool,
    pub min_feed_mm_min: f64,
    pub max_feed_mm_min: f64,
    pub default_max_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostprocessSettings {
    pub max_s: f64,
    pub default_power_mode: PowerMode,
}

#[derive(Debug, thiserror::Error)]
pub enum GcodeError {
    #[error("{0}")]
    Refused(String),
    #[error("validation failed: {0}")]
    Validation(String),
}

impl GcodeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Refused(_) => "export_refused",
            Self::Validation(_) => "gcode_validation",
        }
    }
}

pub trait FirmwareProfile {
    fn capabilities(&self) -> &FirmwareCapabilities;
    fn postprocess(
        &self,
        plan: &MotionPlan,
        machine: &MachineSettings,
        settings: &PostprocessSettings,
    ) -> Result<String, GcodeError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaserState {
    Off,
    Constant,
    Dynamic,
}

#[derive(Clone, Debug)]
pub struct ProgramSnapshot {
    pub laser: LaserState,
    pub power_s: f64,
    pub feed: Option<f64>,
    pub absolute: bool,
    pub mm: bool,
    pub position: Option<PointMm>,
    pub has_frame_laser_on: bool,
}

pub fn map_power_percent(percent: f64, max_s: f64) -> Result<u32, GcodeError> {
    if !percent.is_finite() || !(0.0..=100.0).contains(&percent) {
        return Err(GcodeError::Refused(format!(
            "power percent {percent} is out of range"
        )));
    }
    if !max_s.is_finite() || max_s <= 0.0 {
        return Err(GcodeError::Refused("machine max S value is invalid".into()));
    }
    Ok(((percent / 100.0) * max_s).round() as u32)
}

pub fn validate_program(source: &str, in_frame: impl Fn(usize) -> bool) -> Result<(), GcodeError> {
    let mut laser = LaserState::Off;
    let mut ended = false;
    for (i, raw) in source.lines().enumerate() {
        let line = raw.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if upper.contains("M3") {
            laser = LaserState::Constant;
            if in_frame(i) {
                return Err(GcodeError::Validation(
                    "frame operation enabled the laser (M3)".into(),
                ));
            }
        }
        if upper.contains("M4") {
            laser = LaserState::Dynamic;
            if in_frame(i) {
                return Err(GcodeError::Validation(
                    "frame operation enabled the laser (M4)".into(),
                ));
            }
        }
        if upper.contains("M5") {
            laser = LaserState::Off;
        }
        if upper.starts_with("G0") && laser != LaserState::Off {
            return Err(GcodeError::Validation(format!(
                "rapid motion on line {} while laser is on",
                i + 1
            )));
        }
        if upper.contains("M2") || upper.contains("M30") {
            ended = true;
            if laser != LaserState::Off {
                return Err(GcodeError::Validation("program ended with laser on".into()));
            }
        }
        for token in upper.split_whitespace() {
            if let Some(rest) = token.strip_prefix('X').or_else(|| token.strip_prefix('Y')) {
                if rest.parse::<f64>().ok().filter(|v| v.is_finite()).is_none() {
                    return Err(GcodeError::Validation(format!(
                        "non-finite coordinate on line {}",
                        i + 1
                    )));
                }
            }
        }
    }
    if !ended {
        return Err(GcodeError::Validation(
            "program does not contain M2/M30 end".into(),
        ));
    }
    Ok(())
}

pub fn fmt_coord(value: f64, decimals: u32) -> String {
    format!("{:.*}", decimals as usize, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_percent_through_max_s() {
        assert_eq!(map_power_percent(70.0, 1000.0).unwrap(), 700);
        assert_eq!(map_power_percent(100.0, 255.0).unwrap(), 255);
    }
}
