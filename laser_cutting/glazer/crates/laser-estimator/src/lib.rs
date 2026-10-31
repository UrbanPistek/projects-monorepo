//! # Runtime estimator
//!
//! Estimates job time from a [`MotionPlan`] with a deliberately simple,
//! disclosed model: cut time = length / feed, travel time = length / rapid
//! feed. It does **not** model acceleration, cornering or buffering — those
//! assumptions are returned in `EstimateReport.assumptions` so the UI can show
//! them next to the number instead of hiding them. Per-layer breakdowns are
//! computed from the planner's per-layer stats so the breakdown always sums
//! to the same motion the planner emitted.

use laser_core::{MachineSettings, PointMm};
use laser_planner::{LayerPlanStats, MotionOp, MotionPlan};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EstimateReport {
    pub total_minutes: Option<f64>,
    pub cut_minutes: f64,
    pub travel_minutes: Option<f64>,
    pub dwell_minutes: f64,
    pub overhead_minutes: f64,
    pub cut_length_mm: f64,
    pub travel_length_mm: f64,
    pub path_count: u32,
    pub segment_count: u32,
    pub layers: Vec<LayerEstimate>,
    pub confidence: Confidence,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerEstimate {
    pub layer_id: laser_core::LayerId,
    pub name: String,
    pub cut_length_mm: f64,
    pub travel_length_mm: f64,
    pub cut_minutes: f64,
    /// `None` when the rapid feed is unknown, mirroring `travel_minutes` on
    /// the top-level report (travel time is then omitted from totals).
    pub travel_minutes: Option<f64>,
    pub total_minutes: Option<f64>,
    pub path_count: u32,
    pub passes: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Confidence {
    Low,
    Medium,
    AssumptionBased,
}

pub fn estimate(plan: &MotionPlan, machine: &MachineSettings) -> EstimateReport {
    let mut cut_min = 0.0;
    let mut travel_min = 0.0;
    let mut cut_len = 0.0;
    let mut travel_len = 0.0;
    let mut pos: Option<PointMm> = None;
    let mut warnings = Vec::new();
    let mut assumptions = vec![
        format!(
            "Linear model using feed rates only; chord error {} mm",
            plan.chord_error_mm
        ),
        "Does not model acceleration, cornering, or controller buffering".into(),
    ];

    let rapid = machine.rapid_feed_mm_min;
    if rapid.is_none() {
        warnings.push("Rapid feed is unknown; travel time is omitted from the total".into());
    } else {
        assumptions.push(format!(
            "Rapid/travel feed assumed {} mm/min",
            rapid.unwrap()
        ));
    }

    for op in &plan.ops {
        match op {
            MotionOp::RapidTo { to } => {
                if let Some(prev) = pos {
                    let d = prev.distance(*to);
                    travel_len += d;
                    if let Some(f) = rapid {
                        if f > 0.0 {
                            travel_min += d / f;
                        }
                    }
                }
                pos = Some(*to);
            }
            MotionOp::LaserOffMoveTo { to, feed_mm_min } => {
                if let Some(prev) = pos {
                    let d = prev.distance(*to);
                    travel_len += d;
                    if *feed_mm_min > 0.0 {
                        travel_min += d / feed_mm_min;
                    }
                }
                pos = Some(*to);
            }
            MotionOp::LaserMoveTo {
                to, feed_mm_min, ..
            } => {
                if let Some(prev) = pos {
                    let d = prev.distance(*to);
                    cut_len += d;
                    if *feed_mm_min > 0.0 {
                        cut_min += d / feed_mm_min;
                    }
                }
                pos = Some(*to);
            }
            MotionOp::SetLaser { .. } | MotionOp::LaserOff | MotionOp::Comment { .. } | MotionOp::ProgramEnd => {}
        }
    }

    let overhead_min = machine.motion_overhead_s / 60.0;
    let travel_minutes = if rapid.is_some() {
        Some(travel_min)
    } else {
        None
    };
    let total = travel_minutes.map(|t| cut_min + t + overhead_min);
    let confidence = if rapid.is_some() {
        Confidence::AssumptionBased
    } else {
        Confidence::Low
    };

    EstimateReport {
        total_minutes: total,
        cut_minutes: cut_min,
        travel_minutes,
        dwell_minutes: 0.0,
        overhead_minutes: overhead_min,
        cut_length_mm: cut_len,
        travel_length_mm: travel_len,
        path_count: plan.layers.iter().map(|l| l.path_count).sum(),
        segment_count: plan.layers.iter().map(|l| l.segment_count).sum(),
        layers: plan.layers.iter().map(|l| layer_estimate(l, rapid)).collect(),
        confidence,
        assumptions,
        warnings,
    }
}

/// Per-layer time derived from the planner's per-layer stats: cutting time is
/// passes x length / feed, travel time is length / rapid feed (when known).
/// Deriving from stats (instead of re-walking the op stream) keeps the
/// estimate consistent with what the planner actually emitted.
fn layer_estimate(layer: &LayerPlanStats, rapid: Option<f64>) -> LayerEstimate {
    let cut_minutes = if layer.speed_mm_min > 0.0 {
        layer.passes.max(1) as f64 * layer.cut_length_mm / layer.speed_mm_min
    } else {
        0.0
    };
    let travel_minutes = match rapid {
        Some(f) if f > 0.0 => Some(layer.travel_length_mm / f),
        _ => None,
    };
    let total_minutes = travel_minutes.map(|t| cut_minutes + t);
    LayerEstimate {
        layer_id: layer.layer_id,
        name: layer.name.clone(),
        cut_length_mm: layer.cut_length_mm,
        travel_length_mm: layer.travel_length_mm,
        cut_minutes,
        travel_minutes,
        total_minutes,
        path_count: layer.path_count,
        passes: layer.passes,
    }
}

#[cfg(test)]
mod tests {
    use laser_core::PointMm;
    use laser_planner::MotionPlan;

    use super::*;

    #[test]
    fn known_length_at_known_feed() {
        let plan = MotionPlan {
            ops: vec![
                MotionOp::RapidTo {
                    to: PointMm { x: 0.0, y: 0.0 },
                },
                MotionOp::LaserMoveTo {
                    to: PointMm { x: 100.0, y: 0.0 },
                    feed_mm_min: 1000.0,
                    power_percent: 50.0,
                },
                MotionOp::LaserOff,
                MotionOp::ProgramEnd,
            ],
            bounds: None,
            layers: vec![],
            chord_error_mm: 0.05,
        };
        let mut machine = MachineSettings::default();
        machine.rapid_feed_mm_min = Some(3000.0);
        let report = estimate(&plan, &machine);
        assert!((report.cut_minutes - 0.1).abs() < 1e-9);
        assert_eq!(report.cut_length_mm, 100.0);
    }
}
