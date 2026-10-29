use serde::{Deserialize, Serialize};

use crate::color::Rgba8;
use crate::error::{check_finite, CoreError};
use crate::ids::LayerId;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OperationKind {
    Cut,
    EngraveLine,
    Frame,
    TravelOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LaserMode {
    Constant,
    Dynamic,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AssignmentRule {
    Manual,
    Color { hex: String, tolerance: f64 },
    Unassigned,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessLayer {
    pub id: LayerId,
    pub name: String,
    pub display_color: Rgba8,
    pub order: u32,
    pub visible: bool,
    pub export_enabled: bool,
    pub operation: OperationKind,
    pub speed_mm_min: f64,
    pub power_percent: f64,
    pub passes: u32,
    pub laser_mode: LaserMode,
    pub source_assignment: AssignmentRule,
    /// Frame operations must never enable the laser.
    pub laser_must_remain_off: bool,
    pub frame_margin_mm: f64,
}

impl ProcessLayer {
    pub fn new(name: impl Into<String>, color: Rgba8, order: u32, operation: OperationKind) -> Self {
        Self {
            id: LayerId::new(),
            name: name.into(),
            display_color: color,
            order,
            visible: true,
            export_enabled: true,
            operation,
            speed_mm_min: 1000.0,
            power_percent: 50.0,
            passes: 1,
            laser_mode: LaserMode::Dynamic,
            source_assignment: AssignmentRule::Manual,
            laser_must_remain_off: operation == OperationKind::Frame,
            frame_margin_mm: 2.0,
        }
    }

    pub fn frame(order: u32, margin_mm: f64) -> Self {
        let mut layer = Self::new("Frame", Rgba8::rgb(255, 160, 0), order, OperationKind::Frame);
        layer.laser_must_remain_off = true;
        layer.power_percent = 0.0;
        layer.frame_margin_mm = margin_mm;
        layer.source_assignment = AssignmentRule::Manual;
        layer
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        check_finite(self.speed_mm_min)?;
        check_finite(self.power_percent)?;
        if self.speed_mm_min <= 0.0 {
            return Err(CoreError::Invalid(format!(
                "layer '{}' speed must be positive",
                self.name
            )));
        }
        if !(0.0..=100.0).contains(&self.power_percent) {
            return Err(CoreError::Invalid(format!(
                "layer '{}' power must be 0..=100 percent",
                self.name
            )));
        }
        if self.passes == 0 {
            return Err(CoreError::Invalid(format!(
                "layer '{}' must have at least one pass",
                self.name
            )));
        }
        if self.laser_must_remain_off && self.operation != OperationKind::Frame {
            return Err(CoreError::Invalid(
                "laser_must_remain_off is only valid for frame operations".into(),
            ));
        }
        if self.operation == OperationKind::Frame && !self.laser_must_remain_off {
            return Err(CoreError::Invalid(
                "frame operations must keep the laser off".into(),
            ));
        }
        Ok(())
    }
}
