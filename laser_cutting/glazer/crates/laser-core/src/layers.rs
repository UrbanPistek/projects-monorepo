//! # Process layers
//!
//! A layer is a **process container**, not a colour: it groups paths that
//! should be machined together with the same settings (operation, speed,
//! power, passes, laser mode). A layer may hold paths of many colours, and a
//! path can be moved between layers at any time.
//!
//! Two special layers exist:
//! - the **frame** layer (`is_frame: true`, always first in the stack), which
//!   traces a dashed rectangle around the drawing with the laser off. Its
//!   operation is `TravelOnly` — a frame is just a travel-only layer with a
//!   margin; `laser_must_remain_off` is enforced by validation, because a
//!   frame must never fire the laser (a safety property, not a preference);
//! - the grey **Unassigned** layer (`AssignmentRule::Unassigned`), the sink
//!   for colour-less paths and for paths of deleted layers. It cannot be
//!   deleted and is not exported by default.

use serde::{Deserialize, Serialize};

use crate::color::Rgba8;
use crate::error::{check_finite, CoreError};
use crate::ids::LayerId;

/// What the laser does for paths on a layer. Frame layers use `TravelOnly`
/// and are marked separately via `ProcessLayer::is_frame`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OperationKind {
    Cut,
    EngraveLine,
    TravelOnly,
}

/// M3 holds power constant (good for cutting); M4 ramps power with speed
/// (good for engraving). GRBL-specific, hence kept here as a neutral enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LaserMode {
    Constant,
    Dynamic,
}

/// How paths were put on this layer. `Color` links an auto-generated layer to
/// the source colour it was created from (used to find-or-create layers in
/// "Assign by color"); it does not restrict which paths may live on the layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AssignmentRule {
    Manual,
    Color { hex: String, tolerance: f64 },
    Unassigned,
}

/// One process layer: a name, a display colour, stacking order, visibility /
/// export toggles, and the machining settings applied to its paths.
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
    /// Frame layers must never enable the laser.
    pub laser_must_remain_off: bool,
    pub frame_margin_mm: f64,
    /// Marks the frame layer (a travel-only layer that traces the drawing
    /// outline). `#[serde(default)]` keeps older project files loadable.
    #[serde(default)]
    pub is_frame: bool,
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
            laser_must_remain_off: false,
            frame_margin_mm: 2.0,
            is_frame: false,
        }
    }

    /// The frame layer: a travel-only layer (laser always off) that traces a
    /// dashed rectangle around the drawing, expanded by `margin_mm`.
    pub fn frame(order: u32, margin_mm: f64) -> Self {
        let mut layer = Self::new(
            "Frame",
            Rgba8::rgb(255, 160, 0),
            order,
            OperationKind::TravelOnly,
        );
        layer.laser_must_remain_off = true;
        layer.power_percent = 0.0;
        layer.frame_margin_mm = margin_mm;
        layer.source_assignment = AssignmentRule::Manual;
        layer.is_frame = true;
        layer
    }

    /// The dedicated grey holding layer for paths that are not assigned to any
    /// process layer. It is always present, is not export-enabled by default,
    /// and cannot be deleted, so deleting another layer always has somewhere
    /// safe to move its paths to.
    pub fn unassigned(order: u32) -> Self {
        let mut layer = Self::new(
            "Unassigned",
            Rgba8::rgb(160, 160, 160),
            order,
            OperationKind::EngraveLine,
        );
        layer.export_enabled = false;
        layer.source_assignment = AssignmentRule::Unassigned;
        layer
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        check_finite(self.speed_mm_min)?;
        check_finite(self.power_percent)?;
        check_finite(self.frame_margin_mm)?;
        if self.frame_margin_mm < 0.0 {
            return Err(CoreError::Invalid(format!(
                "layer '{}' frame margin must not be negative",
                self.name
            )));
        }
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
        if self.is_frame && self.operation != OperationKind::TravelOnly {
            return Err(CoreError::Invalid(
                "frame layers must use the travel-only operation".into(),
            ));
        }
        if self.laser_must_remain_off && !self.is_frame {
            return Err(CoreError::Invalid(
                "laser_must_remain_off is only valid for frame layers".into(),
            ));
        }
        if self.is_frame && !self.laser_must_remain_off {
            return Err(CoreError::Invalid(
                "frame layers must keep the laser off".into(),
            ));
        }
        Ok(())
    }
}
