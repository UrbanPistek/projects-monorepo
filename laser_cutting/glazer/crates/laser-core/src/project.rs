//! # Project model
//!
//! A [`Project`] is the whole saved state: the imported [`VectorDocument`],
//! the process layers, and the job/machine/export/display settings. It is
//! serialized directly as JSON (`.lasercam.json`), so every field here is part
//! of the on-disk format — new fields must use `#[serde(default)]` to keep
//! older files loadable (see `docs/project-format.md`).
//!
//! Invariants enforced by validation: layer speeds positive, power within
//! 0..=100, at least one pass, and frame layers always keep the laser off.

use serde::{Deserialize, Serialize};

use crate::document::VectorDocument;
use crate::error::CoreError;
use crate::geom::{Affine, Bounds2D, PointMm};
use crate::ids::LayerId;
use crate::layers::ProcessLayer;
use crate::units::UnitCertainty;

pub const PROJECT_SCHEMA_VERSION: u32 = 1;
pub const APP_VERSION: &str = "0.1.0";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OriginMode {
    Source,
    LowerLeft,
    Center,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSettings {
    pub scale: f64,
    pub rotation_deg: f64,
    pub offset_mm: PointMm,
    pub origin_mode: OriginMode,
    pub chord_error_mm: f64,
}

impl Default for JobSettings {
    fn default() -> Self {
        Self {
            scale: 1.0,
            rotation_deg: 0.0,
            offset_mm: PointMm::origin(),
            origin_mode: OriginMode::LowerLeft,
            chord_error_mm: 0.05,
        }
    }
}

impl JobSettings {
    pub fn affine(&self) -> Affine {
        Affine::translate(self.offset_mm.x, self.offset_mm.y)
            .then(Affine::rotate_deg(self.rotation_deg))
            .then(Affine::scale(self.scale, self.scale))
    }

    pub fn with_origin(mut self, bounds: Bounds2D, mode: OriginMode) -> Self {
        let xf = Affine::rotate_deg(self.rotation_deg).then(Affine::scale(self.scale, self.scale));
        let transformed = bounds.transformed(xf).unwrap_or(bounds);
        self.origin_mode = mode;
        self.offset_mm = match mode {
            OriginMode::Source => PointMm::origin(),
            OriginMode::LowerLeft => PointMm {
                x: -transformed.min.x,
                y: -transformed.min.y,
            },
            OriginMode::Center => PointMm {
                x: -transformed.center().x,
                y: -transformed.center().y,
            },
        };
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineSettings {
    pub name: String,
    pub rapid_feed_mm_min: Option<f64>,
    pub motion_overhead_s: f64,
    pub work_area_mm: Option<Bounds2D>,
    pub max_s_value: f64,
}

impl Default for MachineSettings {
    fn default() -> Self {
        Self {
            name: "User machine".into(),
            rapid_feed_mm_min: Some(3000.0),
            motion_overhead_s: 0.0,
            work_area_mm: None,
            max_s_value: 1000.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    pub profile_id: String,
    pub include_frame_in_estimate: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            profile_id: "grbl-1.1".into(),
            include_frame_in_estimate: true,
        }
    }
}

/// Canvas grid background theme. `OffWhite` gives a light "paper" look for
/// checking dark artwork; `Dark` is the default night-friendly theme.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GridBackground {
    #[default]
    Dark,
    OffWhite,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplaySettings {
    pub display_unit: crate::units::LengthUnit,
    pub grid_enabled: bool,
    pub snap_to_grid: bool,
    pub color_tolerance: f64,
    /// `#[serde(default)]` keeps project files written before this field
    /// existed loadable (they read as the dark theme).
    #[serde(default)]
    pub grid_background: GridBackground,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            display_unit: crate::units::LengthUnit::Millimeter,
            grid_enabled: true,
            snap_to_grid: false,
            color_tolerance: crate::color::DEFAULT_COLOR_TOLERANCE,
            grid_background: GridBackground::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub schema_version: u32,
    pub app_version: String,
    pub document: VectorDocument,
    pub layers: Vec<ProcessLayer>,
    pub job: JobSettings,
    pub machine: MachineSettings,
    pub export: ExportSettings,
    pub display: DisplaySettings,
}

impl Project {
    /// Build a project from a freshly imported document.
    ///
    /// Layers are deliberately *not* created from the colour palette here:
    /// a layer is a process container, not a colour. The caller
    /// (`laser-app`) immediately runs the colour split so the imported paths
    /// land in one auto-generated layer per distinct source colour; this
    /// constructor only guarantees the always-present grey "Unassigned"
    /// layer that catches colour-less paths and paths of deleted layers.
    pub fn from_document(document: VectorDocument) -> Self {
        let bounds = document.bounds_mm;
        let layers: Vec<ProcessLayer> = vec![ProcessLayer::unassigned(0)];
        let job = JobSettings::default().with_origin(bounds, OriginMode::LowerLeft);
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            app_version: APP_VERSION.into(),
            document,
            layers,
            job,
            machine: MachineSettings::default(),
            export: ExportSettings::default(),
            display: DisplaySettings::default(),
        }
    }

    pub fn layer(&self, id: LayerId) -> Option<&ProcessLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut ProcessLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub fn sorted_layers(&self) -> Vec<&ProcessLayer> {
        let mut layers: Vec<_> = self.layers.iter().collect();
        layers.sort_by_key(|l| l.order);
        layers
    }

    pub fn transformed_bounds(&self) -> Result<Bounds2D, CoreError> {
        let xf = self.job.affine();
        self.document.bounds_mm.transformed(xf)
    }

    pub fn units_block_export(&self) -> bool {
        self.document.units.certainty == UnitCertainty::Ambiguous
    }

    pub fn validate_layers(&self) -> Result<(), CoreError> {
        for layer in &self.layers {
            layer.validate()?;
        }
        Ok(())
    }
}
