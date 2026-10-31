//! # Source style and provenance
//!
//! Why this exists separately from the layer: it records what the *file*
//! said (stroke/fill colours, entity type, source id), which is what the
//! "Assign by color" feature groups on. Layer assignment is a separate,
//! editable decision stored on the path itself.

use serde::{Deserialize, Serialize};

use crate::color::Rgba8;

/// Colours and stroke width as found in the source file. `assignment_color`
/// picks stroke first, then fill, because that is the convention laser
/// artwork uses (outlined shapes are cut/engraved along their stroke).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceStyle {
    pub stroke: Option<Rgba8>,
    pub fill: Option<Rgba8>,
    pub stroke_width_mm: Option<f64>,
}

impl SourceStyle {
    pub fn assignment_color(&self) -> Option<Rgba8> {
        self.stroke.or(self.fill)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceMetadata {
    pub format: SourceFormat,
    pub file_name: String,
    pub source_id: Option<String>,
    pub source_layer: Option<String>,
    pub entity_type: String,
    pub original_color: Option<Rgba8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SourceFormat {
    Svg,
    Dxf,
    Generated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportWarning {
    pub code: String,
    pub message: String,
    pub entity: Option<String>,
}

impl ImportWarning {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            entity: None,
        }
    }
}
