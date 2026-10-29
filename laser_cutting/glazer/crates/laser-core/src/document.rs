use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::flatten::{flatten_geometry, polyline_length};
use crate::geom::{Affine, Bounds2D, PathGeometry, PointMm};
use crate::ids::{DocumentId, LayerId, ObjectId};
use crate::style::{ImportWarning, SourceMetadata, SourceStyle};
use crate::units::UnitMetadata;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathObject {
    pub id: ObjectId,
    pub geometry: PathGeometry,
    pub style: SourceStyle,
    pub source: SourceMetadata,
    pub layer_id: Option<LayerId>,
    pub visible: bool,
    pub export_enabled: bool,
}

impl PathObject {
    pub fn bounds(&self) -> Result<Bounds2D, CoreError> {
        self.geometry.bounds()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorDocument {
    pub id: DocumentId,
    pub units: UnitMetadata,
    pub bounds_mm: Bounds2D,
    pub objects: Vec<PathObject>,
    pub import_warnings: Vec<ImportWarning>,
    pub source_to_canonical: Affine,
    pub source_format: crate::style::SourceFormat,
    pub source_file_name: String,
}

impl VectorDocument {
    pub fn recompute_bounds(&mut self) -> Result<(), CoreError> {
        let mut acc: Option<Bounds2D> = None;
        for obj in &self.objects {
            if let Ok(b) = obj.bounds() {
                acc = Some(match acc {
                    Some(prev) => prev.union(b),
                    None => b,
                });
            }
        }
        self.bounds_mm = acc.unwrap_or(Bounds2D {
            min: PointMm::origin(),
            max: PointMm::origin(),
        });
        Ok(())
    }

    pub fn object_mut(&mut self, id: ObjectId) -> Option<&mut PathObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    pub fn total_path_length(&self, chord_error_mm: f64) -> f64 {
        self.objects
            .iter()
            .flat_map(|o| flatten_geometry(&o.geometry, chord_error_mm))
            .map(|line| polyline_length(&line))
            .sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportReport {
    pub imported: u32,
    pub approximated: u32,
    pub skipped: u32,
    pub unit_certainty: crate::units::UnitCertainty,
    pub bounds_mm: Bounds2D,
    pub warnings: Vec<ImportWarning>,
    pub entity_counts: Vec<EntityCount>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityCount {
    pub entity_type: String,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportResult {
    pub document: VectorDocument,
    pub report: ImportReport,
}
