use laser_core::{
    flatten_geometry, polyline_length, Affine, Bounds2D, CoreError, LayerId, ObjectId,
    OperationKind, PointMm, ProcessLayer, Project,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MotionOp {
    RapidTo {
        to: PointMm,
    },
    LaserMoveTo {
        to: PointMm,
        feed_mm_min: f64,
        power_percent: f64,
    },
    LaserOffMoveTo {
        to: PointMm,
        feed_mm_min: f64,
    },
    SetLaser {
        power_percent: f64,
        dynamic: bool,
    },
    LaserOff,
    Comment {
        text: String,
    },
    ProgramEnd,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerPlanStats {
    pub layer_id: LayerId,
    pub name: String,
    pub path_count: u32,
    pub segment_count: u32,
    pub passes: u32,
    pub cut_length_mm: f64,
    pub travel_length_mm: f64,
    pub is_frame: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionPlan {
    pub ops: Vec<MotionOp>,
    pub bounds: Option<Bounds2D>,
    pub layers: Vec<LayerPlanStats>,
    pub chord_error_mm: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("unit scale is ambiguous; confirm units before planning")]
    AmbiguousUnits,
    #[error("{0}")]
    Invalid(String),
}

impl PlanError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Core(e) => e.code(),
            Self::AmbiguousUnits => "ambiguous_units",
            Self::Invalid(_) => "invalid_plan",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub require_known_units: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            require_known_units: true,
        }
    }
}

pub fn plan_project(project: &Project, options: PlanOptions) -> Result<MotionPlan, PlanError> {
    if options.require_known_units && project.units_block_export() {
        return Err(PlanError::AmbiguousUnits);
    }
    project.validate_layers()?;

    let xf = project.job.affine();
    let chord = project.job.chord_error_mm;
    let mut ops = vec![MotionOp::Comment {
        text: "Job start".into(),
    }];
    let mut stats = Vec::new();
    let mut cursor: Option<PointMm> = None;
    let mut all_pts: Vec<PointMm> = Vec::new();

    let mut layers: Vec<ProcessLayer> = project.layers.clone();
    layers.sort_by_key(|l| l.order);

    for layer in layers {
        if !layer.export_enabled {
            continue;
        }
        layer.validate()?;
        ops.push(MotionOp::Comment {
            text: format!("Layer: {}", layer.name),
        });
        if layer.operation == OperationKind::Frame {
            let frame_stats = emit_frame(project, &layer, xf, &mut ops, &mut cursor, &mut all_pts)?;
            stats.push(frame_stats);
            continue;
        }

        let mut objects: Vec<_> = project
            .document
            .objects
            .iter()
            .filter(|o| o.layer_id == Some(layer.id) && o.export_enabled)
            .collect();
        objects.sort_by_key(|o| o.id.0);

        let mut cut_len = 0.0;
        let mut travel_len = 0.0;
        let mut path_count = 0u32;
        let mut segment_count = 0u32;

        for _pass in 0..layer.passes.max(1) {
            for obj in &objects {
                let geometry = obj.geometry.transformed(xf);
                let polylines = flatten_geometry(&geometry, chord);
                for line in polylines {
                    if line.len() < 2 {
                        continue;
                    }
                    path_count += 1;
                    let start = line[0];
                    if let Some(prev) = cursor {
                        let d = prev.distance(start);
                        if d > 1e-6 {
                            ops.push(MotionOp::LaserOff);
                            ops.push(MotionOp::RapidTo { to: start });
                            travel_len += d;
                        }
                    } else {
                        ops.push(MotionOp::LaserOff);
                        ops.push(MotionOp::RapidTo { to: start });
                    }
                    if layer.operation != OperationKind::TravelOnly {
                        ops.push(MotionOp::SetLaser {
                            power_percent: layer.power_percent,
                            dynamic: matches!(layer.laser_mode, laser_core::LaserMode::Dynamic),
                        });
                    }
                    for w in line.windows(2) {
                        segment_count += 1;
                        let to = w[1];
                        let seg_len = w[0].distance(to);
                        cut_len += seg_len;
                        all_pts.push(to);
                        if layer.operation == OperationKind::TravelOnly
                            || layer.laser_must_remain_off
                        {
                            ops.push(MotionOp::LaserOffMoveTo {
                                to,
                                feed_mm_min: layer.speed_mm_min,
                            });
                        } else {
                            ops.push(MotionOp::LaserMoveTo {
                                to,
                                feed_mm_min: layer.speed_mm_min,
                                power_percent: layer.power_percent,
                            });
                        }
                    }
                    ops.push(MotionOp::LaserOff);
                    cursor = line.last().copied();
                    all_pts.extend_from_slice(&line);
                    let _ = ObjectId::new();
                }
            }
        }

        stats.push(LayerPlanStats {
            layer_id: layer.id,
            name: layer.name,
            path_count,
            segment_count,
            passes: layer.passes,
            cut_length_mm: cut_len,
            travel_length_mm: travel_len,
            is_frame: false,
        });
    }

    ops.push(MotionOp::LaserOff);
    ops.push(MotionOp::ProgramEnd);

    let bounds = Bounds2D::from_points(all_pts).ok();
    Ok(MotionPlan {
        ops,
        bounds,
        layers: stats,
        chord_error_mm: chord,
    })
}

fn emit_frame(
    project: &Project,
    layer: &ProcessLayer,
    xf: Affine,
    ops: &mut Vec<MotionOp>,
    cursor: &mut Option<PointMm>,
    all_pts: &mut Vec<PointMm>,
) -> Result<LayerPlanStats, PlanError> {
    let mut pts = Vec::new();
    for obj in &project.document.objects {
        if !obj.export_enabled {
            continue;
        }
        if let Some(id) = obj.layer_id {
            if let Some(src) = project.layer(id) {
                if !src.export_enabled || src.operation == OperationKind::Frame {
                    continue;
                }
            }
        } else {
            continue;
        }
        let geometry = obj.geometry.transformed(xf);
        for line in flatten_geometry(&geometry, project.job.chord_error_mm) {
            pts.extend(line);
        }
    }
    if pts.is_empty() {
        return Err(PlanError::Invalid(
            "frame requires export-enabled geometry".into(),
        ));
    }
    let bounds = Bounds2D::from_points(pts)?.expand(layer.frame_margin_mm);
    let corners = [
        bounds.min,
        PointMm {
            x: bounds.max.x,
            y: bounds.min.y,
        },
        bounds.max,
        PointMm {
            x: bounds.min.x,
            y: bounds.max.y,
        },
        bounds.min,
    ];
    ops.push(MotionOp::LaserOff);
    ops.push(MotionOp::RapidTo { to: corners[0] });
    let mut travel = 0.0;
    if let Some(prev) = *cursor {
        travel += prev.distance(corners[0]);
    }
    for w in corners.windows(2) {
        travel += w[0].distance(w[1]);
        ops.push(MotionOp::LaserOffMoveTo {
            to: w[1],
            feed_mm_min: layer.speed_mm_min,
        });
    }
    *cursor = Some(corners[0]);
    all_pts.extend_from_slice(&corners);
    Ok(LayerPlanStats {
        layer_id: layer.id,
        name: layer.name.clone(),
        path_count: 1,
        segment_count: 4,
        passes: 1,
        cut_length_mm: 0.0,
        travel_length_mm: travel,
        is_frame: true,
    })
}

pub fn transformed_preview_polylines(
    project: &Project,
) -> Vec<(ObjectId, Vec<Vec<PointMm>>, f64)> {
    let xf = project.job.affine();
    let chord = project.job.chord_error_mm;
    project
        .document
        .objects
        .iter()
        .map(|obj| {
            let geometry = obj.geometry.transformed(xf);
            let lines = flatten_geometry(&geometry, chord);
            let len: f64 = lines.iter().map(|l| polyline_length(l)).sum();
            (obj.id, lines, len)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use laser_core::{
        DocumentId, PathGeometry, PathObject, PathSegment, PointMm, SourceFormat, SourceMetadata,
        SourceStyle, Subpath, UnitMetadata, VectorDocument,
    };

    use super::*;

    fn square_project() -> Project {
        let geometry = PathGeometry {
            subpaths: vec![Subpath {
                start: PointMm { x: 0.0, y: 0.0 },
                segments: vec![
                    PathSegment::LineTo {
                        to: PointMm { x: 10.0, y: 0.0 },
                    },
                    PathSegment::LineTo {
                        to: PointMm { x: 10.0, y: 10.0 },
                    },
                    PathSegment::LineTo {
                        to: PointMm { x: 0.0, y: 10.0 },
                    },
                    PathSegment::LineTo {
                        to: PointMm { x: 0.0, y: 0.0 },
                    },
                ],
                closed: true,
            }],
        };
        let mut document = VectorDocument {
            id: DocumentId::new(),
            units: UnitMetadata::known(laser_core::LengthUnit::Millimeter),
            bounds_mm: Bounds2D {
                min: PointMm::origin(),
                max: PointMm { x: 10.0, y: 10.0 },
            },
            objects: vec![PathObject {
                id: ObjectId::new(),
                geometry,
                style: SourceStyle::default(),
                source: SourceMetadata {
                    format: SourceFormat::Generated,
                    file_name: "square".into(),
                    source_id: None,
                    source_layer: None,
                    entity_type: "path".into(),
                    original_color: None,
                },
                layer_id: None,
                visible: true,
                export_enabled: true,
            }],
            import_warnings: vec![],
            source_to_canonical: Affine::identity(),
            source_format: SourceFormat::Generated,
            source_file_name: "square".into(),
        };
        document.recompute_bounds().ok();
        let mut project = Project::from_document(document);
        let cut_id = project.layers[0].id;
        project.document.objects[0].layer_id = Some(cut_id);
        project
    }

    #[test]
    fn laser_off_between_paths_and_at_end() {
        let plan = plan_project(&square_project(), PlanOptions::default()).unwrap();
        assert!(matches!(plan.ops.last(), Some(MotionOp::ProgramEnd)));
        let mut laser_on = false;
        for op in &plan.ops {
            match op {
                MotionOp::SetLaser { .. } | MotionOp::LaserMoveTo { .. } => laser_on = true,
                MotionOp::LaserOff => laser_on = false,
                MotionOp::RapidTo { .. } => assert!(!laser_on, "rapid while laser on"),
                MotionOp::ProgramEnd => assert!(!laser_on),
                _ => {}
            }
        }
    }
}
