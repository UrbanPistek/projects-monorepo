//! # laser-app
//!
//! The service layer that glues the engine crates together for the desktop
//! app **without depending on Tauri**: import → project, layer operations
//! (assign by colour, delete, reorder, frame), preview geometry, estimation,
//! validation, G-code export and project save/load. `src-tauri` is a thin
//! command wrapper over these functions, which keeps all logic unit-testable
//! as plain Rust.
//!
//! Key behaviours implemented here:
//! - importing a file auto-splits paths into one layer per distinct source
//!   colour (`split_layers_by_color`);
//! - "Assign by color" re-runs that split and overwrites manual assignments;
//! - deleting a layer moves its paths to the grey Unassigned layer;
//! - frame layers are locked server-side (only passes/speed are editable);
//! - exports are written atomically (`.partial` + rename) so a crash never
//!   leaves a half-written G-code file.

use std::fs;
use std::path::{Path, PathBuf};

use laser_core::{
    match_color, AssignmentRule, Bounds2D, ImportResult, LayerId, ObjectId, OperationKind,
    OriginMode, ProcessLayer, Project, PROJECT_SCHEMA_VERSION, Rgba8,
};
use laser_estimator::estimate;
pub use laser_estimator::EstimateReport;
use laser_gcode::{FirmwareProfile, PostprocessSettings, PowerMode};
use laser_import::{sniff_and_import, ImportError, ImportOptions};
use laser_import_dxf::DxfImporter;
use laser_import_svg::SvgImporter;
use laser_planner::{
    frame_bounds, plan_project, transformed_preview_polylines, MotionPlan, PlanError, PlanOptions,
};
use laser_profile_grbl::Grbl11Profile;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error(transparent)]
    Import(#[from] ImportError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Gcode(#[from] laser_gcode::GcodeError),
    #[error(transparent)]
    Core(#[from] laser_core::CoreError),
    #[error("io: {0}")]
    Io(String),
    #[error("{0}")]
    Failed(String),
}

impl AppError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Import(e) => e.code(),
            Self::Plan(e) => e.code(),
            Self::Gcode(e) => e.code(),
            Self::Core(e) => e.code(),
            Self::Io(_) => "io",
            Self::Failed(_) => "failed",
        }
    }
}

impl serde::Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("AppError", 2)?;
        s.serialize_field("code", self.code())?;
        s.serialize_field("message", &self.to_string())?;
        s.end()
    }
}

pub fn import_file(path: &Path, assumed_unit: Option<laser_core::LengthUnit>) -> Result<ImportResult, AppError> {
    let bytes = fs::read(path).map_err(|e| AppError::Io(e.to_string()))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file")
        .to_string();
    import_bytes(&name, &bytes, assumed_unit)
}

pub fn import_bytes(
    file_name: &str,
    bytes: &[u8],
    assumed_unit: Option<laser_core::LengthUnit>,
) -> Result<ImportResult, AppError> {
    let svg = SvgImporter;
    let dxf = DxfImporter;
    let mut options = ImportOptions::new(file_name);
    options.assumed_unit = assumed_unit;
    Ok(sniff_and_import(&[&svg, &dxf], file_name, bytes, options)?)
}

pub fn open_project_from_import(result: ImportResult) -> Project {
    let mut project = Project::from_document(result.document);
    // Import automatically splits paths into one layer per distinct source
    // colour, but only fills in paths that are not assigned yet.
    split_layers_by_color(&mut project, false);
    project
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorAssignmentPreview {
    pub matches: Vec<ColorAssignmentGroup>,
    pub unmatched: Vec<UnmatchedColor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColorAssignmentGroup {
    pub layer_name: String,
    pub color: Rgba8,
    pub object_ids: Vec<ObjectId>,
    pub count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UnmatchedColor {
    pub color: Rgba8,
    pub object_ids: Vec<ObjectId>,
}

/// Display name for an auto-generated colour layer, e.g. `Red #FF0000`.
fn color_layer_name(color: Rgba8) -> String {
    let named = match_color(color, &laser_core::standard_palette(), laser_core::DEFAULT_COLOR_TOLERANCE)
        .map(|m| m.preset_name);
    match named {
        Some(name) if name == "Cut" => format!("Cut {}", color.to_hex()),
        Some(name) => format!("{} {}", name, color.to_hex()),
        None => format!("Color {}", color.to_hex()),
    }
}

/// Group paths by their exact source colour. Each distinct colour becomes one
/// entry in `matches`; paths with no stroke/fill colour are not listed (they
/// belong to the Unassigned layer).
pub fn preview_color_split(project: &Project) -> ColorAssignmentPreview {
    let mut groups: Vec<ColorAssignmentGroup> = Vec::new();
    let unmatched: Vec<UnmatchedColor> = Vec::new();

    for obj in &project.document.objects {
        let Some(color) = obj.style.assignment_color() else {
            continue;
        };
        if let Some(g) = groups.iter_mut().find(|g| g.color == color) {
            g.object_ids.push(obj.id);
            g.count += 1;
        } else {
            groups.push(ColorAssignmentGroup {
                layer_name: color_layer_name(color),
                color,
                object_ids: vec![obj.id],
                count: 1,
            });
        }
    }
    ColorAssignmentPreview {
        matches: groups,
        unmatched,
    }
}

/// The "Assign by color" feature: one layer per distinct source colour,
/// created on demand and linked to its colour via `AssignmentRule::Color`.
/// Layers are plain process containers — a layer may hold paths of several
/// colours, and a path may be moved to any layer manually afterwards.
///
/// `overwrite_manual = true` moves *all* matching paths (used by the toolbar
/// button); `false` only fills in still-unassigned paths (used on import).
pub fn split_layers_by_color(project: &mut Project, overwrite_manual: bool) -> ColorAssignmentPreview {
    let preview = preview_color_split(project);
    let unassigned_id = ensure_unassigned_layer(project);
    let tolerance = project.display.color_tolerance;

    for group in &preview.matches {
        // Reuse an existing layer bound to this colour, else create one.
        let layer_id = match project.layers.iter().find(|l| {
            matches!(&l.source_assignment, AssignmentRule::Color { hex, .. } if Rgba8::from_hex(hex) == Some(group.color))
        }) {
            Some(l) => {
                let id = l.id;
                if let Some(l) = project.layer_mut(id) {
                    l.display_color = group.color;
                }
                id
            }
            None => {
                // Keep the old convention: red-ish colours default to a cut,
                // everything else to an engrave, so bulk assignment does not
                // silently change the process type of existing artwork.
                let is_cut = match_color(group.color, &laser_core::standard_palette(), tolerance)
                    .map(|m| m.preset_name == "Cut")
                    .unwrap_or(false);
                let operation = if is_cut {
                    OperationKind::Cut
                } else {
                    OperationKind::EngraveLine
                };
                let order = project.layers.iter().map(|l| l.order).max().unwrap_or(0) + 1;
                let mut layer = ProcessLayer::new(color_layer_name(group.color), group.color, order, operation);
                layer.source_assignment = AssignmentRule::Color {
                    hex: group.color.to_hex(),
                    tolerance,
                };
                let id = layer.id;
                project.layers.push(layer);
                id
            }
        };
        for id in &group.object_ids {
            if let Some(obj) = project.document.object_mut(*id) {
                if overwrite_manual || obj.layer_id.is_none() {
                    obj.layer_id = Some(layer_id);
                }
            }
        }
    }

    // Colour-less paths always land in Unassigned.
    for obj in &mut project.document.objects {
        if obj.style.assignment_color().is_none() && (overwrite_manual || obj.layer_id.is_none()) {
            obj.layer_id = Some(unassigned_id);
        }
    }
    preview
}

/// Guarantee the grey "Unassigned" layer exists and return its id.
pub fn ensure_unassigned_layer(project: &mut Project) -> LayerId {
    if let Some(l) = project
        .layers
        .iter()
        .find(|l| matches!(l.source_assignment, AssignmentRule::Unassigned))
    {
        return l.id;
    }
    let order = project.layers.iter().map(|l| l.order).max().unwrap_or(0) + 1;
    let layer = ProcessLayer::unassigned(order);
    let id = layer.id;
    project.layers.push(layer);
    id
}

pub fn assign_objects(project: &mut Project, ids: &[ObjectId], layer_id: LayerId) -> Result<(), AppError> {
    if project.layer(layer_id).is_none() {
        return Err(AppError::Failed("unknown layer".into()));
    }
    for id in ids {
        if let Some(obj) = project.document.object_mut(*id) {
            obj.layer_id = Some(layer_id);
        }
    }
    Ok(())
}

pub fn update_layer(project: &mut Project, layer: ProcessLayer) -> Result<(), AppError> {
    layer.validate()?;
    if let Some(existing) = project.layer_mut(layer.id) {
        if existing.operation == OperationKind::Frame {
            // Frame layers are fixed except for the safe-to-change fields:
            // passes, speed, frame margin, and the visibility/export toggles
            // (the frame outline itself scales with the job-level scale).
            // Merge only the allowed fields so the UI round-trip cannot
            // unlock the rest (name, operation, power, laser mode, colour).
            let mut merged = existing.clone();
            merged.passes = layer.passes;
            merged.speed_mm_min = layer.speed_mm_min;
            merged.frame_margin_mm = layer.frame_margin_mm;
            merged.visible = layer.visible;
            merged.export_enabled = layer.export_enabled;
            merged.validate()?;
            *existing = merged;
        } else {
            if layer.operation == OperationKind::Frame {
                return Err(AppError::Failed(
                    "frame layers are created with the Add frame button".into(),
                ));
            }
            *existing = layer;
        }
        Ok(())
    } else {
        Err(AppError::Failed("unknown layer".into()))
    }
}

/// Delete a layer. Its paths are moved to the grey "Unassigned" layer so no
/// geometry is lost; the Unassigned layer itself can never be deleted.
pub fn delete_layer(project: &mut Project, layer_id: LayerId) -> Result<(), AppError> {
    let Some(existing) = project.layer(layer_id) else {
        return Err(AppError::Failed("unknown layer".into()));
    };
    if matches!(existing.source_assignment, AssignmentRule::Unassigned) {
        return Err(AppError::Failed(
            "the Unassigned layer cannot be deleted".into(),
        ));
    }
    let unassigned_id = ensure_unassigned_layer(project);
    for obj in &mut project.document.objects {
        if obj.layer_id == Some(layer_id) {
            obj.layer_id = Some(unassigned_id);
        }
    }
    project.layers.retain(|l| l.id != layer_id);
    // Reindex orders so the remaining layers stay a dense 0..n sequence.
    let mut ordered: Vec<LayerId> = project
        .layers
        .iter()
        .map(|l| l.id)
        .collect();
    ordered.sort_by_key(|id| project.layer(*id).map(|l| l.order).unwrap_or(0));
    reorder_layers(project, &ordered)?;
    Ok(())
}

pub fn reorder_layers(project: &mut Project, ordered_ids: &[LayerId]) -> Result<(), AppError> {
    for (order, id) in ordered_ids.iter().enumerate() {
        if let Some(layer) = project.layer_mut(*id) {
            layer.order = order as u32;
        } else {
            return Err(AppError::Failed("unknown layer in reorder".into()));
        }
    }
    Ok(())
}

/// Create a new empty layer at the end of the stack (frame stays first at
/// order 0). Defaults to a cut operation; the caller can rename/recolor and
/// change the operation afterwards via `update_layer`.
pub fn create_layer(project: &mut Project, name: String, color: Rgba8) -> LayerId {
    let order = project.layers.iter().map(|l| l.order).max().unwrap_or(0) + 1;
    let layer = ProcessLayer::new(name, color, order, OperationKind::Cut);
    let id = layer.id;
    project.layers.push(layer);
    id
}

pub fn create_layer_from_selection(
    project: &mut Project,
    name: String,
    color: Rgba8,
    ids: &[ObjectId],
) -> LayerId {
    let id = create_layer(project, name, color);
    for obj_id in ids {
        if let Some(obj) = project.document.object_mut(*obj_id) {
            obj.layer_id = Some(id);
        }
    }
    id
}

pub fn add_frame(project: &mut Project, margin_mm: f64) -> LayerId {
    project.layers.retain(|l| l.operation != OperationKind::Frame);
    let order = 0u32;
    for layer in &mut project.layers {
        layer.order += 1;
    }
    let layer = ProcessLayer::frame(order, margin_mm);
    let id = layer.id;
    project.layers.push(layer);
    id
}

pub fn set_origin(project: &mut Project, mode: OriginMode) {
    project.job = project
        .job
        .clone()
        .with_origin(project.document.bounds_mm, mode);
}

pub fn assume_units(project: &mut Project, unit: laser_core::LengthUnit) -> Result<(), AppError> {
    let current = project.document.units.conversion_unit();
    let from = current.to_mm_factor().unwrap_or(1.0);
    let to = unit
        .to_mm_factor()
        .ok_or_else(|| AppError::Failed("cannot assume unknown unit".into()))?;
    let scale = to / from;
    let xf = laser_core::Affine::scale(scale, scale);
    for obj in &mut project.document.objects {
        obj.geometry = obj.geometry.transformed(xf);
    }
    project.document.units = laser_core::UnitMetadata::assumed(
        project.document.units.source_unit,
        unit,
        format!("User confirmed units as {}", unit.label()),
    );
    project.document.source_to_canonical = xf.then(project.document.source_to_canonical);
    project.document.recompute_bounds()?;
    set_origin(project, project.job.origin_mode);
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewObject {
    pub id: ObjectId,
    pub layer_id: Option<LayerId>,
    pub visible: bool,
    pub export_enabled: bool,
    pub color: String,
    pub polylines: Vec<Vec<[f64; 2]>>,
    pub source_id: Option<String>,
    pub entity_type: String,
    pub length_mm: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FramePreview {
    pub bounds: Bounds2D,
    pub passes: u32,
    pub speed_mm_min: f64,
    pub margin_mm: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewGeometry {
    pub objects: Vec<PreviewObject>,
    pub bounds: laser_core::Bounds2D,
    pub document_bounds: laser_core::Bounds2D,
    /// Rectangle the frame layer will trace, if a frame layer exists.
    /// Drawn on the canvas as a dashed orange outline on top of everything.
    pub frame: Option<FramePreview>,
}

pub fn preview_geometry(project: &Project) -> PreviewGeometry {
    let lines = transformed_preview_polylines(project);
    let mut objects = Vec::new();
    for (id, polylines, length) in lines {
        let obj = project.document.objects.iter().find(|o| o.id == id).unwrap();
        let color = obj
            .layer_id
            .and_then(|id| project.layer(id))
            .map(|l| l.display_color.to_hex())
            .or_else(|| obj.style.assignment_color().map(|c| c.to_hex()))
            .unwrap_or_else(|| "#888888".into());
        let layer_visible = obj
            .layer_id
            .and_then(|lid| project.layer(lid))
            .map(|l| l.visible)
            .unwrap_or(true);
        objects.push(PreviewObject {
            id: obj.id,
            layer_id: obj.layer_id,
            visible: obj.visible && layer_visible,
            export_enabled: obj.export_enabled
                && obj
                    .layer_id
                    .and_then(|lid| project.layer(lid))
                    .map(|l| l.export_enabled)
                    .unwrap_or(false),
            color,
            polylines: polylines
                .into_iter()
                .map(|line| line.into_iter().map(|p| [p.x, p.y]).collect())
                .collect(),
            source_id: obj.source.source_id.clone(),
            entity_type: obj.source.entity_type.clone(),
            length_mm: length,
        });
    }
    // The frame outline is only previewed while the frame layer is visible
    // (visibility never affects export — that is the export toggle's job).
    let frame = project
        .layers
        .iter()
        .find(|l| l.operation == OperationKind::Frame && l.visible)
        .and_then(|l| {
            frame_bounds(project, l)
                .ok()
                .flatten()
                .map(|bounds| FramePreview {
                    bounds,
                    passes: l.passes,
                    speed_mm_min: l.speed_mm_min,
                    margin_mm: l.frame_margin_mm,
                })
        });
    PreviewGeometry {
        objects,
        bounds: project.transformed_bounds().unwrap_or(project.document.bounds_mm),
        document_bounds: project.document.bounds_mm,
        frame,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub file_name: String,
    pub object_count: usize,
    pub layer_count: usize,
    pub bounds: laser_core::Bounds2D,
    pub units: laser_core::UnitMetadata,
    pub warnings: Vec<laser_core::ImportWarning>,
    pub profile_id: String,
}

pub fn project_summary(project: &Project) -> ProjectSummary {
    ProjectSummary {
        file_name: project.document.source_file_name.clone(),
        object_count: project.document.objects.len(),
        layer_count: project.layers.len(),
        bounds: project.transformed_bounds().unwrap_or(project.document.bounds_mm),
        units: project.document.units.clone(),
        warnings: project.document.import_warnings.clone(),
        profile_id: project.export.profile_id.clone(),
    }
}

pub fn build_plan(project: &Project) -> Result<MotionPlan, AppError> {
    Ok(plan_project(project, PlanOptions::default())?)
}

pub fn estimate_project(project: &Project) -> Result<EstimateReport, AppError> {
    let plan = build_plan(project)?;
    Ok(estimate(&plan, &project.machine))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationReport {
    pub ok: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn validate_project(project: &Project) -> ValidationReport {
    let mut errors = Vec::new();
    let mut warnings = project
        .document
        .import_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect::<Vec<_>>();
    if project.units_block_export() {
        errors.push("Unit scale is ambiguous. Confirm units before export.".into());
    }
    if let Err(e) = project.validate_layers() {
        errors.push(e.to_string());
    }
    let exportable = project.document.objects.iter().any(|o| {
        o.export_enabled
            && o.layer_id
                .and_then(|id| project.layer(id))
                .map(|l| l.export_enabled && l.operation != OperationKind::Frame)
                .unwrap_or(false)
    });
    if !exportable {
        errors.push("No export-enabled geometry is assigned to an export-enabled layer.".into());
    }
    if let Ok(bounds) = project.transformed_bounds() {
        if let Some(area) = project.machine.work_area_mm {
            if bounds.min.x < area.min.x
                || bounds.min.y < area.min.y
                || bounds.max.x > area.max.x
                || bounds.max.y > area.max.y
            {
                warnings.push("Job bounds exceed the configured machine work area.".into());
            }
        }
    }
    match build_plan(project) {
        Ok(_) => {}
        Err(e) => errors.push(e.to_string()),
    }
    ValidationReport {
        ok: errors.is_empty(),
        errors,
        warnings,
    }
}

pub fn export_gcode(project: &Project) -> Result<String, AppError> {
    let report = validate_project(project);
    if !report.ok {
        return Err(AppError::Failed(report.errors.join(" ")));
    }
    if project.export.profile_id != laser_profile_grbl::PROFILE_ID {
        return Err(AppError::Failed(format!(
            "unsupported profile {}",
            project.export.profile_id
        )));
    }
    let plan = build_plan(project)?;
    let profile = Grbl11Profile::default();
    let power_mode = if project
        .layers
        .iter()
        .any(|l| l.export_enabled && l.laser_mode == laser_core::LaserMode::Dynamic)
    {
        PowerMode::DynamicM4
    } else {
        PowerMode::ConstantM3
    };
    Ok(profile.postprocess(
        &plan,
        &project.machine,
        &PostprocessSettings {
            max_s: project.machine.max_s_value,
            default_power_mode: power_mode,
        },
    )?)
}

pub fn export_gcode_atomic(project: &Project, dest: &Path) -> Result<(), AppError> {
    let gcode = export_gcode(project)?;
    let tmp = dest.with_extension("gcode.partial");
    fs::write(&tmp, gcode.as_bytes()).map_err(|e| AppError::Io(e.to_string()))?;
    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        AppError::Io(e.to_string())
    })?;
    Ok(())
}

pub fn save_project(project: &Project, path: &Path) -> Result<(), AppError> {
    if project.schema_version != PROJECT_SCHEMA_VERSION {
        return Err(AppError::Failed("unsupported schema version".into()));
    }
    let json = serde_json::to_string_pretty(project).map_err(|e| AppError::Failed(e.to_string()))?;
    let tmp = PathBuf::from(format!("{}.partial", path.display()));
    fs::write(&tmp, json).map_err(|e| AppError::Io(e.to_string()))?;
    fs::rename(&tmp, path).map_err(|e| AppError::Io(e.to_string()))?;
    Ok(())
}

pub fn open_project(path: &Path) -> Result<Project, AppError> {
    let json = fs::read_to_string(path).map_err(|e| AppError::Io(e.to_string()))?;
    let project: Project =
        serde_json::from_str(&json).map_err(|e| AppError::Failed(format!("project parse: {e}")))?;
    if project.schema_version != PROJECT_SCHEMA_VERSION {
        return Err(AppError::Failed(format!(
            "unsupported project schema {}",
            project.schema_version
        )));
    }
    Ok(project)
}

pub fn select_by_source_color(project: &Project, color: Rgba8) -> Vec<ObjectId> {
    project
        .document
        .objects
        .iter()
        .filter(|o| o.style.assignment_color() == Some(color))
        .map(|o| o.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use laser_core::{
        Affine, Bounds2D, DocumentId, PathGeometry, PathObject, PathSegment, PointMm, SourceFormat,
        SourceMetadata, SourceStyle, Subpath, UnitMetadata, VectorDocument,
    };

    fn project_with_red_line() -> Project {
        let geometry = PathGeometry {
            subpaths: vec![Subpath {
                start: PointMm { x: 0.0, y: 0.0 },
                segments: vec![PathSegment::LineTo {
                    to: PointMm { x: 10.0, y: 0.0 },
                }],
                closed: false,
            }],
        };
        let document = VectorDocument {
            id: DocumentId::new(),
            units: UnitMetadata::known(laser_core::LengthUnit::Millimeter),
            bounds_mm: Bounds2D {
                min: PointMm::origin(),
                max: PointMm { x: 10.0, y: 0.0 },
            },
            objects: vec![PathObject {
                id: ObjectId::new(),
                geometry,
                style: SourceStyle {
                    stroke: Some(Rgba8::rgb(255, 0, 0)),
                    fill: None,
                    stroke_width_mm: None,
                },
                source: SourceMetadata {
                    format: SourceFormat::Generated,
                    file_name: "t".into(),
                    source_id: None,
                    source_layer: None,
                    entity_type: "path".into(),
                    original_color: Some(Rgba8::rgb(255, 0, 0)),
                },
                layer_id: None,
                visible: true,
                export_enabled: true,
            }],
            import_warnings: vec![],
            source_to_canonical: Affine::identity(),
            source_format: SourceFormat::Generated,
            source_file_name: "t".into(),
        };
        Project::from_document(document)
    }

    #[test]
    fn color_split_creates_layer_per_distinct_color() {
        let mut p = project_with_red_line();
        let preview = split_layers_by_color(&mut p, true);
        assert_eq!(preview.matches.len(), 1);
        assert_eq!(preview.matches[0].count, 1);
        let layer_id = p.document.objects[0].layer_id.unwrap();
        let layer = p.layer(layer_id).unwrap();
        // Red matches the "Cut" convention, so the auto layer is a cut and
        // shows the source colour.
        assert_eq!(layer.operation, OperationKind::Cut);
        assert_eq!(layer.display_color, Rgba8::rgb(255, 0, 0));
        assert!(matches!(layer.source_assignment, AssignmentRule::Color { .. }));
    }

    #[test]
    fn color_split_without_overwrite_keeps_manual_assignment() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        let obj_id = p.document.objects[0].id;
        let manual = create_layer_from_selection(&mut p, "Manual".into(), Rgba8::rgb(1, 2, 3), &[obj_id]);
        assert_eq!(p.document.objects[0].layer_id, Some(manual));
        split_layers_by_color(&mut p, false);
        assert_eq!(p.document.objects[0].layer_id, Some(manual));
        split_layers_by_color(&mut p, true);
        assert_ne!(p.document.objects[0].layer_id, Some(manual));
    }

    #[test]
    fn delete_layer_moves_paths_to_unassigned() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        let layer_id = p.document.objects[0].layer_id.unwrap();
        let unassigned_id = p
            .layers
            .iter()
            .find(|l| matches!(l.source_assignment, AssignmentRule::Unassigned))
            .unwrap()
            .id;
        delete_layer(&mut p, layer_id).unwrap();
        assert!(p.layer(layer_id).is_none());
        assert_eq!(p.document.objects[0].layer_id, Some(unassigned_id));
        // Unassigned itself cannot be deleted.
        assert!(delete_layer(&mut p, unassigned_id).is_err());
    }

    #[test]
    fn frame_layer_update_locks_fixed_fields() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        let frame_id = add_frame(&mut p, 2.0);
        let mut incoming = p.layer(frame_id).unwrap().clone();
        incoming.name = "Hacked".into();
        incoming.power_percent = 90.0;
        incoming.laser_mode = laser_core::LaserMode::Constant;
        incoming.passes = 4;
        incoming.speed_mm_min = 321.0;
        incoming.frame_margin_mm = 7.5;
        incoming.visible = false;
        incoming.export_enabled = false;
        update_layer(&mut p, incoming).unwrap();
        let frame = p.layer(frame_id).unwrap();
        // Fixed fields stay locked...
        assert_eq!(frame.name, "Frame");
        assert_eq!(frame.power_percent, 0.0);
        assert_eq!(frame.laser_mode, laser_core::LaserMode::Dynamic);
        // ...while passes, speed, margin and the toggles are editable.
        assert_eq!(frame.passes, 4);
        assert_eq!(frame.speed_mm_min, 321.0);
        assert_eq!(frame.frame_margin_mm, 7.5);
        assert!(!frame.visible);
        assert!(!frame.export_enabled);
        // A normal layer cannot be turned into a frame via update_layer.
        let other = p.document.objects[0].layer_id.unwrap();
        let mut incoming = p.layer(other).unwrap().clone();
        incoming.operation = OperationKind::Frame;
        incoming.laser_must_remain_off = true;
        assert!(update_layer(&mut p, incoming).is_err());
    }

    #[test]
    fn frame_visibility_toggle_persists() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        let frame_id = add_frame(&mut p, 2.0);
        assert!(p.layer(frame_id).unwrap().visible);
        let mut incoming = p.layer(frame_id).unwrap().clone();
        incoming.visible = false;
        update_layer(&mut p, incoming).unwrap();
        assert!(!p.layer(frame_id).unwrap().visible);
        // And the frame rectangle disappears from the preview when hidden?
        // (Preview still reports it; canvas respects layer.visible via object
        // visibility, frame outline visibility is a UI concern — just assert
        // the flag round-trips through a save/load cycle.)
        let json = serde_json::to_string(&p).unwrap();
        let back: Project = serde_json::from_str(&json).unwrap();
        assert!(!back.layer(frame_id).unwrap().visible);
    }

    #[test]
    fn create_layer_appends_empty_layer() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        let id = create_layer(&mut p, "Layer 2".into(), Rgba8::rgb(0, 255, 0));
        let layer = p.layer(id).unwrap();
        assert_eq!(layer.name, "Layer 2");
        assert_eq!(layer.operation, OperationKind::Cut);
        // New layer goes last: the frame (order 0) stays first.
        let frame_id = add_frame(&mut p, 2.0);
        let id2 = create_layer(&mut p, "Layer 3".into(), Rgba8::rgb(0, 0, 255));
        let sorted = p.sorted_layers();
        assert_eq!(sorted[0].id, frame_id);
        assert_eq!(sorted.last().unwrap().id, id2);
        assert!(p.document.objects.iter().all(|o| o.layer_id != Some(id)));
    }

    #[test]
    fn roundtrip_project_json() {
        let p = project_with_red_line();
        let json = serde_json::to_string(&p).unwrap();
        let back: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(back.document.objects.len(), 1);
    }

    #[test]
    fn old_project_json_without_grid_background_loads() {
        // schema v1 files predate display.gridBackground; it must default.
        let p = project_with_red_line();
        let mut json: serde_json::Value = serde_json::to_value(&p).unwrap();
        json["display"].as_object_mut().unwrap().remove("gridBackground");
        let back: Project = serde_json::from_value(json).unwrap();
        assert_eq!(back.display.grid_background, laser_core::GridBackground::Dark);
    }

    #[test]
    fn frame_gcode_never_enables_laser() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        add_frame(&mut p, 1.0);
        let gcode = export_gcode(&p).unwrap();
        let frame_section = gcode.split("Layer: Frame").nth(1).unwrap_or("");
        let until_next = frame_section.split("Layer:").next().unwrap_or(frame_section);
        assert!(!until_next.contains("M3"));
        assert!(!until_next.contains("M4"));
        assert!(until_next.contains("M5"));
    }

    #[test]
    fn preview_includes_frame_rectangle() {
        let mut p = project_with_red_line();
        split_layers_by_color(&mut p, true);
        assert!(preview_geometry(&p).frame.is_none());
        let frame_id = add_frame(&mut p, 2.0);
        let frame = preview_geometry(&p).frame.unwrap();
        assert!((frame.bounds.max.x - 12.0).abs() < 1e-9);
        assert!((frame.bounds.min.x - -2.0).abs() < 1e-9);
        // Hiding the frame layer hides the outline (but not its export).
        let mut incoming = p.layer(frame_id).unwrap().clone();
        incoming.visible = false;
        update_layer(&mut p, incoming).unwrap();
        assert!(preview_geometry(&p).frame.is_none());
    }
}
