use std::fs;
use std::path::{Path, PathBuf};

use laser_core::{
    match_color, ColorMatch, ImportResult, LayerId, ObjectId, OperationKind, OriginMode,
    ProcessLayer, Project, PROJECT_SCHEMA_VERSION, Rgba8,
};
use laser_estimator::estimate;
pub use laser_estimator::EstimateReport;
use laser_gcode::{FirmwareProfile, PostprocessSettings, PowerMode};
use laser_import::{sniff_and_import, ImportError, ImportOptions};
use laser_import_dxf::DxfImporter;
use laser_import_svg::SvgImporter;
use laser_planner::{plan_project, transformed_preview_polylines, MotionPlan, PlanError, PlanOptions};
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
    apply_color_assignment(&mut project, false);
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

pub fn preview_color_assignment(project: &Project) -> ColorAssignmentPreview {
    let presets = laser_core::standard_palette();
    let tolerance = project.display.color_tolerance;
    let mut groups: Vec<ColorAssignmentGroup> = presets
        .iter()
        .map(|p| ColorAssignmentGroup {
            layer_name: p.name.clone(),
            color: p.color,
            object_ids: vec![],
            count: 0,
        })
        .collect();
    let mut unmatched: Vec<UnmatchedColor> = Vec::new();

    for obj in &project.document.objects {
        let Some(color) = obj.style.assignment_color() else {
            continue;
        };
        if let Some(ColorMatch { preset_name, .. }) = match_color(color, &presets, tolerance) {
            if let Some(g) = groups.iter_mut().find(|g| g.layer_name == preset_name) {
                g.object_ids.push(obj.id);
                g.count += 1;
            }
        } else if let Some(u) = unmatched.iter_mut().find(|u| u.color == color) {
            u.object_ids.push(obj.id);
        } else {
            unmatched.push(UnmatchedColor {
                color,
                object_ids: vec![obj.id],
            });
        }
    }
    ColorAssignmentPreview { matches: groups, unmatched }
}

pub fn apply_color_assignment(project: &mut Project, overwrite_manual: bool) -> ColorAssignmentPreview {
    let preview = preview_color_assignment(project);
    for group in &preview.matches {
        let Some(layer_id) = project
            .layers
            .iter()
            .find(|l| l.name == group.layer_name)
            .map(|l| l.id)
        else {
            continue;
        };
        for id in &group.object_ids {
            if let Some(obj) = project.document.object_mut(*id) {
                if overwrite_manual || obj.layer_id.is_none() {
                    obj.layer_id = Some(layer_id);
                }
            }
        }
    }
    preview
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
        *existing = layer;
        Ok(())
    } else {
        Err(AppError::Failed("unknown layer".into()))
    }
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

pub fn create_layer_from_selection(
    project: &mut Project,
    name: String,
    color: Rgba8,
    ids: &[ObjectId],
) -> LayerId {
    let order = project.layers.iter().map(|l| l.order).max().unwrap_or(0) + 1;
    let layer = ProcessLayer::new(name, color, order, OperationKind::Cut);
    let id = layer.id;
    for obj_id in ids {
        if let Some(obj) = project.document.object_mut(*obj_id) {
            obj.layer_id = Some(id);
        }
    }
    project.layers.push(layer);
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
pub struct PreviewGeometry {
    pub objects: Vec<PreviewObject>,
    pub bounds: laser_core::Bounds2D,
    pub document_bounds: laser_core::Bounds2D,
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
    PreviewGeometry {
        objects,
        bounds: project.transformed_bounds().unwrap_or(project.document.bounds_mm),
        document_bounds: project.document.bounds_mm,
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
    fn color_assign_red_to_cut() {
        let mut p = project_with_red_line();
        let preview = apply_color_assignment(&mut p, true);
        assert_eq!(preview.matches.iter().find(|g| g.layer_name == "Cut").unwrap().count, 1);
        assert!(p.document.objects[0].layer_id.is_some());
    }

    #[test]
    fn roundtrip_project_json() {
        let p = project_with_red_line();
        let json = serde_json::to_string(&p).unwrap();
        let back: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(back.document.objects.len(), 1);
    }

    #[test]
    fn frame_gcode_never_enables_laser() {
        let mut p = project_with_red_line();
        apply_color_assignment(&mut p, true);
        add_frame(&mut p, 1.0);
        let gcode = export_gcode(&p).unwrap();
        let frame_section = gcode.split("Layer: Frame").nth(1).unwrap_or("");
        let until_next = frame_section.split("Layer:").next().unwrap_or(frame_section);
        assert!(!until_next.contains("M3"));
        assert!(!until_next.contains("M4"));
        assert!(until_next.contains("M5"));
    }
}
