use std::path::PathBuf;

use laser_app::{
    self, ColorAssignmentPreview, EstimateReport, PreviewGeometry, ProjectSummary, ValidationReport,
};
use laser_core::{LayerId, LengthUnit, ObjectId, OriginMode, ProcessLayer, Project, Rgba8};
use tauri::State;
use tauri_plugin_dialog::DialogExt;

use crate::app_state::AppState;

#[tauri::command]
pub fn import_file(
    path: String,
    assumed_unit: Option<LengthUnit>,
    state: State<AppState>,
) -> Result<ProjectSummary, laser_app::AppError> {
    let result = laser_app::import_file(&PathBuf::from(path), assumed_unit)?;
    let project = laser_app::open_project_from_import(result);
    let summary = laser_app::project_summary(&project);
    *state
        .project
        .lock()
        .map_err(|_| laser_app::AppError::Failed("state lock poisoned".into()))? = Some(project);
    Ok(summary)
}

#[tauri::command]
pub async fn pick_and_import(
    assumed_unit: Option<LengthUnit>,
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, laser_app::AppError> {
    let path = app
        .dialog()
        .file()
        .add_filter("Vector", &["svg", "dxf"])
        .blocking_pick_file();
    let path = path.ok_or_else(|| laser_app::AppError::Failed("import cancelled".into()))?;
    let path = path
        .into_path()
        .map_err(|e| laser_app::AppError::Failed(e.to_string()))?;
    import_file(path.to_string_lossy().into_owned(), assumed_unit, state)
}

#[tauri::command]
pub fn get_project_summary(state: State<AppState>) -> Result<ProjectSummary, laser_app::AppError> {
    state.with_project(laser_app::project_summary)
}

#[tauri::command]
pub fn get_preview_geometry(state: State<AppState>) -> Result<PreviewGeometry, laser_app::AppError> {
    state.with_project(laser_app::preview_geometry)
}

#[tauri::command]
pub fn get_project(state: State<AppState>) -> Result<Project, laser_app::AppError> {
    state.with_project(|p| p.clone())
}

#[tauri::command]
pub fn assign_objects_to_layer(
    object_ids: Vec<ObjectId>,
    layer_id: LayerId,
    state: State<AppState>,
) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| laser_app::assign_objects(p, &object_ids, layer_id))
}

#[tauri::command]
pub fn auto_assign_by_color(
    apply: bool,
    state: State<AppState>,
) -> Result<ColorAssignmentPreview, laser_app::AppError> {
    if apply {
        state.with_project_mut(|p| Ok(laser_app::apply_color_assignment(p, true)))
    } else {
        state.with_project(laser_app::preview_color_assignment)
    }
}

#[tauri::command]
pub fn update_layer(layer: ProcessLayer, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| laser_app::update_layer(p, layer))
}

#[tauri::command]
pub fn reorder_layers(ordered_ids: Vec<LayerId>, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| laser_app::reorder_layers(p, &ordered_ids))
}

#[tauri::command]
pub fn create_layer_from_selection(
    name: String,
    color: Rgba8,
    object_ids: Vec<ObjectId>,
    state: State<AppState>,
) -> Result<LayerId, laser_app::AppError> {
    state.with_project_mut(|p| Ok(laser_app::create_layer_from_selection(p, name, color, &object_ids)))
}

#[tauri::command]
pub fn create_frame_operation(
    margin_mm: f64,
    state: State<AppState>,
) -> Result<LayerId, laser_app::AppError> {
    state.with_project_mut(|p| Ok(laser_app::add_frame(p, margin_mm)))
}

#[tauri::command]
pub fn set_job_origin(mode: OriginMode, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| {
        laser_app::set_origin(p, mode);
        Ok(())
    })
}

#[tauri::command]
pub fn assume_units(unit: LengthUnit, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| laser_app::assume_units(p, unit))
}

#[tauri::command]
pub fn estimate_job(state: State<AppState>) -> Result<EstimateReport, laser_app::AppError> {
    state.with_project(|p| laser_app::estimate_project(p))?
}

#[tauri::command]
pub fn validate_job(state: State<AppState>) -> Result<ValidationReport, laser_app::AppError> {
    state.with_project(laser_app::validate_project)
}

#[tauri::command]
pub fn export_gcode(path: String, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project(|p| laser_app::export_gcode_atomic(p, &PathBuf::from(path)))?
}

#[tauri::command]
pub async fn pick_and_export_gcode(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), laser_app::AppError> {
    let path = app
        .dialog()
        .file()
        .add_filter("G-code", &["gcode", "nc", "cnc"])
        .set_file_name("job.gcode")
        .blocking_save_file();
    let path = path.ok_or_else(|| laser_app::AppError::Failed("export cancelled".into()))?;
    let path = path
        .into_path()
        .map_err(|e| laser_app::AppError::Failed(e.to_string()))?;
    export_gcode(path.to_string_lossy().into_owned(), state)
}

#[tauri::command]
pub fn save_project(path: String, state: State<AppState>) -> Result<(), laser_app::AppError> {
    state.with_project(|p| laser_app::save_project(p, &PathBuf::from(path)))?
}

#[tauri::command]
pub async fn pick_and_save_project(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), laser_app::AppError> {
    let path = app
        .dialog()
        .file()
        .add_filter("Glazer project", &["lasercam.json", "json"])
        .set_file_name("job.lasercam.json")
        .blocking_save_file();
    let path = path.ok_or_else(|| laser_app::AppError::Failed("save cancelled".into()))?;
    let path = path
        .into_path()
        .map_err(|e| laser_app::AppError::Failed(e.to_string()))?;
    save_project(path.to_string_lossy().into_owned(), state)
}

#[tauri::command]
pub fn open_project(path: String, state: State<AppState>) -> Result<ProjectSummary, laser_app::AppError> {
    let project = laser_app::open_project(&PathBuf::from(path))?;
    let summary = laser_app::project_summary(&project);
    *state
        .project
        .lock()
        .map_err(|_| laser_app::AppError::Failed("state lock poisoned".into()))? = Some(project);
    Ok(summary)
}

#[tauri::command]
pub async fn pick_and_open_project(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<ProjectSummary, laser_app::AppError> {
    let path = app
        .dialog()
        .file()
        .add_filter("Glazer project", &["json"])
        .blocking_pick_file();
    let path = path.ok_or_else(|| laser_app::AppError::Failed("open cancelled".into()))?;
    let path = path
        .into_path()
        .map_err(|e| laser_app::AppError::Failed(e.to_string()))?;
    open_project(path.to_string_lossy().into_owned(), state)
}

#[tauri::command]
pub fn select_by_source_color(
    color: Rgba8,
    state: State<AppState>,
) -> Result<Vec<ObjectId>, laser_app::AppError> {
    state.with_project(|p| laser_app::select_by_source_color(p, color))
}

#[tauri::command]
pub fn update_machine_rapid(
    rapid_feed_mm_min: Option<f64>,
    max_s_value: f64,
    state: State<AppState>,
) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| {
        p.machine.rapid_feed_mm_min = rapid_feed_mm_min;
        p.machine.max_s_value = max_s_value;
        Ok(())
    })
}

#[tauri::command]
pub fn update_job_settings(
    scale: f64,
    rotation_deg: f64,
    chord_error_mm: f64,
    state: State<AppState>,
) -> Result<(), laser_app::AppError> {
    state.with_project_mut(|p| {
        p.job.scale = scale;
        p.job.rotation_deg = rotation_deg;
        p.job.chord_error_mm = chord_error_mm;
        Ok(())
    })
}
