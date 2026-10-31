//! # Glazer desktop shell (Tauri)
//!
//! Thin wrapper around `laser-app`: owns the global [`AppState`] (the open
//! project behind a mutex) and exposes the `commands` as Tauri IPC handlers
//! that the Svelte frontend calls via `src/lib/api.ts`. All real logic lives in
//! `laser-app`; if it can be tested without Tauri, it is.

mod app_state;
mod commands;

use app_state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::import_file,
            commands::pick_and_import,
            commands::get_project_summary,
            commands::get_preview_geometry,
            commands::get_project,
            commands::assign_objects_to_layer,
            commands::auto_assign_by_color,
            commands::delete_layer,
            commands::update_display,
            commands::set_grid_background,
            commands::update_layer,
            commands::reorder_layers,
            commands::create_layer_from_selection,
            commands::create_frame_operation,
            commands::set_job_origin,
            commands::assume_units,
            commands::estimate_job,
            commands::validate_job,
            commands::export_gcode,
            commands::pick_and_export_gcode,
            commands::save_project,
            commands::pick_and_save_project,
            commands::open_project,
            commands::pick_and_open_project,
            commands::select_by_source_color,
            commands::update_machine_rapid,
            commands::update_job_settings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
