//! Global application state: the currently open project behind a mutex.
//! The `with_project`/`with_project_mut` helpers centralize locking and the
//! "no project is open" error so commands stay one-liners.

use std::sync::Mutex;

use laser_core::Project;

/// Holds the open project. `Mutex` because Tauri commands can run on
/// different threads; all access goes through the helpers below.
#[derive(Default)]
pub struct AppState {
    pub project: Mutex<Option<Project>>,
}

impl AppState {
    pub fn with_project<T>(
        &self,
        f: impl FnOnce(&Project) -> T,
    ) -> Result<T, laser_app::AppError> {
        let guard = self
            .project
            .lock()
            .map_err(|_| laser_app::AppError::Failed("state lock poisoned".into()))?;
        let project = guard
            .as_ref()
            .ok_or_else(|| laser_app::AppError::Failed("no project is open".into()))?;
        Ok(f(project))
    }

    pub fn with_project_mut<T>(
        &self,
        f: impl FnOnce(&mut Project) -> Result<T, laser_app::AppError>,
    ) -> Result<T, laser_app::AppError> {
        let mut guard = self
            .project
            .lock()
            .map_err(|_| laser_app::AppError::Failed("state lock poisoned".into()))?;
        let project = guard
            .as_mut()
            .ok_or_else(|| laser_app::AppError::Failed("no project is open".into()))?;
        f(project)
    }
}
