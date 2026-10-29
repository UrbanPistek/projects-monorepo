use std::sync::Mutex;

use laser_core::Project;

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
