mod system_info;

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub use system_info::{job_status, spawn_system_info_job};

/// Metadata for a single background job, exposed via /health.
#[derive(Debug, Clone, Serialize)]
pub struct JobStatus {
    pub id: String,
    pub name: String,
    pub interval_secs: u64,
    pub last_run_at: Option<DateTime<Utc>>,
    pub run_count: u64,
    pub status: String,
}

/// Top-level /health response combining server identity and all job statuses.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub server: String,
    pub total_jobs: usize,
    pub jobs: Vec<JobStatus>,
}

/// In-memory registry keyed by job id so each background task can update its own entry.
pub struct JobRegistry {
    jobs: HashMap<String, JobStatus>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    /// Register a job before spawning its background task so /health always lists it.
    pub fn register_job(&mut self, status: JobStatus) {
        self.jobs.insert(status.id.clone(), status);
    }

    /// Snapshot current state for the /health handler.
    pub fn to_health_response(&self) -> HealthResponse {
        let jobs: Vec<JobStatus> = self.jobs.values().cloned().collect();
        HealthResponse {
            server: "embed-server".to_string(),
            total_jobs: jobs.len(),
            jobs,
        }
    }
}

/// Shared handle passed to both HTTP handlers and background job tasks.
pub type SharedRegistry = Arc<RwLock<JobRegistry>>;

/// Record that a job just completed a run so /health reflects fresh timestamps.
pub fn update_job_run(registry: &SharedRegistry, job_id: &str) {
    let mut guard = registry.write().expect("job registry lock poisoned");
    if let Some(job) = guard.jobs.get_mut(job_id) {
        job.last_run_at = Some(Utc::now());
        job.run_count += 1;
    }
}
