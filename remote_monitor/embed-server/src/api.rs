use crate::jobs::{HealthResponse, SharedRegistry};
use axum::{extract::State, Json};

/// GET /health — returns server identity and metadata for all background jobs.
pub async fn health_handler(State(registry): State<SharedRegistry>) -> Json<HealthResponse> {
    // Read lock is sufficient because handlers only snapshot; jobs hold write locks briefly.
    let guard = registry.read().expect("job registry lock poisoned");
    Json(guard.to_health_response())
}
