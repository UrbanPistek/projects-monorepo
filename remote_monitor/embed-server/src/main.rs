mod api;
mod jobs;

use jobs::{spawn_system_info_job, JobRegistry, SharedRegistry};
use std::sync::{Arc, RwLock};

const PORT: u16 = 2849;

#[tokio::main]
async fn main() {
    // Build registry and pre-register jobs so /health lists them before the first tick.
    let mut registry = JobRegistry::new();
    registry.register_job(jobs::job_status());

    let shared_registry: SharedRegistry = Arc::new(RwLock::new(registry));

    // Start background jobs before binding HTTP so they run alongside the server.
    spawn_system_info_job(shared_registry.clone());

    let app = axum::Router::new()
        .route("/health", axum::routing::get(api::health_handler))
        .with_state(shared_registry);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", PORT))
        .await
        .expect(format!("failed to bind to port {}", PORT).as_str());

    println!("embed-server listening on http://0.0.0.0:{}", PORT);

    axum::serve(listener, app)
        .await
        .expect("server error");
}
