use super::{update_job_run, JobStatus, SharedRegistry};
use std::time::Duration;
use sysinfo::System;

const JOB_ID: &str = "system_info";
const INTERVAL_SECS: u64 = 15;

/// Spawn the system-info reporter that ticks every 15 seconds.
pub fn spawn_system_info_job(registry: SharedRegistry) {
    tokio::spawn(async move {
        // Skip the immediate first tick so the server can start before the first report.
        let mut interval = tokio::time::interval(Duration::from_secs(INTERVAL_SECS));
        interval.tick().await;

        // Reuse one System instance across ticks to avoid reallocating collectors each cycle.
        let mut system = System::new();

        loop {
            interval.tick().await;

            // Refresh only the fields we print to keep each tick lightweight.
            system.refresh_memory();
            system.refresh_cpu_usage();

            let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
            let cpu_count = system.cpus().len();
            let cpu_usage: f32 = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
                / cpu_count.max(1) as f32;
            let total_mem = system.total_memory();
            let used_mem = system.used_memory();
            let uptime = System::uptime();

            println!(
                "[system_info] host={hostname} cpus={cpu_count} cpu_usage={cpu_usage:.1}% \
                 mem={used_mem}/{total_mem} KB uptime={uptime}s"
            );

            // Update registry after printing so /health shows when this job last ran.
            update_job_run(&registry, JOB_ID);
        }
    });
}

/// Build the initial JobStatus entry registered at startup.
pub fn job_status() -> JobStatus {
    JobStatus {
        id: JOB_ID.to_string(),
        name: "System Info Reporter".to_string(),
        interval_secs: INTERVAL_SECS,
        last_run_at: None,
        run_count: 0,
        status: "running".to_string(),
    }
}
