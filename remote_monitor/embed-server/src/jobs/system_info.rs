use super::{update_job_run, JobStatus, SharedRegistry};
use chrono::Utc;
use secrecy::{ExposeSecret, SecretString};
use sqlx::postgres::PgConnection;
use sqlx::Connection;
use std::time::Duration;
use sysinfo::System;

const JOB_ID: &str = "system_info";
const INTERVAL_SECS: u64 = 60*60*12; // 12 hours

/// Spawn the system-info reporter that ticks every 15 seconds.
pub fn spawn_system_info_job(registry: SharedRegistry) {
    tokio::spawn(async move {
        // Load credentials once at startup; the connection itself is recreated each tick.
        dotenvy::dotenv().ok();
        let database_url = SecretString::from(
            std::env::var("DATABASE_URL").expect("DATABASE_URL must be set in .env"),
        );

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

            // Source column stores the full report so rows are self-contained in the DB.
            let source = format!(
                "host={hostname} cpus={cpu_count} cpu_usage={cpu_usage:.1}% \
                 mem={used_mem}/{total_mem} KB uptime={uptime}s"
            );

            // Fresh connection per iteration as requested; avoids holding stale pool state.
            let host_name_etry = hostname.as_str();
            let db_insert = insert_log_entry(&database_url, &host_name_etry).await;

            println!(
                "[system_info] {source} db_insert={}",
                match &db_insert {
                    Ok(()) => "ok".to_string(),
                    Err(err) => format!("err: {err}"),
                }
            );

            // Update registry after logging so /health shows when this job last ran.
            update_job_run(&registry, JOB_ID);
        }
    });
}

pub async fn get_initial_system_info(database_url: &SecretString) {
    let mut system = System::new();
    system.refresh_memory();
    system.refresh_cpu_usage();

    let hostname = System::host_name().unwrap_or_else(|| "unknown".to_string());
    let cpu_count = system.cpus().len();
    let cpu_usage: f32 = system.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>()
        / cpu_count.max(1) as f32;
    let total_mem = system.total_memory();
    let used_mem = system.used_memory();
    let uptime = System::uptime();

    // Source column stores the full report so rows are self-contained in the DB.
    let host_name_etry = hostname.as_str();
    let source = format!(
        "host={hostname} cpus={cpu_count} cpu_usage={cpu_usage:.1}% \
         mem={used_mem}/{total_mem} KB uptime={uptime}s"
    );

    // Fresh connection at startup; same one-shot insert pattern as the periodic tick.
    let db_insert = insert_log_entry(database_url, &host_name_etry).await;

    println!(
        "[system_info] {source} db_insert={}",
        match &db_insert {
            Ok(()) => "ok".to_string(),
            Err(err) => format!("err: {err}"),
        }
    );
}

/// Open a new connection and insert one row into embed_server_logging.
async fn insert_log_entry(
    database_url: &SecretString,
    source: &str,
) -> Result<(), sqlx::Error> {
    let mut connection =
        PgConnection::connect(database_url.expose_secret()).await?;

    sqlx::query(
        "INSERT INTO embed_server_logging (ts, source) VALUES ($1, $2)",
    )
    .bind(Utc::now())
    .bind(source)
    .execute(&mut connection)
    .await?;

    Ok(())
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
