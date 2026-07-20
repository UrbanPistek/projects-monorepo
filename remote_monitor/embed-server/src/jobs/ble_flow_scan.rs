use super::{update_job_run, JobStatus, SharedRegistry};
use bluer::{
    Adapter, AdapterEvent, Address, DeviceEvent, DiscoveryFilter, DiscoveryTransport,
};
use chrono::Utc;
use futures::{pin_mut, stream::SelectAll, StreamExt};
use secrecy::{ExposeSecret, SecretString};
use sqlx::postgres::PgConnection;
use sqlx::Connection;
use std::time::{Duration, Instant};

const JOB_ID: &str = "ble_flow_scan";
const TARGET_NAME: &str = "pico-w-ub";
const COMPANY_ID: u16 = 0x000D;
const PAYLOAD_LEN: usize = 9;
const QUIET_SECS: u64 = 5;
const RETRY_SECS: u64 = 5;

/// Decoded manufacturer payload from the Pico beacon (big-endian, 9 bytes total).
#[derive(Debug, Clone, Copy)]
struct FlowPayload {
    wake_count: u8,
    avg_flow_rate_lpm: f32,
    total_volumn_l: f32,
}

/// Tracks whether we already uploaded the first packet of the current advertising burst.
enum BurstState {
    Idle,
    InBurst { last_seen: Instant },
}

/// Spawn the BLE flow scanner that listens for Pico advertisements and writes to Postgres.
pub fn spawn_ble_flow_scan_job(registry: SharedRegistry) {
    tokio::spawn(async move {
        // Credentials are loaded once; each DB insert opens a fresh connection like system_info.
        dotenvy::dotenv().ok();
        let database_url = SecretString::from(
            std::env::var("DATABASE_URL").expect("DATABASE_URL must be set in .env"),
        );

        // Outer loop restarts the scan session after adapter/BLE failures.
        loop {
            set_job_status(&registry, JOB_ID, "running");

            match run_scan_loop(&registry, &database_url).await {
                Ok(()) => {
                    println!("[{JOB_ID}] scan loop ended unexpectedly, restarting");
                }
                Err(err) => {
                    eprintln!("[{JOB_ID}] scan error: {err}, retrying in {RETRY_SECS}s");
                    set_job_status(&registry, JOB_ID, "error");
                    tokio::time::sleep(Duration::from_secs(RETRY_SECS)).await;
                }
            }
        }
    });
}

/// Main scan loop: discover LE devices, subscribe to property changes, apply burst dedup.
async fn run_scan_loop(
    registry: &SharedRegistry,
    database_url: &SecretString,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let session = bluer::Session::new().await?;
    let adapter = session.default_adapter().await?;
    adapter.set_powered(true).await?;

    // LE-only filter keeps classic Bluetooth noise out of the discovery stream.
    let filter = DiscoveryFilter {
        transport: DiscoveryTransport::Le,
        ..Default::default()
    };
    adapter.set_discovery_filter(filter).await?;

    let device_events = adapter.discover_devices().await?;
    pin_mut!(device_events);

    let mut change_events = SelectAll::new();
    let mut burst_state = BurstState::Idle;
    let mut quiet_check = tokio::time::interval(Duration::from_secs(1));

    loop {
        tokio::select! {
            Some(adapter_event) = device_events.next() => {
                if let AdapterEvent::DeviceAdded(addr) = adapter_event {
                    // Name/manufacturer_data may arrive later; subscribe to property updates.
                    if let Ok(device) = adapter.device(addr) {
                        if let Ok(events) = device.events().await {
                            change_events.push(events.map(move |evt| (addr, evt)));
                        }
                    }

                    handle_device_update(
                        &adapter,
                        addr,
                        registry,
                        database_url,
                        &mut burst_state,
                    )
                    .await;
                }
            }
            Some((addr, device_event)) = change_events.next() => {
                // Manufacturer data and resolved name often show up on PropertyChanged.
                if matches!(device_event, DeviceEvent::PropertyChanged(_)) {
                    handle_device_update(
                        &adapter,
                        addr,
                        registry,
                        database_url,
                        &mut burst_state,
                    )
                    .await;
                }
            }
            _ = quiet_check.tick() => {
                // Return to Idle once the Pico has been silent for the full quiet window.
                if let BurstState::InBurst { last_seen } = &burst_state {
                    if last_seen.elapsed() >= Duration::from_secs(QUIET_SECS) {
                        burst_state = BurstState::Idle;
                    }
                }
            }
        }
    }
}

/// Read one device's current properties and apply name/payload/burst filters.
async fn handle_device_update(
    adapter: &Adapter,
    addr: Address,
    registry: &SharedRegistry,
    database_url: &SecretString,
    burst_state: &mut BurstState,
) {
    let device = match adapter.device(addr) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("[{JOB_ID}] device handle for {addr}: {err}");
            return;
        }
    };

    // Match the Python prototype: only process advertisements from the Pico short name.
    let name = match device.name().await {
        Ok(Some(name)) => name,
        Ok(None) => return,
        Err(err) => {
            eprintln!("[{JOB_ID}] name query for {addr}: {err}");
            return;
        }
    };

    if name != TARGET_NAME {
        return;
    }

    let manufacturer_data = match device.manufacturer_data().await {
        Ok(Some(data)) => data,
        Ok(None) => return,
        Err(err) => {
            eprintln!("[{JOB_ID}] manufacturer_data for {addr}: {err}");
            return;
        }
    };

    let payload_bytes = match manufacturer_data.get(&COMPANY_ID) {
        Some(bytes) => bytes.as_slice(),
        None => return,
    };

    let payload = match decode_manufacturer_payload(payload_bytes) {
        Some(payload) => payload,
        None => return,
    };

    let now = Instant::now();

    match burst_state {
        BurstState::InBurst { last_seen } => {
            // Any in-burst advertisement resets the quiet timer, even duplicate packets.
            *last_seen = now;
            return;
        }
        BurstState::Idle => {}
    }

    // Upload only the first non-zero packet per burst; zero payloads are ignored entirely.
    if !is_nonzero(&payload) {
        return;
    }

    let db_result = insert_flow_log(database_url, &payload).await;

    println!(
        "[{JOB_ID}] addr={addr} wake_count={} avg_flow_rate_lpm={:.3} total_volumn_l={:.3} db_insert={}",
        payload.wake_count,
        payload.avg_flow_rate_lpm,
        payload.total_volumn_l,
        match &db_result {
            Ok(()) => "ok".to_string(),
            Err(err) => format!("err: {err}"),
        }
    );

    if db_result.is_ok() {
        update_job_run(registry, JOB_ID);
        *burst_state = BurstState::InBurst { last_seen: now };
    }
}

/// Parse the 9-byte big-endian manufacturer payload emitted by the Pico firmware.
fn decode_manufacturer_payload(data: &[u8]) -> Option<FlowPayload> {
    if data.len() != PAYLOAD_LEN {
        return None;
    }

    let wake_count = data[0];
    let avg_flow_rate_lpm = f32::from_be_bytes(data[1..5].try_into().ok()?);
    let total_volumn_l = f32::from_be_bytes(data[5..9].try_into().ok()?);

    Some(FlowPayload {
        wake_count,
        avg_flow_rate_lpm,
        total_volumn_l,
    })
}

/// Gate uploads so idle/zero advertisements do not create DB rows.
fn is_nonzero(payload: &FlowPayload) -> bool {
    payload.wake_count != 0
        || payload.avg_flow_rate_lpm != 0.0
        || payload.total_volumn_l != 0.0
}

/// Open a new connection and insert one decoded flow row.
async fn insert_flow_log(
    database_url: &SecretString,
    payload: &FlowPayload,
) -> Result<(), sqlx::Error> {
    let mut connection = PgConnection::connect(database_url.expose_secret()).await?;

    sqlx::query(
        "INSERT INTO flow_data_logs (ts, wake_count, avg_flow_rate_lpm, total_volumn_l)
         VALUES ($1, $2, $3, $4)",
    )
    .bind(Utc::now())
    .bind(payload.wake_count as i32)
    .bind(payload.avg_flow_rate_lpm as f64)
    .bind(payload.total_volumn_l as f64)
    .execute(&mut connection)
    .await?;

    Ok(())
}

/// Update the job's health status string for /health reporting.
fn set_job_status(registry: &SharedRegistry, job_id: &str, status: &str) {
    let mut guard = registry.write().expect("job registry lock poisoned");
    if let Some(job) = guard.jobs.get_mut(job_id) {
        job.status = status.to_string();
    }
}

/// Build the initial JobStatus entry registered at startup.
pub fn job_status() -> JobStatus {
    JobStatus {
        id: JOB_ID.to_string(),
        name: "BLE Flow Scan".to_string(),
        interval_secs: QUIET_SECS,
        last_run_at: None,
        run_count: 0,
        status: "running".to_string(),
    }
}
