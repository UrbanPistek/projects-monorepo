# Mini Server for Raspberry Pi

## Deploy Steps

Using cross: `cargo install cross`

```sh
cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/embed-server "$HOST:/home/herb-remote-pi/embed-server"
scp .env "$HOST:/home/herb-remote-pi/embed-server"
```

## Data Usage

These are typical on-the-wire totals (TCP + TLS + PostgreSQL), not Rust/sqlx internal allocations.

Operation	Approx. bytes (round-trip total)	What dominates
Open connection	~5–20 KB+	TLS handshake + auth (you use sslmode=require)
Close connection	~200–500 B	Terminate + TCP FIN
Idle open connection	~0 B/min (no traffic)	Small TCP keepalive probes only
One simple INSERT	~1–3 KB + your row size	Extended query messages + 1 row of payload

On most Linux systems (including Raspberry Pi OS), defaults look like:

Setting	Typical default	Meaning
tcp_keepalive_time	7200 s (2 h)	Idle time before the first probe
tcp_keepalive_intvl	75 s	Gap between failed probes
tcp_keepalive_probes	9	Probes before the connection is dropped
So for a healthy idle connection:

Hour 0–1: no keepalive probes → ~0 bytes
At ~2 h idle: 1 probe + 1 ACK → ~100–200 bytes total
Then the idle timer resets; another ~2 h until the next cycle
Average over 24 h with defaults: about 12 probe cycles × ~150 B ≈ ~1.8 KB/day — roughly ~75 bytes/hour long-term average, but not evenly spread per hour.
