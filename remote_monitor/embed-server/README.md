# Mini Server for Raspberry Pi

```
Ubuntu 24.04.4 LTS (GNU/Linux 6.8.0-1060-raspi aarch64)
```

## Pi BLE Prerequisites

Before deploying the BLE flow scan job on the Pi:

```sh
sudo apt install bluez
sudo systemctl enable --now bluetooth
# Optional: allow scanning without root
sudo setcap cap_net_raw,cap_net_admin+eip ~/embed-server/embed-server
```

## Deploy Steps

Using cross: `cargo install cross`

```
cross build --release --target aarch64-unknown-linux-gnu
```

```sh
cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/embed-server "$HOST:/home/herb-remote-pi/embed-server"
scp .env "$HOST:/home/herb-remote-pi/embed-server"
```

## Auto start on Pi

```
/home/herb-remote-pi/embed-server/
├── embed-server      # binary
└── .env              # DATABASE_URL, etc.
```

Make the binary executable: `chmod +x /home/herb-remote-pi/embed-server/embed-server`

Create /etc/systemd/system/embed-server.service: `sudo vim /etc/systemd/system/embed-server.service`

```
[Unit]
Description=Remote monitor embed server
After=network-online.target bluetooth.service
Wants=network-online.target
Requires=bluetooth.service

[Service]
Type=simple
User=herb-remote-pi
Group=herb-remote-pi
WorkingDirectory=/home/herb-remote-pi/embed-server
ExecStart=/home/herb-remote-pi/embed-server/embed-server
Restart=on-failure
RestartSec=5

# Optional: log to journal instead of stdout only
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start it

```
sudo systemctl daemon-reload
sudo systemctl enable embed-server.service
sudo systemctl start embed-server.service
```

Verify

```
sudo systemctl status embed-server
journalctl -u embed-server -f
curl http://localhost:2849/health
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
