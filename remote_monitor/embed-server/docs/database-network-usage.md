# Database Network Usage

Notes on wire-level data costs for PostgreSQL connections used by `embed-server`
(Neon over TLS, `sqlx`, one `INSERT` into `embed_server_logging` per tick).

**Important:** sqlx does not expose byte counters. The figures below are typical
on-the-wire totals (TCP + TLS + PostgreSQL protocol), not Rust heap allocations.
Latency (round-trip time) usually matters more than raw byte count for remote
databases like Neon.

---

## Operation sizes (typical)

| Operation | Approx. bytes (round-trip total) | What dominates |
|---|---|---|
| **Open connection** | ~5–20 KB+ | TLS handshake + auth (`sslmode=require`) |
| **Close connection** | ~200–500 B | PostgreSQL `Terminate` + TCP FIN |
| **Idle open connection** | ~0 B/min (no app traffic) | TCP keepalive probes only (see below) |
| **One simple INSERT** | ~1–3 KB + row payload | Extended query messages + one row |

For our `system_info` job, the `source` column is ~80–120 bytes, so the INSERT
payload is small. **Opening a connection is usually much more expensive than the
query itself.**

With Neon (remote, pooled server-side), each new client connect often pays:

1. TCP to AWS
2. Full TLS handshake
3. PostgreSQL startup + auth
4. Possible server-side pool assignment

That can be **100–500+ ms** even if only ~10 KB moved.

---

## Persisting a connection (pool)

### On the wire (idle)

Essentially **zero bytes** between queries when the connection is healthy and idle.
Only occasional TCP keepalive probes (see [TCP keepalive](#tcp-keepalive-probes)).

### In memory

| Side | Cost |
|---|---|
| **Client (Pi)** | One open socket + TLS session state — tens to low hundreds of KB |
| **Server (Neon/Postgres)** | One backend process — often several MB RAM per connection |

Persisting a connection saves repeated TLS + handshake bytes and latency, at the
cost of holding RAM and a server slot.

For a 15-second tick rate, a **`PgPool` with `max_connections(1)`** reused
across ticks is almost always better than connect → insert → drop every time.

---

## Per-tick estimate (current pattern)

The `system_info` job currently opens a fresh `PgConnection` each tick:

```rust
let mut connection = PgConnection::connect(database_url.expose_secret()).await?;
sqlx::query("INSERT INTO embed_server_logging (ts, source) VALUES ($1, $2)")
    .bind(Utc::now())
    .bind(source)
    .execute(&mut connection)
    .await?;
```

Approximate **per 15-second tick**:

| Phase | Approx. bytes |
|---|---|
| Open (TLS + auth) | ~5–20 KB in + out |
| INSERT | ~1–3 KB |
| Close | ~0.5 KB |
| **Total per tick** | **~7–25 KB** |

Rough average: **~0.5–1.7 KB/s** — but **latency and Neon connection churn**
matter more than bandwidth.

With a pooled connection (reuse after first tick):

| Phase | Approx. bytes |
|---|---|
| First tick | Same as above (~7–25 KB) |
| Subsequent ticks | ~1–3 KB (INSERT only) |

---

## How to test each component

### 1. Measure connect vs query latency (easiest)

Add timing around each phase in Rust. This does not give byte counts but shows
which phase dominates in practice.

```rust
use std::time::Instant;

async fn insert_log_entry(
    database_url: &SecretString,
    source: &str,
) -> Result<(), sqlx::Error> {
    let t0 = Instant::now();
    let mut connection =
        PgConnection::connect(database_url.expose_secret()).await?;
    let connect_ms = t0.elapsed().as_millis();

    let t1 = Instant::now();
    sqlx::query("INSERT INTO embed_server_logging (ts, source) VALUES ($1, $2)")
        .bind(Utc::now())
        .bind(source)
        .execute(&mut connection)
        .await?;
    let query_ms = t1.elapsed().as_millis();

    println!("connect={connect_ms}ms query={query_ms}ms");

    Ok(())
}
```

**What to look for:** With Neon, `connect_ms` is usually much larger than
`query_ms`. If connect dominates, switch to a connection pool.

**Compare patterns:**

| Test | How |
|---|---|
| Connect + query + drop (current) | Time as above every tick |
| Pooled connection | Create `PgPool` once at startup; time only the `execute` call |
| Query only (warm pool) | After pool is warm, skip connect timing on ticks 2+ |

---

### 2. Capture actual wire bytes (`tcpdump` + Wireshark)

This is the only way to get real byte counts for TLS + PostgreSQL combined.

**On the Pi** (replace host with your Neon hostname from `DATABASE_URL`):

```bash
# Record traffic while embed-server runs for one or more ticks
sudo tcpdump -i any -nn host YOUR_NEON_HOST and port 5432 -w pg.pcap
```

Stop with `Ctrl+C` after a few insert cycles.

**Analyze in Wireshark** (copy `pg.pcap` to your dev machine if needed):

1. Open `pg.pcap`
2. Filter: `tcp.port == 5432`
3. Select one full cycle: TCP connect → TLS → INSERT → disconnect
4. **Statistics → Conversations → TCP** — bytes per flow
5. Or select packets for that flow and sum the **Length** column

**Isolate components in one capture:**

| Component | What to select in Wireshark |
|---|---|
| **Open connection** | Packets from TCP SYN through first PostgreSQL `ReadyForQuery` |
| **Query only** | Packets between `Parse`/`Bind`/`Execute` and the matching response |
| **Close connection** | Final PostgreSQL `Terminate` + TCP FIN/RST |

**One idle hour (keepalive only):**

```bash
# Start capture, then open ONE connection and leave it idle for 1+ hours
sudo tcpdump -i any -nn host YOUR_NEON_HOST and port 5432 -w pg-idle.pcap
```

In Wireshark, use **Statistics → I/O Graph** or count packets/bytes over the
idle window. Compare hour 0–1 vs hour 1–2 (defaults often differ; see keepalive
section below).

---

### 3. Server-side query stats (`pg_stat_statements`)

Measures **query execution time and call count**, not wire bytes. Useful for
comparing INSERT cost over time.

Enable on Neon (if available on your plan) or self-hosted Postgres, then:

```sql
SELECT calls, mean_exec_time, total_exec_time, query
FROM pg_stat_statements
WHERE query LIKE '%embed_server_logging%';
```

Also useful:

```sql
-- Server-side work for a single INSERT (not network bytes)
EXPLAIN ANALYZE
INSERT INTO embed_server_logging (ts, source)
VALUES (NOW(), 'test');
```

---

### 4. Neon dashboard

Neon provides connection count, query volume, and compute metrics. These do not
show per-query byte counts but help spot connection churn (many connects/disconnects
vs one persistent connection).

---

## TCP keepalive probes

When a PostgreSQL connection is left open and idle, the application sends no data.
The only ongoing traffic is **TCP keepalive probes** (unless a load balancer or
Neon pooler closes the connection due to its own idle timeout).

### Linux defaults (Raspberry Pi OS)

Check on the Pi:

```bash
sysctl net.ipv4.tcp_keepalive_time net.ipv4.tcp_keepalive_intvl net.ipv4.tcp_keepalive_probes
```

| Setting | Typical default | Meaning |
|---|---|---|
| `tcp_keepalive_time` | **7200 s (2 h)** | Idle time before the **first** probe |
| `tcp_keepalive_intvl` | 75 s | Gap between **unanswered** probes |
| `tcp_keepalive_probes` | 9 | Probes before the kernel drops the connection |

For a **healthy idle connection** (peer responds to each probe):

| Time window | Probes | Approx. bytes |
|---|---|---|
| Hour 0–1 | 0 | **~0 bytes** |
| At ~2 h idle | 1 probe + 1 ACK | **~100–200 bytes** |
| Next 2 h idle | same cycle repeats | timer resets after successful ACK |

Long-term average with defaults: ~12 probe cycles/day × ~150 B ≈ **~1.8 KB/day**
(~**75 bytes/hour** average), but **not evenly spread** — most hours are 0 bytes.

### Bytes per probe (when one fires)

| Layer | Approx. size |
|---|---|
| One direction (IP + TCP, often with timestamps) | ~40–60 bytes |
| Round trip (probe + ACK) | ~80–120 bytes at IP level |
| On the wire (Ethernet framing) | ~100–200 bytes total |

### PostgreSQL / libpq keepalive overrides

libpq (used by sqlx) can override OS defaults via the connection URL:

```text
?keepalives=1&keepalives_idle=600&keepalives_interval=30&keepalives_count=3
```

With `keepalives_idle=600` (10 min):

- First probe after 10 min idle
- Roughly 1 probe + ACK every 10 min while idle
- **Per hour: ~6 cycles × ~150 B ≈ ~900 bytes/hour**

Still negligible compared to opening a new TLS connection (~5–20 KB once).

### How to test keepalive bytes

**Step 1 — Check OS settings:**

```bash
sysctl net.ipv4.tcp_keepalive_time net.ipv4.tcp_keepalive_intvl net.ipv4.tcp_keepalive_probes
```

**Step 2 — Check libpq settings** in your `DATABASE_URL` for `keepalives_*` params.

**Step 3 — Capture idle traffic:**

```bash
sudo tcpdump -i any -nn host YOUR_NEON_HOST and port 5432 -w pg-idle.pcap
```

Open one connection (e.g. `psql "$DATABASE_URL"` or a small script that connects
and sleeps), leave idle for 1–2 hours, stop capture.

**Step 4 — Analyze in Wireshark:**

- Filter `tcp.port == 5432`
- **Statistics → I/O Graph** — bytes over time during idle period
- Count small ACK-only packets during hours with no INSERTs

**Step 5 — Optional: force more frequent probes for testing**

Temporarily add to `DATABASE_URL`:

```text
keepalives=1&keepalives_idle=60&keepalives_interval=10&keepalives_count=3
```

Reconnect, idle for 5–10 minutes, capture again. You should see probe cycles
roughly every 60 s (~150 B each) — easier to observe in a short capture.

### Neon / cloud caveat

Neon and load balancers may have **idle timeouts** (sometimes a few minutes).
They may close idle connections rather than relying on TCP keepalive alone.
In that case you pay **reconnect cost** on the next query, not ongoing keepalive
bytes.

---

## Summary comparison

| Scenario | ~Bytes per idle hour | Notes |
|---|---|---|
| Linux defaults (`keepalive_time=7200`) | **0** in first hour; often 0 in second | Probe at ~2 h mark |
| Aggressive keepalive (`idle=600`) | **~0.5–1 KB/hour** | libpq URL params |
| New TLS connect every 15 s | **~1–5 MB/hour** | Dominated by handshakes |
| Pooled connection, INSERT every 15 s | **~240 B/s avg** (~1–3 KB per INSERT) | Recommended |

---

## Recommendations

1. **Time connect vs query** in code first — quickest way to see if pooling helps.
2. **Use `tcpdump` + Wireshark** when you need exact byte counts.
3. **Use `pg_stat_statements`** for query timing trends, not wire size.
4. **Prefer `PgPool` with `max_connections(1)`** for the 15 s tick job unless you
   have a reason to reconnect every tick.
5. **Keepalive traffic is negligible** (~0–1 KB/hour typical) compared to
   repeated TLS handshakes.
