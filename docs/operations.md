# Operations

Running this server day to day: starting/stopping, logs, health checks,
backups, updates, and host tuning. For first-time setup, see the
[README quickstart](../README.md#quickstart).

## Start / stop / restart

The server is started by hand when it's needed (no service, nothing runs at boot).

**Start:** double-click `scripts\start-laya-mcp.bat` (or run it from a prompt). It
- says so and just prints the status if the server is already running,
- otherwise opens a window titled "Laya MCP Server :8765" running the server,
- waits for `/health` (the english model loads in ~30-60 s; gives up after 180 s),
- prints the health status, the intranet URL and the `claude mcp add` command for clients.

Settings live at the top of the .bat (port, threads, preload); anything else goes in
`D:\Laya_Sandbox\.env` (see `.env.example`).

**Stop:** close the "Laya MCP Server" window, or press Ctrl+C in it. In-flight batch
items stay queued and resume on the next start.

**Restart** (after changing settings or updating code): stop, then start again.

Without the .bat, from an ordinary (non-elevated) prompt:

```powershell
cd D:\Laya_Sandbox
venv\Scripts\laya-mcp.exe
```

### Reaching it from other machines (firewall)

The first time the server listens on the network, Windows Defender Firewall may ask
whether to allow `python.exe`: allow it on **Private** networks (and make sure the
Wi-Fi profile is Private, not Public). If there was no prompt and other machines
still can't connect, an administrator can add a rule once:

```powershell
New-NetFirewallRule -DisplayName "Laya MCP 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress LocalSubnet -Action Allow -Profile Private,Domain
```

### Optional: run as a service instead

`scripts/install_task.ps1` (run as Administrator) registers a scheduled task that starts
the server at boot and restarts it on failure, and adds the firewall rule above. Not
used on this host today; `install_task.ps1 -Uninstall` removes both if it's ever
installed.

## Logs

Structured JSON logs (structlog) write under `data/logs` inside the project
directory (`D:\Laya_Sandbox\data\logs`). Raw request/state content is never
logged by default (`LAYA_MCP_LOG_CONTENT=0`) — logs cover request metadata,
timings, and errors only. Turn content logging on temporarily only for local
debugging, and turn it back off — this host may see real customer data once
support/sales come online.

Windows Scheduled Task history for `LayaMCP` (start/stop/crash/restart
events, not application logs) is visible in Task Scheduler under
Task Scheduler Library, or:

```powershell
Get-ScheduledTaskInfo -TaskName LayaMCP
Get-WinEvent -LogName 'Microsoft-Windows-TaskScheduler/Operational' |
    Where-Object { $_.Message -like '*LayaMCP*' } | Select-Object -First 20
```

## Health and metrics

```bash
curl http://localhost:8765/health      # liveness + loaded models, from any intranet host: http://10.10.29.81:8765/health
curl http://localhost:8765/metrics     # Prometheus text format (prometheus-fastapi-instrumentator)
```

For a fuller runtime snapshot from inside Claude Code (or any MCP client),
call the `laya_status` tool: loaded/available checkpoints, device, thread
count, queue depth (`waiting`), RSS memory, uptime, and the rolling ms/row
estimate used for batch-job ETAs.

## Backups

Everything that matters lives in two SQLite files under `data/`:

| File | Contents |
|---|---|
| `data/laya_mcp.db` | Schemas + versions, datasets, evaluation reports, calibrations, usage events |
| `data/jobs.db` | Huey job queue + batch/evaluate job results |

Back these up before any upgrade, and periodically otherwise (e.g. before a
`laya_promote_schema` you'd hate to redo). Since they're SQLite files, a
straight file copy is a valid backup as long as the server is stopped (or at
least not mid-write) when you take it:

```powershell
Stop-ScheduledTask -TaskName LayaMCP
Copy-Item data\laya_mcp.db, data\jobs.db "D:\Backups\laya-mcp\$(Get-Date -Format yyyyMMdd)\" -Force
Start-ScheduledTask -TaskName LayaMCP
```

`data/laya_mcp.db` is the one that's expensive to lose (evaluated/promoted
schemas represent real labeling effort); `data/jobs.db` mostly just replays
in-flight batch work, which is cheaper to re-submit.

## Updating

**Updating laya-mcp itself** (this project's code):

```powershell
Stop-ScheduledTask -TaskName LayaMCP
cd D:\Laya_Sandbox
git pull                                         # or however you're syncing changes
venv\Scripts\python.exe -m pip install -e ".[dev]"
# If a new Alembic migration was added, it runs automatically at startup
# (Repo.migrate() runs `alembic upgrade head` on start) -- back up
# data\laya_mcp.db first regardless.
Start-ScheduledTask -TaskName LayaMCP
curl http://localhost:8765/health
```

**Updating the `laya` library** (the model/inference package, pinned in
`pyproject.toml` as `laya==0.3.7` because the server reuses some of its
internals — check `D:\laya` and the plan doc before bumping the pin, since a
version bump can change internals this project depends on directly):

```powershell
Stop-ScheduledTask -TaskName LayaMCP
venv\Scripts\python.exe -m pip install laya==<new-version>
venv\Scripts\python.exe -I -c "import laya; print(laya.__version__)"   # confirm
Start-ScheduledTask -TaskName LayaMCP
```

A new `laya` version can change confidence calibration even for the same
checkpoint — re-run `laya_evaluate`/`laya_calibrate` on `trusted` schemas
after a bump rather than assuming old thresholds still hold.

## Changing threads / row budgets

Both live in `.env` (see `.env.example` for the full list) and take effect on
the next restart:

- `LAYA_MCP_THREADS` — torch intra-op threads. Keep at or below the 8
  physical cores, and leave headroom for MySQL/Postgres/httpd/Ollama also
  running on this host. `scripts/bench.py` (Phase 0) sweeps 4/6/8 and reports
  the best-performing setting for this hardware; start there rather than
  guessing.
- `LAYA_MCP_SYNC_ROW_BUDGET` — max `items x questions` allowed in one
  synchronous call before it's rejected with a "use `laya_classify_batch`"
  hint. Raising it trades a longer worst-case interactive latency for fewer
  round trips; lowering it protects interactive latency under load.
- `LAYA_MCP_MAX_WAITING_REQUESTS` — how many interactive calls queue behind
  the single inference lock before the server returns a retryable busy
  error instead of queuing indefinitely.
- `LAYA_MCP_JOB_WORKERS` — Huey worker threads for batch/evaluate jobs. Keep
  at 1: workers share the same single inference lock as interactive calls,
  so more threads add contention, not throughput, on this CPU-only host.

After editing `.env`, restart the task (see above) — settings are read once
at startup (`pydantic-settings`, cached via `lru_cache`).

## Memory notes

This host has 16 GB RAM with roughly 6.6 GB free once MySQL, Postgres,
httpd, Ollama and the other uvicorn app (:8000) are all running. Each fp32
checkpoint is roughly 1.7 GB resident.

- `LAYA_MCP_PRELOAD=english` keeps only the English checkpoint warm at
  startup — the right default while traffic is dev-team-only and
  overwhelmingly English. Preloading `multilingual` and `typed-decisions` too
  costs another ~3.4 GB combined; only do that once support/sales traffic
  actually needs them resident rather than lazily loaded.
- `LAYA_MCP_MAX_LOADED=2` caps how many checkpoints stay resident at once
  (LRU eviction) even if more get loaded lazily via `laya_load_models` or a
  language switch. Raise it to 3 only if you've confirmed the RAM headroom —
  check `laya_status`'s `rss_gb` after peak load first.
- `LAYA_MCP_QUANTIZE` (int8 dynamic quantization of Linear layers, CPU-only)
  is off by default. Only turn it on after `scripts/bench.py` has confirmed
  on a fixed probe set that it doesn't change any answer's argmax and moves
  no confidence by more than 0.02 — quantization that quietly shifts
  calibrated confidence would undermine every threshold in the schema
  library.

If the process's RSS creeps well past ~4 GB steady-state, check `laya_status`
for how many models are actually loaded before assuming a leak — the number
should track `LAYA_MCP_PRELOAD`/`LAYA_MCP_MAX_LOADED`, not grow unbounded
with traffic.

## If the IP changes

The host's Wi-Fi address is DHCP-assigned (`10.10.29.81` as of this writing,
sleep disabled on AC so it shouldn't lease-expire mid-session, but a reboot
or router change can still reassign it). If clients across the intranet
start failing to connect:

1. Confirm the current IP on the host: `ipconfig` (look at the Wi-Fi
   adapter), or `Get-NetIPAddress -AddressFamily IPv4`.
2. Update every client's MCP config (`claude mcp add ...` or `.mcp.json`,
   see [client-setup.md](client-setup.md)) to the new IP, or —
3. **Preferably, get a DHCP reservation or a stable hostname for this host**
   from whoever manages WoCo's network, so this stops being a recurring
   problem. Until that's in place, treat the IP in every doc in this repo
   (`10.10.29.81`) as "the last known address, verify before relying on it".
4. The firewall rule (see "Reaching it from other machines") is bound to the
   port, not the IP, so it doesn't need to change when the IP does — only
   `-RemoteAddress` (who's allowed in) would need updating if the intranet's
   subnet layout itself changes. `start-laya-mcp.bat` prints the current IP
   each time it starts the server.

## Smoke tests after a deploy or upgrade

With the server running:

```powershell
# full workflow over HTTP with the real model (~30 s): classify, schema, dataset, evaluate,
# calibrate, promote (refused: small dataset), decide, batch, scan, status, usage
venv\Scripts\python scripts\e2e_smoke.py --url http://127.0.0.1:8765/mcp --team e2e

# a LangChain agent (MCP SDK 1.x) can use the server; runs from its own venv (see langchain-client.md)
venv-clients\Scripts\python scripts\langchain_smoke.py --url http://127.0.0.1:8765/mcp --team dev
```

The e2e script writes a schema and dataset under team `e2e`; run it against a throwaway data
directory (`LAYA_MCP_DATA_DIR`) if you don't want them in the production database.
