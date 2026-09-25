# laya-mcp

An MCP server that lets Claude Code (and any other MCP-compatible agent) hand
structured decisions to [Laya](https://github.com/NandhaKishorM/laya), a small,
non-autoregressive classifier — the way you'd delegate to a subagent. Claude
sends content plus typed questions (`choice` / `score` / `noul` yes-no), Laya
answers all of them in one forward pass with calibrated confidence, and Claude
only spends reasoning on the answers marked `needs_review`.

Runs as one long-lived process on a WoCo intranet host, served over
**streamable HTTP** (MCP) at `http://<host>:8765/mcp`. No authentication —
this is an intranet-only server behind the firewall; team identity comes from
an `X-Laya-Team` header set once in each client's MCP config. First users are
the dev team; support and sales follow later with their own schemas.

## Why hand decisions to Laya instead of just reasoning about them?

Laya is fast (roughly 0.5-1.6 s per question on this CPU-only host; longer input costs more) and cheap
compared to an LLM call, and its confidence is calibrated per schema — so
Claude can batch-classify, triage, or filter without spending its own context
and reasoning budget on decisions that are usually easy. Claude stays in the
loop for the ones Laya flags `needs_review`. See
[docs/dev-playbook.md](docs/dev-playbook.md) for when to delegate and when not
to, and [docs/claude-md-snippet.md](docs/claude-md-snippet.md) for a block to
paste into a project's `CLAUDE.md`.

## Architecture

```
                                 WoCo intranet (10.10.0.0/16)
                                          |
   Claude Code / other MCP agents ───HTTP───▶  :8765  FastAPI app  (this host)
   (X-Laya-Team: <team> header)                 │
                                                 ├─ /mcp           FastMCP 4 http_app
                                                 │                 (stateless streamable HTTP)
                                                 ├─ /v1/systemone  Jev-compatible REST (laya.serve contract, via the engine)
                                                 ├─ /health
                                                 └─ /metrics       Prometheus
                                                 │
                              FastMCP middleware: structlog logging
                                    -> timing/usage -> error mapping
                                    -> response-size limit -> team header
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
                    │                    InferenceEngine                       │
                    │              (shared laya.Router, one process)          │
                    │                                                          │
                    │   interactive calls            batch/eval jobs          │
                    │   asyncio semaphore(1)          Huey consumer threads,   │
                    │   + anyio.to_thread             yielding to interactive  │
                    │   (torch threads = LAYA_MCP_THREADS)  calls              │
                    │                                                          │
                    │   row budget · per-request timeout · post-hoc           │
                    │   temperature calibration                               │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
                    │  data/laya_mcp.db  (SQLModel/SQLite, Alembic-migrated)   │
                    │    schemas · schema versions · datasets · reports ·      │
                    │    calibrations · usage events                          │
                    │  data/jobs.db      (SqliteHuey job queue + results)      │
                    └──────────────────────────────────────────────────────────┘
```

A single inference worker is deliberate: on an 8-core CPU, parallel torch
calls fight for the same cores, so parallelism comes from batching many
questions/items into one forward pass, not from running passes concurrently.
See the plan referenced in `docs/operations.md` for the full design rationale.

## Quickstart

Run these from the project root (`D:\Laya_Sandbox`) using the project's own
virtualenv, `venv\Scripts\python.exe` (do not rely on whatever `python` is on
`PATH` — this host also runs other Python-based services).

```powershell
# 1. Install the project (editable) into the venv
venv\Scripts\python.exe -m pip install -e ".[dev]"

# 2. Pull the checkpoints not already cached (only `english` is cached today).
#    Do this BEFORE setting HF_HUB_OFFLINE=1, since it needs network access once.
venv\Scripts\python.exe scripts\prefetch_models.py

# 3. Copy the example environment file and adjust if needed
copy .env.example .env

# 4. Run the server (foreground, for a first check)
venv\Scripts\laya-mcp.exe
# or: venv\Scripts\python.exe -m laya_mcp.app

# 5. From another shell (or another machine on the intranet), confirm it's up
curl http://localhost:8765/health
```

Day to day, start it when needed with `scripts\start-laya-mcp.bat` (opens the server in
its own window, waits until it's healthy, prints the intranet URL); close that window to
stop it. See [docs/operations.md](docs/operations.md), including the one-time firewall
step so other machines can reach it. Running it as a boot-time service
(`scripts/install_task.ps1`, Administrator) is optional and not used on this host.

### Add it to Claude Code

Recommended: the `laya` plugin (MCP server + usage and setup skills), from the shared
`marketplace/` folder:

```
claude plugin marketplace add "\\DESKTOP-QF8TM70\laya-plugins"
claude plugin install laya@woco
```

Tools only, without the skills:

```
claude mcp add --transport http laya http://10.10.29.81:8765/mcp --header "X-Laya-Team: dev"
```

Full instructions, project-scoped config, other MCP clients, and
troubleshooting: [docs/client-setup.md](docs/client-setup.md).

## Seeding the dev team's schemas

Once the server is running, load the four ready-made dev schemas
(`dev/issue-triage`, `dev/log-classify`, `dev/source-relevance`,
`dev/task-tagging`) into the schema library:

```powershell
venv\Scripts\python.exe scripts\seed_schemas.py
```

Each starts as an unevaluated `draft` — see
[docs/dev-playbook.md](docs/dev-playbook.md) for the evaluate → calibrate →
promote workflow before treating their decisions as anything but
`unverified`.

## Documentation

| Doc | For |
|---|---|
| [docs/client-setup.md](docs/client-setup.md) | Connecting Claude Code or another MCP agent to this server |
| [docs/dev-playbook.md](docs/dev-playbook.md) | The dev team: when to delegate to Laya, how to write good questions, the schema lifecycle |
| [docs/claude-md-snippet.md](docs/claude-md-snippet.md) | A block to paste into a project's `CLAUDE.md` |
| [docs/langchain-client.md](docs/langchain-client.md) | Calling this server from LangChain/LangGraph agents |
| [docs/tools.md](docs/tools.md) | Reference for every MCP tool, resource, and prompt |
| [docs/operations.md](docs/operations.md) | Start/stop/restart, logs, backups, updates, host tuning |

## Project layout

```
src/laya_mcp/
  config.py        pydantic-settings, env prefix LAYA_MCP_ (see .env.example)
  app.py           FastAPI app; mounts /mcp, /v1/systemone, /health, /metrics
  server.py        FastMCP("laya") instance + middleware stack
  models/          Pydantic I/O contracts (questions, answers, jobs, schemas, reports)
  engine/          InferenceEngine around laya.Router: budgets, calibration, chunking
  db/              SQLModel tables + repository
  jobs/            SqliteHuey queue + batch/evaluate tasks
  tools/           FastMCP sub-servers, one per tool group (decide/batch/guard/workbench/library/ops)
tests/             pytest + FastMCP in-memory client + langchain-mcp-adapters smoke test
scripts/           start-laya-mcp.bat, prefetch_models.py, seed_schemas.py, e2e_smoke.py,
                   langchain_smoke.py, bench.py, install_task.ps1 (optional service)
examples/schemas/  ready-to-seed schema JSON, by team (examples/schemas/dev/*.json)
docs/              everything under Documentation above
```

`laya` itself (the model/library) lives separately at `D:\laya`; this project
only depends on it (`laya==0.3.7`, pinned).
