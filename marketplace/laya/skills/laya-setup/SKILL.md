---
name: laya-setup
description: Install, connect, configure and troubleshoot the Laya MCP server for Claude Code - checking that the laya_* tools are available, that the server at http://10.10.29.81:8765 is reachable, setting the team (X-Laya-Team) and server URL, removing duplicate registrations, and starting the server on its host. Use this whenever the user wants to install or connect Laya, asks why the Laya tools are missing, disconnected or timing out, wants to switch their Laya team (dev, support, sales...), sees "connection refused" or "busy" errors from laya tools, or asks how to run or start the Laya server.
---

# Setting up and fixing the Laya connection

The **laya** plugin registers an MCP server named `laya` (streamable HTTP) from its
`.mcp.json`:

- URL: `${LAYA_MCP_URL}`, default `http://10.10.29.81:8765/mcp`
- Header `X-Laya-Team: ${LAYA_TEAM}`, default `dev` - which team's schemas and usage
  stats the calls belong to

The server runs on one Windows host on the intranet and is **started on demand**, so
"not reachable" often just means nobody has started it. Work through the checks below in
order and stop at the first one that explains the problem. Tell the user what you found
at each step; changing their settings or restarting things is their call.

## 1. Are the tools there?

If `laya_status` is among your tools, call it. A result means everything works; report
`team`, loaded models and the limits, and you're done. If the team is wrong, go to
step 4.

If the laya tools are missing entirely:

- Plugin installed? `claude plugin list` should show `laya@woco`. If not, install it
  (section "Installing" below).
- Installed but disabled? `claude plugin enable laya@woco`.
- MCP servers are loaded when a session starts: after installing or changing settings
  the user must restart Claude Code (or reconnect it in `/mcp`).

## 2. Is the server reachable?

```bash
curl -s http://10.10.29.81:8765/health
```

(Use the host from `LAYA_MCP_URL` if the user overrides it.) A JSON body with
`"status":"ok"` means the server is up - go to step 3. Otherwise:

- **Connection refused / timed out**: the server is most likely not running. It is
  started on the host (`D:\Laya_Sandbox`) by double-clicking
  `scripts\start-laya-mcp.bat`; ask whoever owns that machine to start it. If you are ON
  the host, you can offer to run the .bat for the user (it opens a server window and
  prints the URL when ready, ~30-60 s).
- **Server running but other machines can't reach it**: Windows Firewall on the host
  must allow inbound TCP 8765 (see `D:\Laya_Sandbox\docs\operations.md`, "Reaching it
  from other machines"). That's an administrator change on the host - point the user to
  it rather than making it yourself.
- **The host's IP changed** (it's DHCP): the .bat prints the current URL when it starts.
  Set `LAYA_MCP_URL` to it (step 4).

## 3. Reachable, but the tools still fail

- **Duplicate registration**: if someone also added the server by hand, there are two
  `laya` servers. `claude mcp list` - if a plain `laya` entry exists besides the plugin's,
  remove the manual one with `claude mcp remove laya` (ask first; add `-s user` or
  `-s project` to match where it was added).
- **Busy errors**: other callers are ahead in the single inference queue. Wait the
  suggested time or use batch jobs; nothing to fix.
- **Timeouts on big calls**: too many rows in one synchronous call - use
  `laya_classify_batch` (see the laya skill).

## 4. Setting the team or URL

The plugin reads two environment variables when Claude Code starts. The cleanest place is
the `env` block of the user's Claude Code settings (`~/.claude/settings.json`, or a
project's `.claude/settings.json` so a whole repo uses one team):

```json
{
  "env": {
    "LAYA_TEAM": "support",
    "LAYA_MCP_URL": "http://10.10.29.81:8765/mcp"
  }
}
```

Show the user the change and let them confirm before editing a settings file; merge into
any existing `env` block rather than replacing the file. A plain environment variable
(`setx LAYA_TEAM support` on Windows, `export LAYA_TEAM=support` in a shell profile)
works too. Restart Claude Code, then confirm with `laya_status` that `team` changed.

Teams in use: `dev` (default), `support`, `sales`. The value is lowercased; schemas are
namespaced by it (`support/ticket-intake`).

## Installing

The plugin comes from the `woco` marketplace, a shared folder on the Laya host
(`\\DESKTOP-QF8TM70\laya-plugins`, or wherever the host owner shared
`D:\Laya_Sandbox\marketplace`):

```bash
claude plugin marketplace add "\\DESKTOP-QF8TM70\laya-plugins"
claude plugin install laya@woco
```

(Or `/plugin marketplace add ...` and `/plugin install laya@woco` inside a session.) If the
UNC path isn't accepted, map the share to a drive letter first and add that path. The
share only needs to be reachable when installing or updating
(`claude plugin marketplace update woco` then `claude plugin update laya@woco`).

Without the plugin (tools only, no skills):

```bash
claude mcp add --transport http laya http://10.10.29.81:8765/mcp --header "X-Laya-Team: dev"
```

## Running the server (host owner)

On the host machine (`D:\Laya_Sandbox`):

- Start: `scripts\start-laya-mcp.bat` - opens a "Laya MCP Server :8765" window, waits for
  health, prints the intranet URL. Stop: close that window (queued batch work resumes on
  the next start).
- Health: `http://localhost:8765/health`; logs in `D:\Laya_Sandbox\data\logs`.
- Only the english checkpoint is downloaded by default; `scripts\prefetch_models.py`
  fetches the multilingual and typed-decisions ones (~1.6 GB, needs internet once).
- Full operations guide: `D:\Laya_Sandbox\docs\operations.md`.
