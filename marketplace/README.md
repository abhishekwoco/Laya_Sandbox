# WoCo plugin marketplace (`woco`)

Contains one Claude Code plugin, **laya**:

| Part | What it does |
|---|---|
| `.mcp.json` | Registers the Laya MCP server (`laya`, streamable HTTP) with the team header |
| `skills/laya` | Teaches Claude when and how to hand decisions to Laya (classify, decide, batch, scan, schemas) |
| `skills/laya-setup` | Installs, connects, switches team and troubleshoots the connection |

## Host: share this folder (once)

Share `D:\Laya_Sandbox\marketplace` read-only as `laya-plugins`. In Explorer:
right-click the folder -> Properties -> Sharing -> Advanced Sharing -> Share this folder,
name `laya-plugins`, Permissions: Everyone = Read. Or, in an elevated PowerShell:

```powershell
New-SmbShare -Name laya-plugins -Path D:\Laya_Sandbox\marketplace -ReadAccess Everyone
```

Teammates then reach it as `\\DESKTOP-QF8TM70\laya-plugins` (the host's computer name).
The share is only needed while installing or updating; the Laya server itself is
separate and started with `D:\Laya_Sandbox\scripts\start-laya-mcp.bat`.

## Teammates: install

```bash
claude plugin marketplace add "\\DESKTOP-QF8TM70\laya-plugins"
claude plugin install laya@woco
```

Inside a Claude Code session the same works as `/plugin marketplace add ...` and
`/plugin install laya@woco`. If the UNC path is rejected, map the share to a drive
(`net use L: \\DESKTOP-QF8TM70\laya-plugins`) and add `L:\` instead.

Restart Claude Code, then ask it "check my Laya connection" - the laya-setup skill runs
`laya_status` and reports the team and server state.

### Team and server address

Defaults: team `dev`, server `http://10.10.29.81:8765/mcp`. Override in
`~/.claude/settings.json` (or a repo's `.claude/settings.json`), then restart:

```json
{ "env": { "LAYA_TEAM": "support", "LAYA_MCP_URL": "http://10.10.29.81:8765/mcp" } }
```

If you previously added the server by hand (`claude mcp add ... laya`), remove that entry
(`claude mcp remove laya`) so the tools aren't registered twice.

## Updating

Edit the plugin here, bump `version` in `laya/.claude-plugin/plugin.json` and in
`.claude-plugin/marketplace.json`, then teammates run:

```bash
claude plugin marketplace update woco
claude plugin update laya@woco
```

Validate before sharing: `claude plugin validate D:\Laya_Sandbox\marketplace`.
