# WoCo plugin marketplace (`woco`)

Contains one Claude Code plugin, **laya**:

| Part | What it does |
|---|---|
| `.mcp.json` | Registers the Laya MCP server (`laya`, streamable HTTP) with the team header |
| `skills/laya` | Teaches Claude when and how to hand decisions to Laya (classify, decide, batch, scan, schemas) |
| `skills/laya-setup` | Installs, connects, switches team and troubleshoots the connection |

The marketplace is served from the private GitHub repo **`abhishekwoco/Laya_Sandbox`**:
`/.claude-plugin/marketplace.json` at the repo root lists the plugin in
`marketplace/laya`. (`marketplace/.claude-plugin/marketplace.json` is the same catalog
for installing from a local copy or network share.)

## Teammates: install

Prerequisites (the repo is private):

1. Read access to `abhishekwoco/Laya_Sandbox` (the repo owner adds you as a collaborator,
   or the repo moves to a WoCo GitHub organization you belong to).
2. Git can reach it without prompting: `git clone https://github.com/abhishekwoco/Laya_Sandbox`
   works in a terminal. If not, sign in once via Git Credential Manager (bundled with Git
   for Windows) or `gh auth login` + `gh auth setup-git`.

Then:

```bash
claude plugin marketplace add abhishekwoco/Laya_Sandbox
claude plugin install laya@woco
```

Inside a Claude Code session the same works as `/plugin marketplace add ...` and
`/plugin install laya@woco`. If the add tries SSH and fails, set
`CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1` and retry.

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

## Offering it to everyone who opens a repo

Add to that repo's `.claude/settings.json`; teammates who trust the folder are prompted
to install the marketplace and plugin:

```json
{
  "extraKnownMarketplaces": {
    "woco": { "source": { "source": "github", "repo": "abhishekwoco/Laya_Sandbox" } }
  },
  "enabledPlugins": { "laya@woco": true }
}
```

On a Claude Team/Enterprise plan, an org admin can set the same two keys once for
everyone under claude.ai **Organization settings > Claude Code > Managed settings**.

## Updating

Edit the plugin, bump `version` in `laya/.claude-plugin/plugin.json` and in both
marketplace catalogs (`/.claude-plugin/marketplace.json` and
`marketplace/.claude-plugin/marketplace.json`), validate, commit and push:

```bash
claude plugin validate D:\Laya_Sandbox
claude plugin validate D:\Laya_Sandbox\marketplace\laya
```

Teammates then run:

```bash
claude plugin marketplace update woco
claude plugin update laya@woco
```

## Why not Anthropic's public marketplace

The official Anthropic marketplace doesn't take public submissions, and the community
directory requires a public repo. This plugin only works inside WoCo's network (its MCP
server is an unauthenticated intranet host), and making the repo public would publish the
host's address and internal details. Keep it private.
