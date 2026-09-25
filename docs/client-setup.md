# Client setup

laya-mcp is a stateless streamable-HTTP MCP server at:

```
http://10.10.29.81:8765/mcp
```

`10.10.29.81` is the host's current Wi-Fi IP (DHCP). If it stops responding,
check whether the IP changed — see [Troubleshooting](#troubleshooting) and
[operations.md](operations.md#if-the-ip-changes). Ask whoever runs the host
for a DNS name or a DHCP reservation if you want a stable address instead of
the raw IP.

There is no authentication (intranet-only by design). Every client must send
an `X-Laya-Team` header — it's how the server attributes schema usage and
namespaces (`team/name`) to your team. Pick one of: `dev`, `support`,
`sales`, or ask what's already in use.

## Claude Code

### Recommended: install the `laya` plugin

The plugin registers the server *and* adds two skills: `laya` (when and how to hand
decisions to Laya) and `laya-setup` (connect, switch team, troubleshoot). It is served
from the `woco` marketplace, a shared folder on the Laya host:

```bash
claude plugin marketplace add "\\DESKTOP-QF8TM70\laya-plugins"
claude plugin install laya@woco
```

Team and server address come from `LAYA_TEAM` (default `dev`) and `LAYA_MCP_URL`
(default `http://10.10.29.81:8765/mcp`), e.g. in `~/.claude/settings.json`:
`{"env": {"LAYA_TEAM": "support"}}`. Restart Claude Code afterwards. Details:
[marketplace/README.md](../marketplace/README.md). If you use the plugin, don't also add
the server by hand (below) - you'd get every tool twice.

### Quick add without the plugin (user scope — available in every project you open)

```bash
claude mcp add --transport http laya http://10.10.29.81:8765/mcp --header "X-Laya-Team: dev"
```

### Project scope (checked into the repo, shared with your team)

Add to `.mcp.json` at the root of your project:

```json
{
  "mcpServers": {
    "laya": {
      "transport": "http",
      "url": "http://10.10.29.81:8765/mcp",
      "headers": {
        "X-Laya-Team": "dev"
      }
    }
  }
}
```

Use a scope that fits how the team works:

| Scope | Command flag | When to use |
|---|---|---|
| User | `claude mcp add --transport http laya ... -s user` (default) | Personal, available in every project on your machine |
| Project | `-s project` (writes `.mcp.json`) | Whole team should get it when they open this repo |
| Local | `-s local` | Just this one project, just for you (not committed) |

Change the `X-Laya-Team` value per project if different repos belong to
different teams (e.g. a support-tooling repo should send `X-Laya-Team:
support`).

### Verify it's connected

Inside a Claude Code session:

```
/mcp
```

`laya` should show as connected, with its tool list (`laya_classify`,
`laya_decide`, `laya_apply_preset`, ... — see [tools.md](tools.md) for the
full set). If it shows an error, check
[Troubleshooting](#troubleshooting) below.

From a shell, independent of Claude Code:

```bash
curl http://10.10.29.81:8765/health
```

A healthy server returns 200 with a small JSON body (loaded models, device).
If you get a JSON-RPC error trying to call a tool but `/health` is fine, the
issue is in your MCP client config, not the server.

You can also point the standalone
[MCP Inspector](https://github.com/modelcontextprotocol/inspector) at
`http://10.10.29.81:8765/mcp` (streamable HTTP transport) to browse tools,
resources and prompts interactively without writing any client code.

## Other MCP-compatible agents

Any client that speaks MCP over streamable HTTP can connect the same way.
The shape is usually close to this generic config (adjust to your client's
schema):

```json
{
  "mcpServers": {
    "laya": {
      "url": "http://10.10.29.81:8765/mcp",
      "transport": "streamable-http",
      "headers": {
        "X-Laya-Team": "dev"
      }
    }
  }
}
```

For a LangChain/LangGraph agent specifically, see
[langchain-client.md](langchain-client.md) — it uses
`langchain-mcp-adapters` rather than a client config file.

## Troubleshooting

**Can't connect / connection refused**
- Confirm the server is actually running: `curl http://10.10.29.81:8765/health`
  from another intranet machine. The server is started on demand (not at boot),
  so it may simply be off: ask the host owner to run `scripts\start-laya-mcp.bat`.
- Check the Windows Firewall on the host lets your machine in (see "Reaching it
  from other machines" in [operations.md](operations.md)). A rule limited to
  `LocalSubnet` only admits the host's own subnet, which may be narrower than
  the whole WoCo intranet; ask the host owner to widen it (e.g. `10.10.0.0/16`)
  if your machine is on a different subnet.
- The host's IP is DHCP-assigned (`10.10.29.81` at time of writing) and can
  change on lease renewal or reboot. If it suddenly stops resolving, ask the
  host owner to confirm the current IP, or push for the DHCP
  reservation/hostname noted in [operations.md](operations.md).

**Tool call hangs or times out**
- Each row (one question on one item) costs roughly 0.5-1.6 s of CPU (longer input costs more)
  inference on this host, and only one forward pass runs at a time. A
  request with many items x questions can legitimately take tens of seconds.
  Prefer `laya_classify_batch` for anything beyond a handful of items — it
  runs asynchronously and you poll `laya_job_status`/`laya_job_results`
  instead of holding a synchronous call open.
- The server enforces a request timeout (`LAYA_MCP_REQUEST_TIMEOUT_S`,
  default 120s) and a sync row budget (`LAYA_MCP_SYNC_ROW_BUDGET`, default
  20 = items x questions). Going over the row budget fails fast with a
  message telling you to use `laya_classify_batch` instead — that's
  expected, not a bug.

**"busy" / 503-style errors**
- Interactive calls queue behind a single inference lock
  (`LAYA_MCP_MAX_WAITING_REQUESTS`, default 4 waiting). If you get a busy
  error, it names a retry-after estimate — back off and retry, or switch to
  `laya_classify_batch`. This is more likely while a large batch or
  evaluation job is running, since batch work yields to interactive calls
  but still occupies some CPU between yields.

**Answers come back `status: "unverified"`**
- That's not an error: it means the schema you called through
  `laya_decide` isn't `trusted` yet (still a draft, or evaluated without
  meeting its accuracy targets / not promoted). Only trusted schemas return
  `decided`. For an unverified schema, `needs_review` still lists the least
  certain answers (using measured thresholds once evaluated) — treat the rest
  as suggestions, not decisions. See
  [dev-playbook.md](dev-playbook.md#schema-lifecycle) for how to move a
  schema from `draft` to `evaluated` to `trusted`.

**Wrong team's schemas / usage showing up**
- Double check the `X-Laya-Team` header value in your client config — it's
  case-insensitive but must otherwise match exactly what your team's
  schemas were saved under. `laya_list_schemas` (no team filter) shows every
  team's schemas if you need to check what's actually there.
