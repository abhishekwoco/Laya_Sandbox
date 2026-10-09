# laya-support

The WoCo support team's daily toolkit. After installing, you drive everything with
**three slash commands** typed into Claude Code — no prompt writing:

| Command | What happens |
|---|---|
| `/triage` | Reads the support channel since the last digest, classifies every issue with Laya (bug vs config, urgency, resolved, escalation, channel rules), finds duplicates, and shows you a digest **draft**. You approve, it posts. |
| `/chase` | Finds unresolved issues silent for 24h+ where the ball is with dev, and drafts polite in-thread reminders. You approve each one. |
| `/labels` | Collects the corrections your team replied under past digests ("2 = config", "3 resolved") and turns them into Laya training examples, so Laya keeps getting better at your tickets. |

Default channel is **#errors-bugs**; pass a channel to point elsewhere:
`/triage #support-helpdesk`. Once a day, the first Claude session you open also
offers to run the digest — say no and it won't ask again until tomorrow.

These commands use the Laya server on the WoCo intranet, so they work in Claude
Code / Claude Cowork on the office network only. Nothing is ever posted to Slack
without showing you the exact message and getting your yes.

## Install (one time, ~2 minutes)

Requirements: Claude Code desktop app, the **Slack connector** enabled in your
Claude settings, and being on the WoCo office network.

```bash
claude plugin marketplace add abhishekwoco/Laya_Sandbox
claude plugin install laya-support@woco
```

Restart Claude Code. Type `/triage` to test.

Notes:

- Don't also install the `laya` plugin — you'd register the Laya server twice.
  This plugin already connects you with team `support`.
- The repo is private: you need GitHub read access and working git credentials
  (`git clone https://github.com/abhishekwoco/Laya_Sandbox` must not prompt).
- If Laya tools fail ("cannot connect"), the server may be off — ask the host
  owner to start it (`scripts\start-laya-mcp.bat` on the Laya machine).

## What's inside

```
skills/triage/SKILL.md   the /triage workflow — readable, editable markdown
skills/chase/SKILL.md    the /chase workflow
skills/labels/SKILL.md   the /labels workflow
hooks/hooks.json         one hook: the daily nudge (no other automation runs)
hooks/daily.ps1          once-a-day /triage offer on your first session
                         (stamp files in ~/.laya-support/; Windows/PowerShell)
.mcp.json                the Laya server connection (team: support)
```

Local files the skills create on your machine: `~/.laya-support/runs/*.json`
(each day's triage items + answers; the /labels sweep needs them) and
`~/.laya-support/dataset-log.json` (which examples were already saved).

## For maintainers

- The skills start with ad-hoc `laya_classify` questions. Once the support
  dataset reaches 50+ labeled examples, run the schema lifecycle (save →
  evaluate → calibrate → promote `support/ticket-intake`); the /triage skill
  automatically switches to `laya_decide` when it finds the trusted schema.
- Update flow: bump `version` in `.claude-plugin/plugin.json` and the two
  marketplace.json files, push; members get it via
  `claude plugin marketplace update woco` + `claude plugin update laya-support@woco`.
