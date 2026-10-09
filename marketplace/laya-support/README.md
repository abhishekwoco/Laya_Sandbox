# laya-support

The WoCo support team's daily toolkit. After installing, you drive everything with
**three one-word commands** typed into Claude Code — no prompt writing:

| You type | What happens |
|---|---|
| `triage` | Reads the support channel since the last digest, classifies every issue with Laya (bug vs config, urgency, resolved, escalation, channel rules), finds duplicates, and shows you a digest **draft**. You approve, it posts. |
| `chase` | Finds unresolved issues that have gone quiet for 24h+ where the ball is with dev, and drafts polite in-thread reminders. You approve each one. |
| `labels` | Collects the corrections your team replied under past digests ("2 = config", "3 resolved") and saves them as training examples, so Laya keeps getting better at your tickets. |

Default channel is **#errors-bugs**; say `triage #some-channel` to point elsewhere.
Once a day, the first Claude session you open will also offer to run the digest —
say no and it won't ask again until tomorrow.

Nothing is ever posted to Slack without showing you the exact message and getting
your yes.

## Install (one time, ~2 minutes)

Requirements: Claude Code desktop app, the **Slack connector** enabled in your
Claude settings, and being on the WoCo office network (the Laya server is
intranet-only). Windows only for now (the hooks are PowerShell).

```bash
claude plugin marketplace add abhishekwoco/Laya_Sandbox
claude plugin install laya-support@woco
```

Restart Claude Code. Type `triage` to test.

Notes:

- Don't also install the `laya` plugin — you'd register the Laya server twice.
  This plugin already connects you with team `support`.
- The repo is private: you need GitHub read access and working git credentials
  (`git clone https://github.com/abhishekwoco/Laya_Sandbox` must not prompt).
- If Laya tools fail ("cannot connect"), the server may be off — ask the host
  owner to start it (`scripts\start-laya-mcp.bat` on the Laya machine).

## What's inside

```
hooks/hooks.json     the two hooks (no other automation runs on your machine)
hooks/expand.ps1     turns triage/chase/labels into the full recipe for Claude
hooks/daily.ps1      once-a-day digest offer on your first session (stamp files
                     in ~/.laya-support/)
recipes/*.md         the actual instructions Claude follows — readable, editable
.mcp.json            the Laya server connection (team: support)
```

Local files the recipes create on your machine: `~/.laya-support/runs/*.json`
(each day's triage items + answers; the labels sweep needs them) and
`~/.laya-support/dataset-log.json` (which examples were already saved).

## For maintainers

- The recipes start with ad-hoc `laya_classify` questions. Once the support
  dataset reaches 50+ labeled examples, run the schema lifecycle (save →
  evaluate → calibrate → promote `support/ticket-intake`); the triage recipe
  automatically switches to `laya_decide` when it finds the trusted schema.
- Update flow: bump `version` in `.claude-plugin/plugin.json` and the two
  marketplace.json files, push; members get it via
  `claude plugin marketplace update woco` + `claude plugin update laya-support@woco`.
