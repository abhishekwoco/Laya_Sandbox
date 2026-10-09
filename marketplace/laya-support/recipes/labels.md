# Recipe: labels sweep (grow the support dataset from digest corrections)

Turn the team's replies on past triage digests into labeled examples for the
`ticket-intake` dataset, so Laya's support schema can be evaluated, calibrated and
promoted. Nothing is saved without showing the user the final example list.

## 1. Gather corrections

- Channel: named in the user's message, else **#errors-bugs**.
- Find `Laya triage —` digest messages from the last 14 days (`slack_read_channel`,
  look for the marker text).
- For each digest, `slack_read_thread` and parse correction replies. Expected
  shapes (be tolerant — interpret free text too):
  - `2 = config` / `2 is config` → item 2's category = config_or_user
  - `3 resolved` / `3 is fixed` → item 3's resolved = true
  - `4 not a bug, question` → item 4's category = question
  - `all good` / `rest ok` from a support member → every uncorrected item's
    applied answers are CONFIRMED labels.
- Match item numbers to the digest's numbered list, then join to the stored run
  file `~/.laya-support/runs/<digest date>.json` to recover each item's condensed
  `state` and the applied answers. A digest with no run file can only contribute
  items whose full state you can reconstruct from the digest itself — usually
  none; skip it and say so.

## 2. Build the examples

For every corrected item, and for confirmed items when a "rest ok"-style reply
exists:

```json
{"state": {"issue": "...", "discussion": "..."},
 "expected": {"category": "config_or_user", "urgency": 1,
              "resolved": true, "escalation": false, "valid_post": true}}
```

`expected` must be complete per example (fill uncorrected answers from the run
file's applied answers). Skip items already saved in a previous sweep: keep a
`saved_ids` list in `~/.laya-support/dataset-log.json` and append to it after
each save.

## 3. Review and save

- Show the user a compact table: item, source digest date, each expected answer,
  and whether it came from a correction or a confirmation.
- On approval: `laya_save_dataset` with name `ticket-intake`, `append: true`.
- Report the dataset's new total. When it crosses **50 examples**, tell the user
  the schema lifecycle can start and offer to run it:
  `laya_save_schema` (name `ticket-intake`, the five triage questions, targets
  `{"min_accuracy": 0.85, "min_examples": 50}`) → `laya_evaluate` →
  `laya_get_report` → `laya_calibrate` → `laya_promote_schema`. Never promote
  automatically — show the report and let the user decide.
