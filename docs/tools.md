# Tool reference

> **This document is generated from the build plan and the Pydantic I/O
> models in `src/laya_mcp/models/`, written before the tool implementations
> landed.** Field names for the "decisions", "schema library" and "ops" tools
> come directly from those models and should be accurate. Tools whose request
> shape isn't pinned down by a model yet (`laya_scan_untrusted`,
> `laya_compare_models`, `laya_sample_for_labeling`, `laya_load_models`,
> `laya_unload_models`, `laya_detect_language`) are described by intent only.
> **Verify every entry against a live `list_tools` call (or `/mcp` in Claude
> Code, or MCP Inspector) after integration, and fix any drift here.**

All tools are namespaced `laya_*` and return `model`, `latency_ms`, and
`rows` somewhere in their response (usually inherited from `ClassifyResponse`)
so a caller can always see what answered and how much it cost. Every tool
accepts an optional `team` argument that overrides the `X-Laya-Team` header
for that one call.

Two response detail levels apply wherever an `Answer` is returned:
- `detail="compact"` (default): value + confidence + status only.
- `detail="full"`: adds `probabilities` (and `legend` for `score`).

## Decisions

### `laya_classify`
Run typed questions over one or more items in a single synchronous call.

- **Input**: `state` (single item) **or** `items` (list) — one of the two;
  `questions` (`{id: {type, instructions, criteria}}`, see
  [question types](#question-format)); optional `model`
  (`english`/`multilingual`/`typed-decisions`, else auto-routed), `lang`,
  `detail`.
- **Output**: `ClassifyResponse` — `results: [ItemResult]` (each with
  `answers`, `model`, `route_reason`, `needs_review: [question_id]`), `rows`,
  `latency_ms`, `warnings`.
- **Limits**: fails fast with a "use `laya_classify_batch`" hint if
  `items x questions` exceeds `LAYA_MCP_SYNC_ROW_BUDGET` (default 20), or a
  state exceeds `LAYA_MCP_MAX_STATE_CHARS`.
- **Example**:
  ```json
  {
    "state": {"body": "Billed twice for March, refund today or we cancel."},
    "questions": {
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["low", "medium", "high"]},
      "wants_refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}
    }
  }
  ```

### `laya_decide`
Like `laya_classify`, but against a **saved schema** so thresholds and
calibration are applied consistently across calls.

- **Input**: `schema` (`"team/name"` or `"team/name@version"`), `state` or
  `items`, `detail`.
- **Output**: `DecideResponse` (extends `ClassifyResponse`) + `schema_ref`,
  `schema_status` (`draft`/`evaluated`/`trusted`), `decided` (count at/above
  threshold), `needs_review` (count).
- **Key behavior**: if the schema hasn't been evaluated yet, every answer's
  `status` is `unverified` regardless of confidence — see
  [dev-playbook.md](dev-playbook.md#schema-lifecycle). Only an `evaluated` or
  `trusted` schema gets real `decided`/`needs_review` splits, using the
  schema's stored per-question thresholds (falling back to
  `LAYA_MCP_DEFAULT_THRESHOLD`) and calibration temperatures.
- **Example**: `{"schema": "dev/issue-triage", "state": {"title": "...", "body": "..."}}`

### `laya_apply_preset`
Run one of Laya's built-in question presets (`laya/presets.py`) without
defining your own questions.

- **Input**: `preset` (`triage` | `email` | `guard` | `moderation` |
  `router`), `state`, optional `categories` (email preset only — overrides
  its default department list).
- **Output**: `ClassifyResponse`.
- Use this to try Laya on real data before investing in a custom schema; the
  `triage`/`email`/`guard`/`moderation` presets map to the same content Laya
  ships for support tickets, inbound email, LLM-input guardrails, and content
  moderation respectively; `router` scores task difficulty/domain for
  model-routing decisions.

### `laya_classify_batch`
Submit up to 500 items per call for asynchronous processing; call again with
the same `job_id` to append more (e.g. paging through a larger dataset from
the client side).

- **Input**: `items` (≤500), `questions` **or** `schema` (mutually
  exclusive — a schema also applies its thresholds/calibration), optional
  `model`, `lang`, `job_id` (append to an open job), `finalize` (default
  `true`; set `false` while still appending chunks).
- **Output**: `JobInfo` — `job_id`, `status`, `total`, `done`, `eta_seconds`
  (from the engine's rolling ms/row estimate), etc.
- Batch work shares the same single inference lock as interactive calls but
  yields to them, so a running batch job won't starve `laya_classify`/`laya_decide`.

### `laya_job_status`
- **Input**: `job_id`.
- **Output**: `JobInfo` (progress, ETA, error if failed).

### `laya_job_results`
Paginated results for a batch or evaluate job.

- **Input**: `job_id`, `offset` (default 0), `limit` (default 50),
  `needs_review_only`, `question_id` + `value` (filter to one question's
  answer, e.g. `question_id="category", value="billing"`), `detail`.
- **Output**: `JobResultsPage` — `job`, `items: [ItemResult]`,
  `next_offset` (`null` when done), `summary` (question id -> answer value ->
  count, across the **whole** job, not just this page).

### `laya_job_cancel`
- **Input**: `job_id`.
- **Output**: `JobInfo` with `status: "cancelled"`. Already-completed items
  keep their results (retrievable via `laya_job_results`); un-started items
  are dropped.

## Guardrails

### `laya_scan_untrusted`
Advisory scan of untrusted text (e.g. a fetched web page or pasted document)
for prompt-injection / jailbreak / sensitive-data content before an agent
acts on it, using `laya/presets.py`'s `guard_questions()` over token-aware
chunks of the input (long text is split so each chunk fits the model's
context window; see `engine/chunking.py`).

- **Input** (expected): `text`, optional `model`.
- **Output** (expected): an overall verdict plus a list of flagged chunks
  with their span and scores — e.g. `{"flagged": true, "chunks": [{"start":
  0, "end": 512, "answers": {...}}], ...}`. **This is advisory only**: treat
  a flag as a reason to look closer or escalate, not as a hard block, and
  confirm the exact shape against `list_tools` (this tool's request/response
  models weren't finalized when this doc was written).

## Schema library

Schemas are namespaced `team/name`, versioned (every save creates a new
version), and move through `draft -> evaluated -> trusted`. See
[dev-playbook.md](dev-playbook.md#schema-lifecycle) for the full lifecycle
and four ready-made dev schemas.

### `laya_save_schema`
- **Input**: `team`, `name`, `questions`, optional `description`, `targets`
  (`min_accuracy` default 0.9, `min_examples` default 50, optional
  `per_question` overrides).
- **Output**: `SchemaInfo` for the new version (`status: "draft"`).
- Always creates a **new version**; never overwrites or deletes an existing
  one. `scripts/seed_schemas.py` calls this for every file under
  `examples/schemas/**/*.json`.

### `laya_get_schema`
- **Input**: `ref` (`"team/name"` = latest, or `"team/name@N"`).
- **Output**: `SchemaInfo` (questions, targets, thresholds, temperatures,
  `latest_report_id`).

### `laya_list_schemas`
- **Input**: optional `team` filter.
- **Output**: `[SchemaSummary]` (ref, status, description, question ids,
  version count, `updated_at`).

### `laya_promote_schema`
Marks a schema version `trusted`. **Refuses** unless its latest evaluation
report (`laya_evaluate`) meets `targets.min_accuracy` (overall and any
`per_question` override) on at least `targets.min_examples` labeled
examples.

- **Input**: `ref`.
- **Output**: `SchemaInfo` with `status: "trusted"`, or an error explaining
  which target(s) weren't met.

## Reports

### `laya_get_report`
- **Input**: `report_id` (an int returned by `laya_evaluate` /
  `laya_calibrate`, or a schema's `latest_report_id`).
- **Output**: `EvalReport` — per-question accuracy/confusion
  matrix/precision-recall/MAE/ECE, confidence-band breakdown, recommended
  threshold + coverage, worst confident misses, overall accuracy,
  `passes_targets`. Also readable as the `laya://reports/{id}` resource.

## Workbench

### `laya_validate_questions`
Structural + budget check with **no inference run** — use this before
`laya_save_schema` to catch mistakes for free.

- **Input**: `questions`, optional `model` (to check against that
  checkpoint's real tokenizer/`head_max_len`; without it only structural
  checks run).
- **Output**: errors (reusing `Agent._check_question`'s wording verbatim —
  it already names the offending question and the fix) and warnings (too
  many choice labels, near-duplicate label descriptions, missing
  descriptions, instructions likely to blow the token budget).

### `laya_save_dataset`
Store labeled examples for evaluation, keyed by `team/name`.

- **Input**: `team`, `name`, `examples: [{state, expected: {question_id:
  answer}}]`, `append` (default `true`; `false` replaces the dataset).
  `expected` values: a choice label (string), a score level index (int), or
  a noul boolean.
- **Output**: `DatasetInfo` (count, question ids covered, timestamps).
- Send large datasets in chunks with `append=true` calls — there's no
  separate "append vs create" tool, the flag does both.

### `laya_list_datasets`
- **Input**: optional `team` filter.
- **Output**: `[DatasetInfo]`.

### `laya_evaluate`
Runs a schema against a labeled dataset as an async job (large datasets cost
real CPU time at ~1s/row).

- **Input**: `schema` (ref), `dataset` (`"team/name"`), optional
  `target_accuracy` (defaults to the schema's own targets, then
  `LAYA_MCP_DEFAULT_TARGET_ACCURACY`), optional `compare_models` (also run
  the other two checkpoints for comparison).
- **Output**: `JobInfo` (`kind: "evaluate"`); once `status: "completed"`,
  `result_ref.report_id` points to the `EvalReport` — fetch it with
  `laya_get_report` or poll `laya_job_results`.

### `laya_calibrate`
Fits one temperature per question on a held-out split of the dataset and
stores it on the schema version (shipped checkpoints are over-confident —
see [dev-playbook.md](dev-playbook.md#writing-good-questions)).

- **Input**: `schema`, `dataset`, optional holdout split fraction.
- **Output**: `CalibrationResult` — fitted `temperatures`, `ece_before` /
  `ece_after` per question, `n_fit`/`n_holdout`, and a new `report_id`
  computed with the calibration applied.

### `laya_compare_models`
Run the same state and questions on all three checkpoints
(`english`/`multilingual`/`typed-decisions`) side by side.

- **Input**: `state`, `questions`.
- **Output** (expected): per-model `ItemResult`s plus a summary of where
  they disagree — useful when deciding which checkpoint a schema should
  pin, or when multilingual traffic needs checking against `typed-decisions`.

### `laya_sample_for_labeling`
Given a finished batch job, pick items worth hand-labeling next (mix of
lowest-confidence and random items), to grow a dataset efficiently instead
of labeling everything.

- **Input**: `job_id`, optional `n` (how many to sample).
- **Output** (expected): the sampled items (`state` + the model's current
  answers) so a human (or Claude) can label them and feed the labels into
  `laya_save_dataset`.

## Ops

### `laya_status`
- **Input**: none.
- **Output**: `EngineStatus` — `laya_version`, `device`, `threads`,
  `loaded`/`available` checkpoints, `busy`, `waiting` (queued interactive
  calls), `rss_gb`, `uptime_s`, `ms_per_row` (rolling average, used for job
  ETAs).

### `laya_load_models` / `laya_unload_models`
Manually load or evict checkpoints ahead of expected traffic (e.g. warm
`typed-decisions` before running an evaluation), or free RAM.

- **Input**: `models` (list of `english`/`multilingual`/`typed-decisions`);
  `laya_unload_models` with no argument unloads everything not currently in
  `LAYA_MCP_PRELOAD`.
- **Output** (expected): the resulting loaded-model list (same shape as
  `EngineStatus.loaded`).

### `laya_detect_language`
Cheap script/language detection with no model forward pass (`laya.lang.analyse`
/ `laya.detect_language`) — useful to decide `lang_guess` before a real call,
or just to inspect routing.

- **Input**: `text` or `state`.
- **Output** (expected): script, detected language code, and whether it
  would route to the English checkpoint.

### `laya_usage_stats`
- **Input**: optional `since` (defaults to a recent window), optional `team`
  filter.
- **Output**: `UsageStats` — rows grouped by `(team, tool, schema_ref)`:
  `calls`, `rows`, `p50_ms`/`p95_ms`, `needs_review_rate`, `errors`. Raw
  state text is never included (`LAYA_MCP_LOG_CONTENT` only controls
  logging, not this — usage stats are counts/latencies only, always).

## Question format

Every `questions` argument is `{question_id: {type, instructions, criteria}}`:

| type | `criteria` | answer |
|---|---|---|
| `choice` | `{label: description}` (description may be `null`) or a list of labels | chosen label + probabilities per label |
| `score` | ordered list of level descriptions, index 0 = lowest | expected value (float) + most likely level index |
| `noul` | optional `{"true": "...", "false": "..."}` | probability of true (`p_true`) |

Full format, limits and tips: the `laya://docs/question-types` resource, and
[dev-playbook.md](dev-playbook.md#writing-good-questions).

## Resources

| URI | Contents |
|---|---|
| `laya://docs/question-types` | Question format, limits, how to write good criteria |
| `laya://presets/{name}` | One of the built-in presets (`triage`/`email`/`guard`/`moderation`/`router`), as sent to Laya |
| `laya://schemas/{team}/{name}` | Latest version of a saved schema |
| `laya://reports/{id}` | An evaluation report (same content as `laya_get_report`) |

## Prompts

| Prompt | Purpose |
|---|---|
| `design-decision-schema` | Walks through turning a task into a well-formed set of typed questions |
| `evaluate-and-promote` | Guides running `laya_evaluate` -> `laya_calibrate` -> `laya_promote_schema` on a schema |
| `triage-dataset` | Helps pick and label examples (pairs with `laya_sample_for_labeling`) to build an evaluation dataset |
