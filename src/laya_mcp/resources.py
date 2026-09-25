"""MCP resources: the question-type guide, presets, saved schemas and evaluation reports."""
from __future__ import annotations

import json
from string import Template

from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError, ResourceError

from .models import SchemaRef
from .state import get_state
from .tools._core import run_sync
from .tools.decide import PRESET_DESCRIPTIONS, PRESET_FIELDS, preset_questions

QUESTION_TYPES_DOC = """\
# Laya question types

Laya is a System-1 classifier: it reads a short piece of content (the **state**) and answers every
question you ask about it in **one forward pass**, with a probability for each answer. It does not
reason, plan or look things up. Use it for judgements a person makes at a glance: which team, which
category, is this urgent, how severe, does this mention X. Keep reasoning for yourself.

## The three types

| type | answers | `criteria` | example |
|---|---|---|---|
| `choice` | one label out of several | `{label: description}` (preferred) or `[labels]` | which team owns a bug |
| `score` | a level on an ordered scale | list of level descriptions, **lowest first** | severity 0-3 |
| `noul` | yes / no | optional `{"true": "...", "false": "..."}` | does the log line show a timeout? |

```json
{
  "team":     {"type": "choice", "instructions": "Which team owns the bug in `title` and `error`?",
               "criteria": {"backend": "API, database, background jobs",
                            "frontend": "UI, CSS, browser errors",
                            "infra": "deploys, networking, certificates"}},
  "severity": {"type": "score", "instructions": "How severe is the bug in `error`?",
               "criteria": ["cosmetic", "degraded but usable", "feature broken", "outage or data loss"]},
  "has_repro":{"type": "noul", "instructions": "Does `body` include steps to reproduce?"}
}
```

## What comes back

- choice: `value` = chosen label, `confidence` = its probability.
- score: `value` = expected level (float, e.g. 1.7), `level` = most likely level, `confidence` = its probability.
- noul: `value` = true/false, `p_true`, `confidence` = max(p_true, 1 - p_true).
- `detail="full"` adds every label's probability (and the score legend).
- With a threshold (laya_classify `threshold=`, or a schema in laya_decide), each answer gets a
  `status`: **decided** (at or above threshold: act on it) or **needs_review** (below: you decide).
  Draft schemas answer **unverified**: not measured yet, treat as suggestions.

## Writing good questions

- **Name the field** you mean in backticks: "Which team owns the bug in `title`?" and send the state as
  a JSON object with those fields.
- **Describe every label.** `{"backend": "API, database, jobs"}` beats `["backend"]`. Descriptions are
  the only thing the model knows about a label.
- **Make labels mutually exclusive** and add a catch-all (`"other": "none of the above"`) when
  inputs can fall outside your list.
- **Keep label sets small**: 2-8 labels works best; beyond ~12 accuracy drops. Split a big taxonomy
  into two questions (area, then sub-area) instead.
- **Score levels are ordered and concrete**: "cosmetic / degraded / broken / outage", not "1/2/3/4".
- **One question, one decision.** "Is it urgent and from a customer?" is two noul questions.
- **Short instructions** (one sentence). Put definitions in label descriptions, not in instructions.
- **Concise states**: pre-extract what matters (title, first lines of a stack trace, the changed hunk).
  The english checkpoint reads ~512 tokens including the question and options; the rest is cut off.

## Limits

- Up to $max_questions questions per call; state up to $max_state_chars characters (but ~2,000 is plenty).
- Synchronous calls (laya_classify, laya_decide, laya_apply_preset): at most **$sync_row_budget rows**,
  where rows = items x questions. laya_status shows the live limits.
- Batch jobs (laya_classify_batch): up to $max_items_per_call items per call; append more chunks to the same job.
- Languages: English text goes to the `english` checkpoint, other languages/scripts to `multilingual`
  (auto-routed; see laya_detect_language).

## Cost model

This server runs on a CPU. Each **row** (one question on one item) costs about **0.5 s** for a
short item, **~0.9 s** for ~200 tokens and **~1.6 s** for ~480 tokens: 5 questions on a short item
take ~3 s, on a 480-token item ~8 s. Only one inference runs at a
time, so large jobs belong in the batch queue where they don't block interactive calls. Longer
states cost more; asking 5 questions at once is much cheaper than 5 separate calls.

## Which tool

| situation | tool |
|---|---|
| a one-off judgement, ad-hoc questions, <= $sync_row_budget rows | `laya_classify` |
| a decision you make repeatedly (triage, tagging) with measured accuracy | `laya_save_schema` once, then `laya_decide` |
| a common workflow (support triage, email, moderation, routing, guardrails) | `laya_apply_preset` |
| more than $sync_row_budget rows, a dataset, a backlog | `laya_classify_batch` (+ `laya_job_status`, `laya_job_results`) |
| untrusted text you are about to read or act on | `laya_scan_untrusted` |
| checking a question set before using it | `laya_validate_questions` |
| measuring and calibrating a schema | `laya_save_dataset`, `laya_evaluate`, `laya_calibrate`, `laya_promote_schema` |

## Trusting answers

Shipped checkpoints are over-confident, so a raw 0.9 is not 90% accuracy. Schemas fix this: evaluate
a schema on ~50-100 labeled examples, calibrate it, and laya_decide applies per-question thresholds
chosen for a target accuracy (default $target_pct%). Until then, treat answers as a fast first pass and
review anything that matters.
"""


def render_question_types_doc() -> str:
    """The guide with the server's live limits filled in."""
    from .config import get_settings

    try:
        s = get_state().settings
    except RuntimeError:
        s = get_settings()
    return Template(QUESTION_TYPES_DOC).safe_substitute(
        sync_row_budget=s.sync_row_budget,
        max_questions=s.max_questions,
        max_state_chars=f"{s.max_state_chars:,}",
        max_items_per_call=s.max_items_per_call,
        target_pct=round(s.default_target_accuracy * 100),
    )


def register(mcp: FastMCP) -> None:
    @mcp.resource(
        "laya://docs/question-types",
        name="question-types",
        title="Laya question types guide",
        description="How to write Laya questions (choice/score/noul), limits, cost model and which tool to use when.",
        mime_type="text/markdown",
        tags={"docs"},
    )
    def question_types() -> str:
        return render_question_types_doc()

    @mcp.resource(
        "laya://presets/{name}",
        name="preset",
        title="Laya preset question set",
        description="Questions of a built-in preset: triage, email, guard, moderation or router.",
        mime_type="application/json",
        tags={"presets"},
    )
    def preset(name: str) -> str:
        key = name.strip().lower()
        if key not in PRESET_FIELDS:
            raise NotFoundError(f"unknown preset {name!r}; available: {', '.join(sorted(PRESET_FIELDS))}")
        return json.dumps(
            {
                "preset": key,
                "description": PRESET_DESCRIPTIONS[key],
                "state_field": PRESET_FIELDS[key],
                "questions": preset_questions(key),
            },
            indent=2,
        )

    @mcp.resource(
        "laya://schemas/{team}/{name}",
        name="schema",
        title="Saved decision schema",
        description="Latest version of a saved schema: questions, status, thresholds, calibration temperatures.",
        mime_type="application/json",
        tags={"schemas"},
    )
    async def schema(team: str, name: str) -> str:
        try:
            ref = SchemaRef.parse(f"{team}/{name}")
        except ValueError as e:
            raise ResourceError(str(e)) from e
        try:
            info = await run_sync(get_state().repo.get_schema, ref)
        except LookupError as e:
            raise NotFoundError(f"schema {ref} not found") from e
        return info.model_dump_json(indent=2)

    @mcp.resource(
        "laya://reports/{report_id}",
        name="report",
        title="Evaluation report",
        description="An evaluation report: per-question accuracy, confusion matrix, calibration (ECE), thresholds.",
        mime_type="application/json",
        tags={"reports"},
    )
    async def report(report_id: str) -> str:
        try:
            rid = int(report_id)
        except ValueError as e:
            raise ResourceError(f"report id must be an integer, got {report_id!r}") from e
        try:
            rep = await run_sync(get_state().repo.get_report, rid)
        except LookupError as e:
            raise NotFoundError(f"report {rid} not found") from e
        return rep.model_dump_json(indent=2)
