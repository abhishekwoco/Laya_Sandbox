# Dev playbook: working with Laya

This is for the dev team using `laya-mcp` from Claude Code. It covers when to
hand a decision to Laya instead of reasoning about it yourself, how the
propose/score/review loop works, how to write questions that actually
calibrate well, and the lifecycle for turning a draft schema into something
you can trust unattended.

## When to delegate to Laya

Laya is a fast, cheap, calibrated classifier — not a reasoning engine. Hand it
work that's fundamentally a **lookup or a judgment call over a fixed set of
options**, at volume:

- **Bulk classification / triage**: labeling a backlog of issues, tickets, or
  log lines by category, severity, or urgency.
- **Tagging**: area, effort, risk tags on planning tasks — anything you'd
  otherwise do with a dropdown.
- **Filtering research sources**: is this doc/page/snippet relevant, current,
  authoritative — before you spend context reading it in full.
- **Scanning untrusted text**: a first-pass guardrail on fetched web content
  or pasted documents (`laya_scan_untrusted`) before you act on it.

**Don't** delegate:

- **Open-ended reasoning**: anything where the "answer" is a plan, a
  synthesis, or depends on chaining several judgments together. Laya answers
  one flat set of typed questions in one pass — it doesn't reason about its
  own answers.
- **Code correctness**: whether a diff is right, whether a test should pass,
  whether an approach is sound. That's exactly the reasoning Claude should
  keep doing itself.
- **Anything where being wrong is expensive and confidence won't catch it**:
  Laya's calibration is *statistical* — a well-calibrated 90%-confidence
  question is right about 90% of the time, not every time. Fine for triage
  (a misrouted ticket gets re-routed); not fine for anything where a single
  wrong answer, confidently given, causes real harm.

## The loop: Claude proposes, Laya scores, Claude decides

The pattern that makes this worth doing:

1. **Claude proposes** — extracts or writes a concise `state` (see
   [cost model](#cost-model-and-writing-concise-states) below) and picks (or
   already has) a schema of typed questions.
2. **Laya scores** — one forward pass answers every question with a
   calibrated confidence, via `laya_decide` (schema-based, thresholded) or
   `laya_classify`/`laya_apply_preset` (ad hoc, no threshold).
3. **Claude decides** — reads `needs_review` (the list of question ids below
   the schema's threshold) and only spends reasoning on those. Everything
   else, Laya already decided with a confidence you can act on.

For volume work, replace step 2 with `laya_classify_batch` + polling
(`laya_job_status`/`laya_job_results`) instead of one call per item — see
[tools.md](tools.md#laya_classify_batch).

An **unevaluated schema never produces `decided`/`needs_review`** — every
answer comes back `status: "unverified"` until the schema has been evaluated
(see [Schema lifecycle](#schema-lifecycle)). Treat `unverified` answers as
suggestions to double-check, not as settled.

## Cost model and writing concise states

Cost scales with **items x questions**, not with how much text you send per
item — but a longer state still costs more per row (more tokens to encode)
and eats into the model's option-token budget (`head_max_len`; see
[Honest limits](../../laya/README.md) in the laya README if you're pushing
past ~12 choice labels or long criteria text). On this host, a single
question over a short state costs roughly 0.5 s of CPU time (1.6 s for a ~480-token state), and
only one forward pass runs at a time server-wide.

Practical guidance:
- **Extract before you classify.** Send the ticket body, not the ticket body
  plus its full comment thread plus unrelated metadata. If you only need the
  subject line and first paragraph to answer the questions, send only that.
- **Batch your questions, not your calls.** Asking `area`, `type`,
  `severity`, `needs_repro`, `duplicate_likely` about one issue is one state
  x five questions = five rows, still one forward pass. Five separate calls
  asking one question each cost the same rows but five times the round trips.
- **Use `laya_classify_batch` once you're past a handful of items.** The sync
  row budget (`LAYA_MCP_SYNC_ROW_BUDGET`, default 20) exists so an
  interactive call can't accidentally block for a minute; batch jobs are the
  right tool for anything bigger and don't hold a connection open.
- **`detail="compact"` (the default) is enough for automated flow.** Only ask
  for `detail="full"` (adds probability distributions) when you're actually
  going to look at the distribution — e.g. debugging why a question is
  uncalibrated.

## Writing good questions and criteria

- **One clear decision per question.** "What department and how urgent" is
  two questions (`department` choice + `urgency` score), not one.
- **Choice labels need to be distinguishable in a few tokens.** Laya splits a
  fixed budget across all of a choice question's labels+descriptions
  (`head_max_len`, 192 tokens on the English checkpoint). More than ~8-12
  labels, or long descriptions, means less signal per label — `
  laya_validate_questions` warns you about this before you save a schema.
- **Write descriptions, not just labels.** `{"billing": "invoices, payments,
  refunds"}` gives the model something to match against; a bare `{"billing":
  null}` relies on the label text alone.
- **Score criteria must be a strictly ordered list**, lowest level first —
  the model's answer is an expected value over that ordering, so an
  out-of-order list silently produces nonsense scores.
- **Be careful with `noul` (yes/no) on the English checkpoint.** It's known
  to sometimes anchor on its own `true:`/`false:` option labels rather than
  the state content (laya README, "Honest limits"). If a `noul` question
  looks stuck on one answer regardless of input during evaluation, rephrase
  it as a two-option `choice` with neutral keys instead:
  ```json
  {"type": "choice", "instructions": "Is this review positive?",
   "criteria": {"A": "yes, the review is positive", "B": "no, the review is negative"}}
  ```
- **Avoid overlapping criteria.** If two choice labels' descriptions could
  both plausibly apply to the same input, expect the model to split
  confidence between them and never clear a sensible threshold.
- **Validate before you save.** `laya_validate_questions` runs the same
  structural checks Laya itself would (bad type, missing criteria, etc. —
  surfaced verbatim from `Agent._check_question`) plus budget/label-count
  warnings, with **no inference cost**.

## Schema lifecycle

```
save (draft) ──▶ attach a labeled dataset ──▶ evaluate ──▶ calibrate ──▶ promote (trusted)
```

1. **`laya_save_schema`** — team, name, your questions, and `targets`
   (`min_accuracy`, default 0.9; `min_examples`, default 50; optional
   `per_question` overrides). Creates version 1, status `draft`. Every
   `laya_decide` call against a `draft` schema returns `unverified` answers —
   it works, but nothing is gated on confidence yet.
2. **`laya_save_dataset`** — collect ~50-100+ real, labeled examples (e.g.
   closed issues with their actual area/type/severity, sampled log lines
   someone has triaged, tasks with actual effort/risk in hindsight). Send in
   chunks with `append=true` as you gather more.
3. **`laya_evaluate`** — schema + dataset -> an async job -> an `EvalReport`:
   per-question accuracy, confusion matrix, precision/recall, score MAE, ECE,
   accuracy broken down by confidence band, a recommended threshold and its
   coverage, and the worst confident misses (the cases most worth reading by
   hand). Schema status becomes `evaluated` once a report exists, but
   `laya_promote_schema` still checks it against `targets`.
4. **`laya_calibrate`** — fits one temperature per question on a held-out
   split and stores it on the schema version. **Do this before trusting
   confidence numbers**: Laya's shipped checkpoints are over-confident as
   distributed (mean ECE around 0.3-0.5 before fitting, per the laya
   README), so an un-calibrated 95%-confidence answer is not actually right
   95% of the time.
5. **`laya_promote_schema`** — marks the schema `trusted`, but **refuses**
   unless the latest evaluation clears `targets.min_accuracy` (and any
   `per_question` override) on at least `targets.min_examples` examples.
   Read the refusal message — it names which question(s) are short, and by
   how much.

Re-run evaluate/calibrate after any meaningful change to a schema's questions
or to its dataset; a new schema version starts back at `draft`.

`laya_compare_models` is worth running early in this process too — if a
schema's traffic isn't purely English, check whether `multilingual` or
`typed-decisions` actually answers your specific questions better before you
sink evaluation effort into the auto-routed default.

## The four dev schemas

Full files (with `targets`) are under `examples/schemas/dev/*.json`; load them
all at once with `scripts/seed_schemas.py` (see the
[README quickstart](../README.md#seeding-the-dev-teams-schemas)). Each starts
as an unevaluated draft — run it through
[the lifecycle above](#schema-lifecycle) with your own labeled examples
before treating its decisions as anything but `unverified`.

### `dev/issue-triage`
Area, type, severity, and two flags for an incoming issue/ticket.

```json
{
  "area": {
    "type": "choice",
    "instructions": "Which part of the system does this issue concern?",
    "criteria": {
      "frontend": "UI, client-side rendering, browser behavior",
      "backend": "server-side logic, APIs, business rules",
      "infra": "deployment, CI/CD, servers, networking, scaling",
      "data": "database schema, migrations, data quality, reporting",
      "docs": "documentation, README, comments, onboarding material",
      "other": "none of the above, or spans multiple areas"
    }
  },
  "type": {
    "type": "choice",
    "instructions": "What kind of issue is this?",
    "criteria": {
      "bug": "something that used to work, or should work, is broken",
      "feature": "a request for new functionality that doesn't exist yet",
      "question": "someone is asking how something works or should work",
      "chore": "maintenance, refactor, dependency bump, or cleanup with no user-facing behavior change"
    }
  },
  "severity": {
    "type": "score",
    "instructions": "How severe is this issue's impact if it stays unresolved?",
    "criteria": [
      "cosmetic or trivial: no functional impact",
      "minor: a workaround exists, or it affects few users",
      "major: no workaround, affects many users or a core workflow",
      "critical: data loss, security exposure, or a full outage"
    ]
  },
  "needs_repro": {
    "type": "noul",
    "instructions": "Is this issue missing clear steps to reproduce, so someone would have to ask the reporter for more information before it can be worked on?"
  },
  "duplicate_likely": {
    "type": "noul",
    "instructions": "Does this report read like a common, well-worn class of complaint or request (the kind that's often already been filed), rather than something novel or specific to this codebase?"
  }
}
```

Note on `duplicate_likely`: Laya has no retrieval step, so this is a coarse
"does this sound generic" heuristic, not an actual search against your
existing issue tracker. For real duplicate detection, do an embedding/search
lookup first and only use Laya to double-check the top match, or drop the
question if it doesn't evaluate well on your data.

### `dev/log-classify`
Category, severity, and an actionability flag for a single log line or short
excerpt.

```json
{
  "category": {
    "type": "choice",
    "instructions": "What kind of log entry is this?",
    "criteria": {
      "error": "an exception, stack trace, or explicit failure",
      "warning": "a recoverable problem or a deprecation notice",
      "security": "auth failures, access violations, suspicious activity",
      "performance": "slow queries, timeouts, resource exhaustion",
      "info": "routine informational or lifecycle logging",
      "other": "none of the above"
    }
  },
  "severity": {
    "type": "score",
    "instructions": "How severe is this log entry?",
    "criteria": [
      "noise: safe to ignore",
      "low: worth noting, no immediate action",
      "high: should be investigated soon",
      "critical: needs immediate attention, likely user-facing impact"
    ]
  },
  "actionable": {
    "type": "noul",
    "instructions": "Does this log entry describe something a human should act on (investigate, fix, or escalate), as opposed to routine, expected, or already-handled noise?"
  }
}
```

Good fit for `laya_classify_batch` over a whole log file, then
`laya_job_results?needs_review_only=true` (low-confidence rows) plus
filtering `question_id=severity, value=3` for the criticals, without reading
every line.

### `dev/source-relevance`
For filtering research sources (docs, articles, code, papers) before reading
them in full. Send `{"query": "<research question>", "source": "<excerpt>"}`
as the state.

```json
{
  "relevance": {
    "type": "score",
    "instructions": "How relevant is this source to the research question given in `query`?",
    "criteria": [
      "not relevant: off-topic",
      "tangential: touches the topic but doesn't answer the question",
      "relevant: directly addresses the question",
      "highly relevant: directly and thoroughly addresses the question"
    ]
  },
  "source_type": {
    "type": "choice",
    "instructions": "What kind of source is this?",
    "criteria": {
      "official_docs": "vendor or project documentation, API reference, or spec",
      "source_code": "actual code, a repository, or inline code comments",
      "blog_post": "an individual's or company's blog or article",
      "forum_qa": "Stack Overflow, forums, Q&A sites, mailing lists",
      "academic_paper": "peer-reviewed or preprint research",
      "other": "none of the above"
    }
  },
  "is_outdated": {
    "type": "noul",
    "instructions": "Does this source describe a version, API, or state of the world that looks superseded or no longer current?"
  },
  "authoritative": {
    "type": "noul",
    "instructions": "Does this source come from an authoritative origin for the topic (the project's own maintainers or docs, a recognized standard, or a primary source), rather than a secondary summary or unverified opinion?"
  }
}
```

This schema's `targets.min_accuracy` is set to 0.85 rather than 0.9 in the
example file — relevance judgments are more subjective than the others here,
so hold it to a slightly lower bar and lean more on human spot checks of the
worst confident misses.

### `dev/task-tagging`
Area, effort, risk, and a blocking flag for planning tasks.

```json
{
  "area": {
    "type": "choice",
    "instructions": "Which part of the system does this task concern?",
    "criteria": {
      "frontend": "UI, client-side rendering, browser behavior",
      "backend": "server-side logic, APIs, business rules",
      "infra": "deployment, CI/CD, servers, networking, scaling",
      "data": "database schema, migrations, data quality, reporting",
      "docs": "documentation, README, comments, onboarding material",
      "other": "none of the above, or spans multiple areas"
    }
  },
  "effort": {
    "type": "score",
    "instructions": "How much engineering effort does this task look like it needs (t-shirt size)?",
    "criteria": [
      "XS: minutes to a couple hours, a trivial change",
      "S: about a day, a small well-scoped change",
      "M: a few days, moderate scope or some unknowns",
      "L: one to two weeks, significant scope or a cross-cutting change",
      "XL: multiple weeks, large or poorly-understood scope"
    ]
  },
  "risk": {
    "type": "score",
    "instructions": "How risky is this task: chance of breaking something, needing rework, or hitting unknowns?",
    "criteria": [
      "low: well-understood, isolated change",
      "medium: touches shared code or has some unknowns",
      "high: touches critical paths or data, or has significant unknowns"
    ]
  },
  "blocking": {
    "type": "noul",
    "instructions": "Does this task's description say or imply that other work is waiting on it?"
  }
}
```

`effort` and `risk` are both weak signals in isolation (ordinal `score`
questions are Laya's least accurate primitive per its own benchmarks) — treat
them as a first-pass sort for planning, not as the final estimate, and check
`needs_review` before committing sprint numbers to them.
