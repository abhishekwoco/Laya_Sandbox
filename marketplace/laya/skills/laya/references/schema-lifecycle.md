# Schema lifecycle: from draft to trusted

A schema is a named, versioned question set owned by a team (`dev/issue-triage@3`).
Status moves `draft -> evaluated -> trusted`. Only `trusted` schemas return `decided`
answers from `laya_decide`; the others return `unverified` so nobody mistakes an
unmeasured guess for a decision.

## 1. Save the questions

`laya_save_schema(name, questions, description, targets)` creates a new draft version
every time. `targets` sets what "good enough" means, e.g.
`{"min_accuracy": 0.9, "min_examples": 50, "per_question": {"severity": 0.8}}`.
Check first whether the team already has one: `laya_list_schemas()`.

## 2. Build a labeled dataset (50-200 examples)

`laya_save_dataset(name, examples, append=true)`, up to 500 examples per call. Each
example is the item plus the correct answers:

```json
{"state": {"title": "Deadlock in payroll transaction"},
 "expected": {"area": "database", "severity": 2, "has_repro": false}}
```

`expected` uses the label for choice, the level index (int) for score, true/false for
yes/no. Where the labels can come from:

- **Real history** (best): closed issues with their final labels, resolved tickets with
  their category, emails with the team that handled them.
- **You label, a human reviews**: label 50-100 items yourself, show the user the list,
  and let them correct it. Say plainly that the dataset then measures agreement with
  your labels as reviewed by the user.
- **Grow it from batch jobs**: `laya_sample_for_labeling(job_id, n, strategy="mixed")`
  returns the least certain plus random items with `suggested_expected` pre-filled for a
  human to correct, then append them to the dataset.

Cover every label, including rare ones; a dataset with no `outage` examples can't tell
you anything about `outage`.

## 3. Evaluate

`laya_evaluate(schema, dataset)` starts a background job; poll `laya_job_status`, and the
report id appears in `result_ref`. `laya_get_report(report_id)` gives, per question:

- `accuracy` and, for scores, `mae`
- `ece` - calibration error: does 0.9 confidence mean right 90% of the time?
- `confusion` + `labels` - which labels get mixed up (fix those descriptions)
- `recommended_threshold`, `accuracy_at_threshold`, `coverage_at_threshold` - above this
  confidence the target accuracy is met, for this share of answers
- `worst_misses` - confident wrong answers: often mislabeled examples, sometimes a
  missing label

Evaluation stores the recommended thresholds on the schema version and marks it
`evaluated`.

## 4. Calibrate

`laya_calibrate(schema)` fits one temperature per question on 70% of the evaluated
examples, measures on the other 30%, stores the temperatures and rebuilds the report.
Needs at least 20 evaluated examples. It changes confidences, not which answer is picked.

## 5. Promote

`laya_promote_schema(schema)` succeeds only if the latest report meets every question's
target with enough examples. When it refuses, the message lists what failed and what to
do: add examples, sharpen criteria (a new schema version), calibrate, re-evaluate.

## Changing a trusted schema

Any change to the questions is a new version (draft again). Re-run evaluate -> calibrate
-> promote on the new version; `laya_decide` with a bare name uses the latest version, so
pin `team/name@N` in automation until the new one is trusted.
