# Designing questions Laya answers well

Laya compares the item text against the question and each option's text. It has no
world knowledge beyond pattern matching, so accuracy comes almost entirely from how
clearly the options are described and how clean the item is.

## Items (the `state`)

- Pass a JSON object with named fields and point questions at them with backticks:
  `{"title": ..., "error": ..., "body": ...}` + "Which area does `title` concern?".
- Pre-extract: the title, the first 5-10 lines of a stack trace, the relevant paragraph.
  Anything past ~512 tokens is cut off, and every extra token costs time on every question.
- Keep items comparable: if you are scoring options against each other, give each the
  same fields at similar length, so differences in answers come from content, not size.
- Strip boilerplate (signatures, quoted reply chains, HTML, log timestamps).

## Choice questions

- 2-12 labels works best; beyond ~12 the label text shares a small token budget and gets
  truncated (`laya_validate_questions` warns about this). Split big taxonomies into two
  questions (e.g. `area` then `component`), or ask the coarse one first.
- Describe every label with the words that actually appear in items:
  `"database": "SQL, queries, migrations, deadlocks, connection pool"` beats
  `"database": "persistence layer issues"`.
- Make labels mutually exclusive, and include an escape hatch (`"other": "none of the
  above"`) so Laya isn't forced to pick a wrong label confidently.
- Stable, short ids (`bug`, `feature`) - the ids are what you branch on in code.

## Score questions

- 3-5 ordered levels, lowest first, each a concrete description:
  `["cosmetic", "annoying but has a workaround", "blocks a feature", "outage or data loss"]`.
- `value` is the expected level (e.g. 2.3); `level` is the single most likely level. Use
  `level` for bucketing, `value` for ranking.

## Yes/no (`noul`) questions

- One property per question, phrased as a question about a field:
  "Does `body` include steps to reproduce?" - not "Is this a good bug report?".
- Avoid negations ("Is it not...?") and compound conditions ("... and ...").
- Optional `criteria`: `{"true": "what counts as yes", "false": "what counts as no"}` helps
  when the boundary is fuzzy.

## Instructions

- One sentence, a direct question, naming the field. Long instructions dilute the signal.
- Ask for what is observable in the text ("Does `error` mention a timeout?"), not for
  judgements that need context Laya doesn't have ("Is this our fault?").

## Checking your questions

1. `laya_validate_questions(questions)` - structure, label budget, overlapping labels,
   cost per item.
2. Run `laya_classify` with `detail="full"` on 3-5 items you know the answers to. If a
   choice question splits its probability between two labels, their descriptions overlap
   - sharpen them.
3. For a recurring decision, move on to a saved schema and a labeled dataset
   (`schema-lifecycle.md`) - that is the only way to know the real accuracy.
