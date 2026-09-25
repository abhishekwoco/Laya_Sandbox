# Recipes

Adapt these; the point is the shape of the call, not the exact labels. Check
`laya_list_schemas()` first - if the team has a schema for the job, use `laya_decide`.

## Contents
- Issue / bug triage
- Log lines
- Research: screening sources
- Planning: comparing options on several criteria
- Planning: tagging tasks
- Support tickets and email
- Sales leads

## Issue / bug triage (dev)

Items: `{"title", "body" (first ~10 lines), "error" (first lines of the trace)}`.
Questions: `area` (choice), `type` (choice: bug / feature / question / chore), `severity`
(score 0-3), `has_repro` (noul), `duplicate_likely` (noul, only if you include the titles
of candidate duplicates in the item). 30 issues x 5 questions = 150 rows -> batch job.
Report: counts per area/type, the high-severity list, and the `needs_review` items you
checked yourself.

## Log lines

Items: one line or one grouped error block each, timestamps and ids stripped
(dedupe identical messages first and keep a count). Questions: `category` (choice:
timeout / auth / db / validation / dependency / other), `severity` (score), `actionable`
(noul). Summarise by category x severity; drill into actionable + high severity.

## Research: screening sources

Items: `{"title", "snippet" (the paragraph that matters), "published", "source"}` per
search result or document. Questions: `relevance` (score: off-topic ... directly answers
the question - put the research question in the instructions), `source_type` (choice:
official docs / vendor blog / forum / paper / news), `is_outdated` (noul - include the
date in the item), `authoritative` (noul). Read in full only the high-relevance,
non-outdated ones; mention what you skipped and why.

## Planning: comparing options on several criteria

This is Laya scoring options you have already researched, not Laya choosing. Items: one
per option, each a short, comparable, fact-bearing summary (same fields for all:
`{"option", "summary", "costs", "risks", "effort"}`). Questions: one `score` per
criterion, each worded for that criterion ("How much operational risk does `risks`
describe?", levels low...high). Then:

1. Run `laya_classify` with `detail="full"`.
2. Combine scores with the weights the user cares about (state them), yourself.
3. Where confidence is low or options are within noise of each other, reason it through
   yourself and say so. Present the table plus your recommendation, and be explicit that
   the scores come from Laya's reading of your summaries.

## Planning: tagging tasks

Items: one per task `{"task", "detail"}`. Questions: `area` (choice), `effort` (score
XS/S/M/L/XL), `risk` (score), `blocking` (noul: "Does `task` have to finish before others
can start?"). Use the tags to order and group the plan; sanity-check XL/high-risk ones
yourself.

## Support tickets and email

`laya_apply_preset(preset="triage", items=[...])` for tickets (intent, is_urgent,
frustration, refund_requested, churn_risk) - plain strings are wrapped as `message`.
`preset="email"` for inbound mail (category, is_spam, is_phishing, urgency,
needs_reply); pass `categories` to use your own team names. Presets are uncalibrated for
WoCo's data: for routine use, copy the questions into a team schema and evaluate it.

## Sales leads

Items: `{"company", "message", "source"}`. Questions: `intent` (choice: pricing / demo /
partnership / support-misrouted / spam), `buying_stage` (score: browsing ...
ready to buy), `decision_maker` (noul: "Does `message` come from someone who can approve
a purchase?"), `needs_reply` (noul). Hand misrouted support to the support flow.
