"""MCP prompts: guided workflows for designing, evaluating and applying Laya schemas."""
from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field


def register(mcp: FastMCP) -> None:
    @mcp.prompt(
        name="design-decision-schema",
        title="Design a Laya decision schema",
        description="Turn a recurring decision into a saved, validated Laya schema.",
        tags={"schemas"},
    )
    def design_decision_schema(
        goal: Annotated[str, Field(description="The decision to automate, e.g. 'route new GitHub issues to a team'")],
        team: Annotated[str, Field(description="Team namespace to save the schema under, e.g. 'dev'")],
    ) -> str:
        return f"""\
Design a Laya decision schema for this recurring decision: {goal}
Save it under the team namespace "{team}".

1. Read the resource laya://docs/question-types for the question format and the rules for good criteria.
2. Decide what the state will contain: a small JSON object with only the fields the decision needs
   (e.g. {{"title": ..., "body_excerpt": ..., "labels": ...}}). Aim for well under 2,000 characters.
3. Draft the questions (usually 1-5). For each one:
   - pick the type: choice (one label of several), score (ordered levels, lowest first) or noul (yes/no);
   - write one short instruction that names the state fields in backticks;
   - give every choice label a description, make labels mutually exclusive, add "other" if needed;
   - keep one decision per question.
4. Call laya_validate_questions with the draft and fix every error and warning it reports.
5. Try it: run laya_classify with detail="full" on 3-5 realistic examples (including a tricky one) and
   check that the answers and confidences make sense. Adjust label descriptions where it goes wrong.
6. Save it with laya_save_schema (team "{team}", a short kebab-case name, a one-line description).
   It starts as a draft: laya_decide will mark answers "unverified" until it is evaluated.
7. Tell me the schema reference, the questions, and what labeled data we need to evaluate it
   (at least 50 examples with the correct answers; the evaluate-and-promote prompt covers that step).
"""

    @mcp.prompt(
        name="evaluate-and-promote",
        title="Evaluate, calibrate and promote a schema",
        description="Measure a schema on a labeled dataset, calibrate it, and promote it if it meets its targets.",
        tags={"schemas", "workbench"},
    )
    def evaluate_and_promote(
        schema: Annotated[str, Field(description="Schema reference, e.g. 'dev/issue-triage'")],
        dataset: Annotated[str, Field(description="Labeled dataset name, e.g. 'dev/issue-triage-labels'")],
    ) -> str:
        return f"""\
Evaluate the Laya schema "{schema}" on the labeled dataset "{dataset}", calibrate it, and promote it if it
meets its targets.

1. Check the dataset with laya_list_datasets: it needs at least 50 examples (more is better) whose
   `expected` answers cover every question of the schema. If it is missing or too small, stop and tell me
   what to label (laya_sample_for_labeling can pick items from a finished batch job).
2. Start laya_evaluate for schema "{schema}" and dataset "{dataset}". It runs as a background job: poll
   laya_job_status every 30-60 s until it completes.
3. Read the report (laya://reports/<report_id> from the job's result_ref). For each question look at
   accuracy, the confusion matrix, ECE, the recommended threshold and its coverage, and the worst
   confident misses. Summarise them for me in a short table.
4. If ECE is above ~0.05 or confident misses appear, run laya_calibrate and compare ECE before/after.
5. If a question misses its target, look at the confusion matrix and the misses: usually the fix is a
   clearer label description, merging two confusable labels, or splitting the question. Propose the change,
   save a new version with laya_save_schema, and evaluate again. Do not lower the target to make it pass.
6. When every question meets its target, call laya_promote_schema. Report the promoted version, its
   thresholds and the expected share of answers that will be 'decided' vs 'needs_review'.
"""

    @mcp.prompt(
        name="triage-dataset",
        title="Triage a dataset with Laya",
        description="Classify a large set of items with a batch job and only review the uncertain ones.",
        tags={"batch"},
    )
    def triage_dataset(
        description: Annotated[str, Field(description="What the items are and what should be decided about each")],
    ) -> str:
        return f"""\
Triage this dataset with Laya: {description}

1. Pick the questions:
   - a saved schema if one fits (laya_list_schemas) - preferred, it has measured thresholds;
   - otherwise a preset (laya://presets/triage, email, moderation, router, guard);
   - otherwise write questions following laya://docs/question-types and check them with
     laya_validate_questions.
2. Prepare each item as a concise JSON object with only the fields the questions refer to (strip
   signatures, quoted replies, boilerplate; keep it under ~2,000 characters).
3. Estimate the cost first: rows = items x questions, ~1 s per row, more for long input (laya_status shows the live figure).
   Tell me the estimate before starting if it is over 30 minutes.
4. Submit with laya_classify_batch in chunks of at most 500 items: the first call with finalize=false,
   then job_id=<id> for the next chunks, finalize=true on the last one.
5. Poll laya_job_status every 30-60 s. When done, read laya_job_results: use `summary` for the overall
   distribution, and needs_review_only=true (paginate with next_offset) for the items Laya was unsure
   about. Decide those yourself; accept the rest as-is.
6. Report the distribution of answers, how many items needed review, and what you decided for them.
"""
