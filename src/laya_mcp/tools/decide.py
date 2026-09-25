"""Decision tools: laya_classify, laya_decide, laya_apply_preset."""
from __future__ import annotations

from typing import Annotated, Any, Literal

import laya
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..engine.answers import to_item_result
from ..models import (
    ClassifyResponse,
    DecideResponse,
    Detail,
    Questions,
    SchemaRef,
    State,
    questions_to_laya,
)
from ..state import current_team, get_state
from ._core import READ_ONLY, ModelName, Stopwatch, count_review, resolve_states, run_sync, tool_errors

server = FastMCP("laya-decide")

# ---- shared parameter types ------------------------------------------------------------------

StateParam = Annotated[
    State | None,
    Field(
        description=(
            "ONE item to judge: a short string, or a JSON object whose fields the questions refer to "
            "in backticks (e.g. {\"title\": ..., \"error\": ...}). Send concise, pre-extracted content "
            "(a title, the first lines of a stack trace, a diff hunk), not whole files: the model "
            "reads ~512 tokens. Use `items` instead for several items."
        )
    ),
]
ItemsParam = Annotated[
    list[State] | None,
    Field(
        description=(
            "SEVERAL items to judge with the same questions (same format as `state`). "
            "Cost is len(items) x questions rows; above the synchronous budget use laya_classify_batch."
        )
    ),
]
QuestionsParam = Annotated[
    Questions,
    Field(
        min_length=1,
        description=(
            "Question id -> question. Types: 'choice' (criteria = {label: description} or [labels]), "
            "'score' (criteria = ordered level descriptions, lowest first), 'noul' (yes/no; criteria "
            "optional {\"true\": ..., \"false\": ...}). All questions are answered in one forward pass. "
            "Example: {\"team\": {\"type\": \"choice\", \"instructions\": \"Which team owns the bug in "
            "`title`?\", \"criteria\": {\"backend\": \"API, database, jobs\", \"frontend\": \"UI, CSS\"}}}"
        ),
    ),
]
ModelParam = Annotated[
    ModelName | None,
    Field(
        description=(
            "Force a checkpoint. Leave unset to auto-route: 'english' for English text, 'multilingual' "
            "for other languages/scripts. 'typed-decisions' is only for its four fine-tuned workflows."
        )
    ),
]
LangParam = Annotated[
    str | None,
    Field(description="Optional ISO 639-1 language hint for routing (e.g. 'hi', 'fr') when you already know it."),
]
DetailParam = Annotated[
    Detail,
    Field(description="'compact' (value + confidence) or 'full' (adds per-label probabilities and score legends)."),
]

PRESET_FIELDS: dict[str, str] = {
    "triage": "message",
    "email": "body",
    "guard": "prompt",
    "moderation": "post",
    "router": "request",
}
PRESET_DESCRIPTIONS: dict[str, str] = {
    "triage": "Customer support ticket triage: intent, is_urgent, frustration, refund_requested, churn_risk.",
    "email": "Inbound email routing and threat filtering: category, is_spam, is_phishing, urgency, needs_reply.",
    "guard": "LLM input guardrail: jailbreak, prompt_injection, sensitive_data, harm_severity, topic.",
    "moderation": "Content moderation: toxic, harassment, threat, spam, severity.",
    "router": "Model routing: difficulty, domain, needs_tools, is_sensitive.",
}


def preset_questions(name: str, categories: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    if name == "triage":
        return laya.triage_questions()
    if name == "email":
        return laya.email_questions(categories)
    if name == "guard":
        return laya.guard_questions()
    if name == "moderation":
        return laya.moderation_questions()
    if name == "router":
        return laya.router_questions()
    raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESET_FIELDS)}")


async def _classify(
    states: list[State],
    qd: dict[str, dict[str, Any]],
    *,
    model: str | None,
    lang: str | None,
    detail: Detail,
    threshold: float | None,
    validate: bool = True,
) -> ClassifyResponse:
    st = get_state()
    sw = Stopwatch()
    with tool_errors(st.settings.request_timeout_s):
        st.engine.check_sync_budget(len(states), len(qd), states)
        warnings = st.engine.validate(qd, model or "english") if validate else []
        raw = await st.engine.predict(states, qd, model=model, lang=lang, timeout=st.settings.request_timeout_s)
    results = [to_item_result(i, r, detail=detail, default_threshold=threshold) for i, r in enumerate(raw)]
    return ClassifyResponse(results=results, rows=len(states) * len(qd), latency_ms=sw.ms, warnings=list(warnings))


# ---- tools -------------------------------------------------------------------------------------

@server.tool(
    name="laya_classify",
    description=(
        "Answer typed questions about content in ONE fast forward pass of the Laya classifier (a calibrated "
        "System-1 model: pattern recognition, no reasoning, no world knowledge). Delegate mechanical judgements "
        "to it the way you would to a subagent - routing, tagging, triage, yes/no checks, severity scores - "
        "instead of reasoning through each item yourself.\n\n"
        "Call with `questions` plus `state` (one item) or `items` (several). Send concise, pre-extracted content, "
        "not whole files. Cost: rows = items x questions at ~0.5-1.6 s per row (longer input costs more) on this CPU server; synchronous calls "
        "are capped (default 20 rows, see laya_status) - use laya_classify_batch above that.\n\n"
        "Each answer has `value` and `confidence` (probability of the chosen answer). Set `threshold` (e.g. 0.8) "
        "to get status decided/needs_review per answer: act on 'decided' answers, and reason about the "
        "'needs_review' ones yourself (they are listed per item in `needs_review`). For a recurring decision, save "
        "the questions as a schema (laya_save_schema) and use laya_decide, which applies measured thresholds."
    ),
    tags={"decisions"},
    annotations={**READ_ONLY, "title": "Classify with Laya"},
)
async def laya_classify(
    questions: QuestionsParam,
    state: StateParam = None,
    items: ItemsParam = None,
    model: ModelParam = None,
    lang: LangParam = None,
    detail: DetailParam = "compact",
    threshold: Annotated[
        float | None,
        Field(ge=0, le=1, description="Optional confidence threshold: answers below it get status 'needs_review'."),
    ] = None,
) -> ClassifyResponse:
    states = resolve_states(state, items)
    return await _classify(states, questions_to_laya(questions), model=model, lang=lang, detail=detail, threshold=threshold)


@server.tool(
    name="laya_decide",
    description=(
        "Apply a SAVED decision schema (a named, versioned question set with measured thresholds and calibration) "
        "to one item (`state`) or several (`items`). Every answer comes back with a status:\n"
        "- 'decided': confidence is at or above the threshold measured for that question - act on it.\n"
        "- 'needs_review': below threshold - YOU decide this one (the ids are listed per item in `needs_review`).\n"
        "- 'unverified': the schema is not trusted yet (draft, or evaluated without meeting its targets); "
        "treat answers as suggestions, and `needs_review` still lists the least certain ones.\n"
        "Only a 'trusted' schema (schema_status in the response) produces 'decided'.\n\n"
        "`schema` is 'team/name', 'team/name@version' or a bare 'name' in your team. Find schemas with "
        "laya_list_schemas or the laya://schemas/{team}/{name} resource. Same cost model as laya_classify "
        "(rows = items x questions, synchronous budget applies; use laya_classify_batch with `schema` for more)."
    ),
    tags={"decisions", "schemas"},
    annotations={**READ_ONLY, "title": "Decide with a saved schema"},
)
async def laya_decide(
    schema: Annotated[str, Field(min_length=1, description="Schema reference: 'team/name', 'team/name@3' or bare 'name' (your team).")],
    state: StateParam = None,
    items: ItemsParam = None,
    lang: LangParam = None,
    model: Annotated[
        ModelName | None,
        Field(description="Force a checkpoint. Leave unset: thresholds were measured with auto-routing."),
    ] = None,
    detail: DetailParam = "compact",
) -> DecideResponse:
    st = get_state()
    sw = Stopwatch()
    states = resolve_states(state, items)
    with tool_errors():
        ref = SchemaRef.parse(schema, default_team=current_team())
        try:
            info = await run_sync(st.repo.get_schema, ref)
        except LookupError as e:
            raise ToolError(
                f"Schema {ref} not found ({e}). List available schemas with laya_list_schemas, "
                "or save one with laya_save_schema."
            ) from e
    qd = questions_to_laya(info.questions)
    with tool_errors(st.settings.request_timeout_s):
        st.engine.check_sync_budget(len(states), len(qd), states)
        raw = await st.engine.predict(states, qd, model=model, lang=lang, timeout=st.settings.request_timeout_s)

    # Only a trusted schema (its evaluation met the team's accuracy targets) may decide on its own.
    # For draft/evaluated schemas every status is 'unverified'; `needs_review` still flags the
    # answers below the measured (evaluated) or default (draft) threshold.
    unverified = info.status != "trusted"
    results = [
        to_item_result(
            i,
            r,
            detail=detail,
            temperatures=info.temperatures,
            thresholds=info.thresholds,
            default_threshold=st.settings.default_threshold,
            unverified=unverified,
        )
        for i, r in enumerate(raw)
    ]
    answers, review = count_review(results)
    warnings: list[str] = []
    if info.status == "draft":
        warnings.append(
            f"Schema {info.ref} is a draft that has never been evaluated: statuses are 'unverified' and "
            f"`needs_review` uses the default threshold {st.settings.default_threshold}. Evaluate it "
            "(laya_evaluate) before relying on it."
        )
    elif info.status == "evaluated":
        warnings.append(
            f"Schema {info.ref} is evaluated but not trusted (it has not met its accuracy targets, or was "
            "never promoted): statuses are 'unverified'. Treat answers outside `needs_review` as strong "
            "suggestions, not decisions. See its report (laya_get_report) and laya_promote_schema."
        )
    missing = [q for q in qd if q not in info.thresholds]
    if missing and not unverified:
        warnings.append(
            f"No measured threshold for {', '.join(missing)}; using the default {st.settings.default_threshold}."
        )
    return DecideResponse(
        results=results,
        rows=len(states) * len(qd),
        latency_ms=sw.ms,
        warnings=warnings,
        schema_ref=info.ref,
        schema_status=info.status,
        decided=answers - review,
        needs_review=review,
    )


@server.tool(
    name="laya_apply_preset",
    description=(
        "Run one of Laya's ready-made question sets on one item (`state`) or several (`items`), with no question "
        "design needed:\n"
        "- triage: support ticket -> intent, is_urgent, frustration (0-3), refund_requested, churn_risk\n"
        "- email: inbound email -> category (override with `categories`), is_spam, is_phishing, urgency, needs_reply\n"
        "- guard: LLM input -> jailbreak, prompt_injection, sensitive_data, harm_severity, topic "
        "(to screen untrusted documents, prefer laya_scan_untrusted)\n"
        "- moderation: user post -> toxic, harassment, threat, spam, severity\n"
        "- router: request -> difficulty, domain, needs_tools, is_sensitive\n"
        "A plain-string state is wrapped in the field the preset reads (message/body/prompt/post/request). "
        "The full question text is in laya://presets/{name}. Rows = items x 4-5 questions; synchronous budget applies."
    ),
    tags={"decisions", "presets"},
    annotations={**READ_ONLY, "title": "Apply a Laya preset"},
)
async def laya_apply_preset(
    preset: Annotated[
        Literal["triage", "email", "guard", "moderation", "router"],
        Field(description="Which preset question set to run."),
    ],
    state: StateParam = None,
    items: ItemsParam = None,
    categories: Annotated[
        dict[str, str] | None,
        Field(description="email preset only: routing categories {label: description} replacing the defaults."),
    ] = None,
    model: ModelParam = None,
    lang: LangParam = None,
    detail: DetailParam = "compact",
    threshold: Annotated[
        float | None,
        Field(ge=0, le=1, description="Optional confidence threshold: answers below it get status 'needs_review'."),
    ] = None,
) -> ClassifyResponse:
    if categories is not None and preset != "email":
        raise ToolError("`categories` only applies to the 'email' preset.")
    if categories is not None and not categories:
        raise ToolError("`categories` is empty; pass at least one {label: description} or omit it.")
    field = PRESET_FIELDS[preset]
    states = [{field: s} if isinstance(s, str) else s for s in resolve_states(state, items)]
    qd = preset_questions(preset, categories)
    return await _classify(
        states, qd, model=model, lang=lang, detail=detail, threshold=threshold, validate=categories is not None
    )
