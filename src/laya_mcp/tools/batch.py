"""Async batch tools: laya_classify_batch, laya_job_status, laya_job_results, laya_job_cancel."""
from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ..middleware import note_usage
from ..models import Detail, JobInfo, JobResultsPage, Questions, SchemaInfo, SchemaRef, State, questions_to_laya
from ..state import current_team, get_state
from ._core import READ_ONLY, ModelName, run_sync, state_chars, tool_errors

server = FastMCP("laya-batch")

JobIdParam = Annotated[str, Field(min_length=1, description="Job id returned by laya_classify_batch or laya_evaluate.")]
MAX_PAGE = 100


async def _load_schema(ref_text: str) -> SchemaInfo:
    st = get_state()
    ref = SchemaRef.parse(ref_text, default_team=current_team())
    try:
        return await run_sync(st.repo.get_schema, ref)
    except LookupError as e:
        raise ToolError(f"Schema {ref} not found ({e}). List schemas with laya_list_schemas.") from e


def _with_eta(job: JobInfo, questions: int | None = None) -> JobInfo:
    """Fill eta_seconds from the engine's rolling cost when the job service left it empty."""
    if job.eta_seconds is None and job.status in ("queued", "running") and questions:
        remaining = max(job.total - job.done - job.failed, 0)
        try:
            job = job.model_copy(update={"eta_seconds": round(get_state().engine.estimate_seconds(remaining * questions), 1)})
        except Exception:
            pass
    return job


@server.tool(
    name="laya_classify_batch",
    description=(
        "Queue a LARGE classification job (anything above the synchronous budget of laya_classify / laya_decide): "
        "up to 500 items per call, answered in the background by the same model. Returns a job_id and an ETA "
        "(~0.5-1.6 s per row depending on input length, rows = items x questions).\n\n"
        "Give either `questions` (same format as laya_classify) or `schema` (a saved schema: its thresholds and "
        "calibration are applied when you read results). For more than 500 items, send them in chunks: the first "
        "call with finalize=false returns a job_id; later calls pass that `job_id` (questions/schema may be "
        "omitted, the job's own are reused) and the last one sets finalize=true.\n\n"
        "Then poll laya_job_status, and read answers with laya_job_results(needs_review_only=true) to see only "
        "the items you must decide yourself. Keep each item concise (pre-extracted fields, not whole files)."
    ),
    tags={"batch"},
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False,
                 "title": "Queue a batch job"},
)
async def laya_classify_batch(
    items: Annotated[
        list[State],
        Field(min_length=1, description="Items to classify (strings or JSON objects), at most 500 per call."),
    ],
    questions: Annotated[
        Questions | None,
        Field(description="Question set (same format as laya_classify). Give this or `schema`, not both."),
    ] = None,
    schema: Annotated[
        str | None,
        Field(description="Saved schema reference ('team/name[@version]' or bare name). Give this or `questions`."),
    ] = None,
    model: Annotated[ModelName | None, Field(description="Force a checkpoint; leave unset to auto-route per item.")] = None,
    lang: Annotated[str | None, Field(description="Optional ISO 639-1 language hint for routing.")] = None,
    job_id: Annotated[
        str | None,
        Field(description="Append these items to an open job (created with finalize=false) instead of starting a new one."),
    ] = None,
    finalize: Annotated[
        bool,
        Field(description="true (default): no more items will be added. false: keep the job open for more chunks."),
    ] = True,
) -> JobInfo:
    st = get_state()
    s = st.settings
    if len(items) > s.max_items_per_call:
        raise ToolError(
            f"{len(items)} items exceeds {s.max_items_per_call} per call; send the first chunk with "
            "finalize=false, then append the rest with job_id=<returned id>."
        )
    if questions is not None and schema is not None:
        raise ToolError("Pass either `questions` or `schema`, not both.")
    too_long = [i for i, it in enumerate(items) if state_chars(it) > s.max_state_chars]
    if too_long:
        raise ToolError(
            f"items {too_long[:10]} exceed {s.max_state_chars} characters; send pre-extracted content "
            "(the model reads ~512 tokens anyway)."
        )

    schema_ref: str | None = None
    qd: dict[str, dict[str, Any]] = {}
    with tool_errors():
        if schema is not None:
            info = await _load_schema(schema)
            schema_ref, qd = info.ref, questions_to_laya(info.questions)
        elif questions is not None:
            qd = questions_to_laya(questions)
        elif job_id is not None:
            # Appending with nothing given: JobService answers appended items with the job's own questions.
            existing = await run_sync(st.jobs.get, job_id)
            schema_ref = existing.schema_ref
        else:
            raise ToolError("Pass `questions` or `schema` (or `job_id` to append to an existing job).")

        if qd:
            if len(qd) > s.max_questions:
                raise ToolError(f"{len(qd)} questions exceeds the limit of {s.max_questions}; split the question set.")
            st.engine.validate(qd, model or "english")

        job = await run_sync(
            st.jobs.submit,
            team=current_team(),
            kind="batch",
            items=list(items),
            questions=qd,
            schema_ref=schema_ref,
            model=model,
            lang=lang,
            append_to=job_id,
            finalize=finalize,
        )
    n_questions = len(qd)
    if not n_questions and job_id is not None:
        # Appended with the job's own questions: count them for usage and the ETA.
        get_job = getattr(st.repo, "get_job", None)
        if get_job is not None:
            n_questions = len((await run_sync(get_job, job_id)).questions or {})
    note_usage(rows=len(items) * n_questions, schema_ref=schema_ref)
    return _with_eta(job, n_questions)


@server.tool(
    name="laya_job_status",
    description=(
        "Progress of a batch or evaluation job: status (queued/running/completed/failed/cancelled), items done "
        "out of total, failures and an ETA in seconds. Poll this at a relaxed pace (every 15-60 s depending on "
        "the ETA) rather than in a tight loop; when completed read answers with laya_job_results (evaluate jobs "
        "link their report in result_ref)."
    ),
    tags={"batch"},
    annotations={**READ_ONLY, "title": "Job status"},
)
async def laya_job_status(job_id: JobIdParam) -> JobInfo:
    st = get_state()
    with tool_errors():
        job = await run_sync(st.jobs.get, job_id)
    return job


@server.tool(
    name="laya_job_results",
    description=(
        "Read a batch job's answers, paginated (limit <= 100; follow `next_offset`), plus a whole-job `summary` "
        "of answer counts per question. Works while the job is still running (completed items only).\n\n"
        "Filters apply before pagination: needs_review_only=true returns just the items with at least one answer "
        "below its threshold - those are the ones YOU must decide; everything else can be taken as-is. "
        "question_id + value select items whose answer to that question equals value (noul: 'true'/'false', "
        "score: level number). Jobs created with a schema use its thresholds and calibration; plain-question "
        "jobs use `threshold` (default: the server default, 0.8)."
    ),
    tags={"batch"},
    annotations={**READ_ONLY, "title": "Job results"},
)
async def laya_job_results(
    job_id: JobIdParam,
    offset: Annotated[int, Field(ge=0, description="Index of the first result to return.")] = 0,
    limit: Annotated[int, Field(ge=1, le=MAX_PAGE, description="Page size (max 100).")] = 50,
    needs_review_only: Annotated[bool, Field(description="Only items with at least one answer below threshold.")] = False,
    question_id: Annotated[str | None, Field(description="Filter on this question's answer (use with `value`).")] = None,
    value: Annotated[
        str | None,
        Field(description="Answer value to match for `question_id`: a choice label, 'true'/'false', or a score level."),
    ] = None,
    detail: Annotated[Detail, Field(description="'compact' or 'full' (adds probabilities).")] = "compact",
    threshold: Annotated[
        float | None,
        Field(ge=0, le=1, description="Confidence threshold for plain-question jobs (schema jobs use the schema's)."),
    ] = None,
) -> JobResultsPage:
    st = get_state()
    if value is not None and question_id is None:
        raise ToolError("`value` needs `question_id` (which question's answer to compare).")
    with tool_errors():
        job = await run_sync(st.jobs.get, job_id)
        temperatures: dict[str, float] | None = None
        thresholds: dict[str, float] | None = None
        default_threshold = threshold if threshold is not None else st.settings.default_threshold
        if job.schema_ref:
            info = await _load_schema(job.schema_ref)
            temperatures, thresholds = info.temperatures, info.thresholds
        page = await run_sync(
            st.jobs.results,
            job_id,
            offset=offset,
            limit=limit,
            needs_review_only=needs_review_only,
            question_id=question_id,
            value=value.lower() if value is not None else None,
            detail=detail,
            temperatures=temperatures,
            thresholds=thresholds,
            default_threshold=default_threshold,
        )
    return page


@server.tool(
    name="laya_job_cancel",
    description=(
        "Cancel a queued or running job: items not yet processed are dropped, results already computed stay "
        "readable with laya_job_results. Returns the job's final status."
    ),
    tags={"batch"},
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False,
                 "title": "Cancel a job"},
)
async def laya_job_cancel(job_id: JobIdParam) -> JobInfo:
    st = get_state()
    with tool_errors():
        job = await run_sync(st.jobs.cancel, job_id)
    return job
