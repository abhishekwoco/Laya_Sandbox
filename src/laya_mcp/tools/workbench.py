"""Evaluation workbench tools: validate questions, manage labeled datasets, evaluate, calibrate,
compare checkpoints and pick items for humans to label."""
from __future__ import annotations

import math
import random
from typing import Annotated, Any, Literal

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, ValidationError

from ..db.repo import NotFound
from ..engine.answers import predicted_value, to_item_result
from ..engine.runtime import KNOWN_MODELS, BudgetExceeded, EngineBusy, InvalidQuestions
from ..models import (
    Answer,
    CalibrationResult,
    DatasetInfo,
    ItemResult,
    JobInfo,
    LabeledExample,
    Question,
    Questions,
    SchemaInfo,
    SchemaRef,
    State,
    questions_to_laya,
)
from ..models.library import ExpectedValue
from ..state import current_team, get_state
from ..workbench.calibration import fit_schema_temperatures
from ..workbench.evaluation import coerce_expected, label_of, question_labels
from ..workbench.hooks import (
    DatasetChanged,
    apply_report,
    dataset_fingerprint,
    effective_min_examples,
    evaluated_examples,
    make_report,
)
from .library import READ_ONLY, WORKFLOW, SchemaParam, TeamParam, parse_name, resolve_schema, schema_ref_of

server = FastMCP("laya-workbench")

MIN_CALIBRATION_EXAMPLES = 20

DatasetParam = Annotated[
    str, Field(description="Dataset reference: 'team/name', or a bare 'name' in your own team")
]


# --------------------------------------------------------------------------- response models
class QuestionValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list, description="Must be fixed before the questions can be used")
    warnings: list[str] = Field(default_factory=list, description="Likely to hurt accuracy or cost; worth fixing")
    rows_per_state: int = Field(description="Rows each state costs (one per question)")
    est_seconds_per_state: float = Field(description="Estimated inference time for one state at current load")


class SaveDatasetResult(BaseModel):
    dataset: DatasetInfo
    added: int
    warnings: list[str] = Field(default_factory=list)
    next_step: str


class DatasetList(BaseModel):
    team: str | None = Field(description="Team listed, or null when listing all teams")
    datasets: list[DatasetInfo]


class EvaluateStarted(BaseModel):
    job: JobInfo
    examples: int = Field(description="Examples submitted for evaluation")
    warnings: list[str] = Field(default_factory=list)
    next_step: str


class Disagreement(BaseModel):
    question_id: str
    values: dict[str, ExpectedValue] = Field(description="checkpoint -> answer (choice label, score level, noul bool)")
    confidences: dict[str, float] = Field(description="checkpoint -> probability of its answer (uncalibrated)")


class ModelComparison(BaseModel):
    models: dict[str, ItemResult] = Field(description="checkpoint -> its answers (uncalibrated, as shipped)")
    disagreements: list[Disagreement]
    agree: bool = Field(description="True when every checkpoint gave the same answer to every question")
    rows: int


class LabelingItem(BaseModel):
    index: int = Field(description="Item index in the job (same order as submitted)")
    state: State | None = Field(description="The item's content; null if the server cannot recover it (use the index)")
    reason: Literal["uncertain", "random"]
    min_confidence: float = Field(description="Lowest answer confidence of this item")
    needs_review: list[str] = Field(default_factory=list)
    answers: dict[str, Answer]
    suggested_expected: dict[str, ExpectedValue] = Field(
        description="Current answers in dataset format; have a human correct them, then save as `expected`"
    )


class LabelingSample(BaseModel):
    job_id: str
    strategy: str
    total_items: int = Field(description="Finished items the sample was drawn from")
    items: list[LabelingItem]
    next_step: str


# --------------------------------------------------------------------------- helpers
def _dataset_ref(dataset: str, team: str | None) -> tuple[str, str]:
    return parse_name(dataset, team, "dataset")


def _engine_error(exc: Exception) -> ToolError:
    if isinstance(exc, InvalidQuestions):
        return ToolError(f"invalid questions: {exc}")
    if isinstance(exc, EngineBusy):
        return ToolError(str(exc))
    if isinstance(exc, TimeoutError):
        return ToolError("Laya timed out on this request; retry with fewer questions or a shorter state")
    return ToolError(str(exc))


def _format_validation_error(qid: str, exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "question"
        out.append(f"question {qid!r}: {loc}: {err.get('msg')}")
    return out


def _load_job(job_id: str) -> JobInfo:
    try:
        return get_state().jobs.get(job_id)
    except NotFound as exc:
        raise ToolError(f"job {job_id!r} not found; laya_evaluate and laya_classify_batch return job ids") from exc


def _job_states(job_id: str, meta: dict[str, Any]) -> dict[int, State]:
    """Item states of a job. Uses JobService.states(job_id) when the service offers it; evaluate
    jobs fall back to their dataset."""
    st = get_state()
    getter = getattr(st.jobs, "states", None)
    if callable(getter):
        try:
            return {int(i): s for i, s in dict(getter(job_id)).items()}
        except Exception:
            pass
    if meta.get("dataset"):
        try:
            examples = evaluated_examples(st.repo, meta)
        except (NotFound, DatasetChanged, ValueError):
            return {}
        return {i: ex.state for i, ex in enumerate(examples)}
    return {}


def _check_expected(schema: SchemaInfo, examples: list[LabeledExample]) -> list[str]:
    """Warnings about labels that the evaluation would have to skip."""
    warnings: list[str] = []
    for qid, q in schema.questions.items():
        labels = question_labels(q)
        labeled = [ex.expected[qid] for ex in examples if qid in ex.expected]
        if not labeled:
            warnings.append(f"{qid}: no example is labeled for this question; it will fail its target")
            continue
        bad = []
        for v in labeled:
            try:
                coerce_expected(q, labels, v)
            except ValueError:
                bad.append(v)
        if bad:
            warnings.append(
                f"{qid}: {len(bad)} expected value(s) are not valid answers and will be skipped "
                f"(e.g. {bad[0]!r}); valid: {', '.join(labels) if q.type != 'noul' else 'true/false'}"
            )
    return warnings


# --------------------------------------------------------------------------- tools
@server.tool(
    name="laya_validate_questions",
    description=(
        "Check a question set before using or saving it: structure, option/token budget and quality warnings "
        "(too many labels, overlapping criteria, missing descriptions). Runs NO inference, so it is instant. "
        "Also reports the cost: every question is one row per state, so fewer, sharper questions are faster. "
        "Question format: {id: {type: 'choice'|'score'|'noul', instructions: '...', criteria: ...}} where choice "
        "criteria are {label: description}, score criteria are an ordered list of level descriptions (lowest "
        "first) and noul (yes/no) criteria are optional {'true': ..., 'false': ...}."
    ),
    tags={"workbench"},
    annotations=READ_ONLY,
)
async def laya_validate_questions(
    questions: Annotated[
        dict[str, Any], Field(description="Question id -> {type, instructions, criteria} in Laya's format")
    ],
) -> QuestionValidation:
    st = get_state()
    errors: list[str] = []
    parsed: dict[str, Question] = {}
    if not questions:
        errors.append("no questions given")
    if len(questions) > st.settings.max_questions:
        errors.append(
            f"{len(questions)} questions exceeds the limit of {st.settings.max_questions}; split them into several schemas"
        )
    for qid, raw in questions.items():
        if not str(qid).strip():
            errors.append("question ids must be non-empty")
            continue
        try:
            parsed[qid] = Question.model_validate(raw)
        except ValidationError as exc:
            errors.extend(_format_validation_error(qid, exc))

    warnings: list[str] = []
    sound: dict[str, Question] = {}
    for qid, q in parsed.items():  # per question, so every broken question is reported at once
        try:
            st.engine.validate(questions_to_laya({qid: q}))
            sound[qid] = q
        except InvalidQuestions as exc:
            errors.append(str(exc))
    if sound:  # set-level warnings (overlaps, budget) for the questions that are valid on their own
        try:
            warnings = list(st.engine.validate(questions_to_laya(sound)))
        except InvalidQuestions as exc:
            errors.append(str(exc))
    rows = len(questions)
    return QuestionValidation(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        rows_per_state=rows,
        est_seconds_per_state=round(float(st.engine.estimate_seconds(rows)), 2),
    )


@server.tool(
    name="laya_save_dataset",
    description=(
        "Store labeled examples on the server for laya_evaluate: [{state, expected: {question_id: answer}}]. "
        "expected answers are a choice label, a score level index (0 = lowest level) or true/false for noul "
        "questions; an example may label only some questions. Send at most 500 examples per call and add the "
        "rest with more calls (append=true, the default); append=false replaces the whole dataset. Aim for "
        "50-200 examples that look like real traffic, including hard cases. " + WORKFLOW
    ),
    tags={"workbench"},
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False),
)
async def laya_save_dataset(
    name: Annotated[str, Field(description="Dataset name, e.g. 'issues-2026q3' (or 'team/issues-2026q3')")],
    examples: Annotated[list[LabeledExample], Field(description="Labeled examples: {state, expected}")],
    append: Annotated[bool, Field(description="Add to the existing examples (true) or replace them (false)")] = True,
    team: TeamParam = None,
) -> SaveDatasetResult:
    st = get_state()
    t, n = _dataset_ref(name, team)
    limit = st.settings.max_items_per_call
    if not examples:
        raise ToolError("no examples given")
    if len(examples) > limit:
        raise ToolError(
            f"{len(examples)} examples in one call exceeds the limit of {limit}; send them in chunks of {limit} "
            "with append=true"
        )
    problems = []
    for i, ex in enumerate(examples):
        if not ex.expected:
            problems.append(f"example {i}: `expected` is empty; label at least one question")
        elif any(not str(k).strip() for k in ex.expected):
            problems.append(f"example {i}: `expected` has an empty question id")
        if ex.state is None or (isinstance(ex.state, str) and not ex.state.strip()) or ex.state in ({}, []):
            problems.append(f"example {i}: `state` is empty")
    if problems:
        shown = problems[:10]
        more = f"\n... and {len(problems) - 10} more" if len(problems) > 10 else ""
        raise ToolError("nothing was saved:\n" + "\n".join(shown) + more)

    info = st.repo.save_dataset(t, n, examples, append=append)
    warnings = []
    if info.count < st.settings.min_eval_examples:
        warnings.append(
            f"{info.count} examples so far; promotion needs at least {st.settings.min_eval_examples} "
            "(100-200 recommended)"
        )
    return SaveDatasetResult(
        dataset=info,
        added=len(examples),
        warnings=warnings,
        next_step=f"Run laya_evaluate(schema=..., dataset='{info.team}/{info.name}') once all examples are saved.",
    )


@server.tool(
    name="laya_list_datasets",
    description="List labeled datasets with their size and which question ids they label. Defaults to your team.",
    tags={"workbench"},
    annotations=READ_ONLY,
)
async def laya_list_datasets(
    team: TeamParam = None,
    all_teams: Annotated[bool, Field(description="List datasets of every team")] = False,
) -> DatasetList:
    repo = get_state().repo
    if all_teams:
        return DatasetList(team=None, datasets=repo.list_datasets(None))
    t = current_team(team)
    return DatasetList(team=t, datasets=repo.list_datasets(t))


@server.tool(
    name="laya_evaluate",
    description=(
        "Measure how well a schema version answers a labeled dataset. Starts an asynchronous job that runs every "
        "example through Laya (roughly 0.5-1 s per question per example; the job reports an ETA). Poll "
        "laya_job_status(job_id); "
        "when it is completed, result_ref.report_id holds the report (laya_get_report): per-question accuracy, "
        "confusion matrix, precision/recall, ECE, accuracy by confidence band and a recommended confidence "
        "threshold for the target accuracy. The schema version becomes 'evaluated' and stores those thresholds. "
        "Next: laya_calibrate, then laya_promote_schema."
    ),
    tags={"workbench"},
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False),
)
async def laya_evaluate(
    schema: SchemaParam,
    dataset: DatasetParam,
    team: TeamParam = None,
) -> EvaluateStarted:
    st = get_state()
    info = resolve_schema(schema, team)
    dt, dn = _dataset_ref(dataset, team)
    try:
        examples = st.repo.get_dataset_examples(dt, dn)
    except NotFound as exc:
        raise ToolError(f"dataset {dt}/{dn} not found; create it with laya_save_dataset (laya_list_datasets lists them)") from exc
    if not examples:
        raise ToolError(f"dataset {dt}/{dn} is empty; add labeled examples with laya_save_dataset")
    labeled = {k for ex in examples for k in ex.expected}
    if not labeled & set(info.questions):
        raise ToolError(
            f"dataset {dt}/{dn} labels {sorted(labeled)} but {info.ref} asks {sorted(info.questions)}; "
            "the `expected` keys must be the schema's question ids"
        )
    warnings = _check_expected(info, examples)
    min_n = effective_min_examples(info, st.settings)
    if len(examples) < min_n:
        warnings.append(f"{len(examples)} examples; the report cannot pass promotion below {min_n}")

    meta = {
        "schema_ref": info.ref,
        "dataset": f"{dt}/{dn}",
        "n_examples": len(examples),
        "fingerprint": dataset_fingerprint(examples),
    }
    team_name = current_team(team)
    try:
        job = await anyio.to_thread.run_sync(
            lambda: st.jobs.submit(
                team=team_name,
                kind="evaluate",
                items=[ex.state for ex in examples],
                questions=questions_to_laya(info.questions),
                schema_ref=info.ref,
                meta=meta,
            )
        )
    except (InvalidQuestions, EngineBusy) as exc:
        raise _engine_error(exc) from exc
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    eta = f" (ETA ~{job.eta_seconds:.0f}s)" if job.eta_seconds else ""
    return EvaluateStarted(
        job=job,
        examples=len(examples),
        warnings=warnings,
        next_step=(
            f"Poll laya_job_status(job_id='{job.job_id}'){eta}. When completed, result_ref.report_id is the report "
            f"id: read it with laya_get_report, then run laya_calibrate(schema='{info.ref}')."
        ),
    )


@server.tool(
    name="laya_calibrate",
    description=(
        "Fix over- or under-confidence of a schema version: fits one temperature per question on 70% of the "
        "evaluated examples and reports ECE before/after on the held-out 30%. The temperatures are stored on the "
        "version (laya_decide applies them), and the report is rebuilt on all examples with calibration applied, "
        "which refreshes the recommended thresholds and returns its report_id. Uses the evaluate job behind the "
        "version's latest report unless job_id is given. Needs a completed laya_evaluate first (20+ examples). "
        "Next: laya_get_report(report_id), then laya_promote_schema if every question meets its target."
    ),
    tags={"workbench"},
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
)
async def laya_calibrate(
    schema: SchemaParam,
    job_id: Annotated[
        str | None, Field(description="Completed evaluate job to fit on; default = the job of the latest report")
    ] = None,
    team: TeamParam = None,
) -> CalibrationResult:
    st = get_state()
    info = resolve_schema(schema, team)
    if job_id is None:
        if info.latest_report_id is None:
            raise ToolError(f"{info.ref} has no evaluation yet; run laya_evaluate(schema='{info.ref}', dataset=...) first")
        try:
            job_id = st.repo.get_report(info.latest_report_id).job_id
        except NotFound as exc:
            raise ToolError(f"report {info.latest_report_id} is missing; re-run laya_evaluate") from exc
        if job_id is None:
            raise ToolError(f"the latest report of {info.ref} has no job behind it; re-run laya_evaluate")
    job = _load_job(job_id)
    if job.kind != "evaluate":
        raise ToolError(f"job {job_id} is a {job.kind} job; calibration needs a laya_evaluate job (it has labels)")
    if job.status != "completed":
        raise ToolError(f"job {job_id} is {job.status}; wait until laya_job_status shows completed")
    meta = st.jobs.meta(job_id)
    if not meta.get("schema_ref") or not meta.get("dataset"):
        raise ToolError(f"job {job_id} has no schema/dataset metadata; re-run laya_evaluate")

    job_ref = SchemaRef.parse(meta["schema_ref"])
    if (job_ref.team, job_ref.name) != (info.team, info.name):
        raise ToolError(f"job {job_id} evaluated {meta['schema_ref']}, not {info.team}/{info.name}")
    if job_ref.version != info.version:
        try:
            job_schema = st.repo.get_schema(job_ref)
        except NotFound as exc:
            raise ToolError(f"{meta['schema_ref']} no longer exists; re-run laya_evaluate for {info.ref}") from exc
        if questions_to_laya(job_schema.questions) != questions_to_laya(info.questions):
            raise ToolError(
                f"job {job_id} evaluated {meta['schema_ref']}, whose questions differ from {info.ref}; "
                f"run laya_evaluate(schema='{info.ref}', dataset='{meta['dataset']}') first"
            )

    try:
        examples = evaluated_examples(st.repo, meta)
    except NotFound as exc:
        raise ToolError(f"dataset {meta['dataset']} no longer exists; re-run laya_evaluate with a dataset") from exc
    except DatasetChanged as exc:
        raise ToolError(str(exc)) from exc
    raw = st.jobs.raw_results(job_id)
    usable = sum(1 for i, _ in raw if 0 <= int(i) < len(examples))
    if usable < MIN_CALIBRATION_EXAMPLES:
        raise ToolError(
            f"only {usable} evaluated examples; calibration needs at least {MIN_CALIBRATION_EXAMPLES} "
            "(50+ recommended). Add labeled examples with laya_save_dataset and re-run laya_evaluate."
        )

    def work() -> tuple[CalibrationResult, int]:
        temps, before, after, n_fit, n_hold = fit_schema_temperatures(info, examples, raw)
        updated = st.repo.update_schema_version(schema_ref_of(info), temperatures=temps)
        report = make_report(
            updated, examples, raw, dataset=meta["dataset"], job_id=job_id, temperatures=temps, settings=st.settings
        )
        report_id = st.repo.save_report(report)
        apply_report(st.repo, updated, report_id, report)
        return (
            CalibrationResult(
                schema_ref=info.ref,
                temperatures=temps,
                ece_before=before,
                ece_after=after,
                n_fit=n_fit,
                n_holdout=n_hold,
                report_id=report_id,
            ),
            report_id,
        )

    result, _ = await anyio.to_thread.run_sync(work)
    return result


@server.tool(
    name="laya_compare_models",
    description=(
        "Run one state through every Laya checkpoint (english, multilingual, typed-decisions) with the same "
        "questions and list the questions where they disagree. Use it to pick a checkpoint for a schema or to "
        "see whether a hard case is ambiguous. Pass either `questions` or a saved `schema`. Confidences are "
        "uncalibrated; costs 3 x questions rows."
    ),
    tags={"workbench"},
    annotations=READ_ONLY,
)
async def laya_compare_models(
    state: Annotated[State, Field(description="Content to judge: text, a JSON object or a list of turns. Keep it concise.")],
    questions: Annotated[Questions | None, Field(description="Questions in Laya's format (or pass `schema`)")] = None,
    schema: Annotated[str | None, Field(description="Saved schema reference to take the questions from")] = None,
    team: TeamParam = None,
) -> ModelComparison:
    st = get_state()
    if (questions is None) == (schema is None):
        raise ToolError("pass exactly one of `questions` or `schema`")
    if schema is not None:
        questions = resolve_schema(schema, team).questions
    assert questions is not None
    if not questions:
        raise ToolError("no questions given")
    laya_q = questions_to_laya(questions)
    try:
        st.engine.check_sync_budget(len(KNOWN_MODELS), len(laya_q), [state])
    except BudgetExceeded as exc:
        raise ToolError(f"laya_compare_models runs each question on {len(KNOWN_MODELS)} checkpoints: {exc}") from exc
    try:
        raw_by_model = await st.engine.predict_all_models(state, laya_q)
    except (InvalidQuestions, EngineBusy, TimeoutError, ValueError) as exc:
        raise _engine_error(exc) from exc

    models = {m: to_item_result(0, raw) for m, raw in raw_by_model.items()}
    disagreements = []
    for qid in laya_q:
        values: dict[str, ExpectedValue] = {}
        confs: dict[str, float] = {}
        for m, raw in raw_by_model.items():
            ans = (raw.get("answers") or {}).get(qid)
            if ans is None:
                continue
            values[m] = predicted_value(ans)
            confs[m] = models[m].answers[qid].confidence
        if len({label_of(v) for v in values.values()}) > 1:
            disagreements.append(Disagreement(question_id=qid, values=values, confidences=confs))
    return ModelComparison(
        models=models,
        disagreements=disagreements,
        agree=not disagreements,
        rows=len(raw_by_model) * len(laya_q),
    )


@server.tool(
    name="laya_sample_for_labeling",
    description=(
        "Pick items from a batch or evaluate job for a human to label, to grow an evaluation dataset where Laya "
        "struggles. strategy='uncertain' takes the lowest-confidence items, 'random' a uniform sample (keeps the "
        "dataset representative), 'mixed' (default) half of each. Each item comes with its current answers and "
        "`suggested_expected` pre-filled in dataset format: have a human correct them, then send "
        "[{state, expected}] to laya_save_dataset (append=true) and re-run laya_evaluate."
    ),
    tags={"workbench"},
    annotations=READ_ONLY,
)
async def laya_sample_for_labeling(
    job_id: Annotated[str, Field(description="Job id from laya_classify_batch or laya_evaluate")],
    n: Annotated[int, Field(ge=1, le=200, description="How many items to return")] = 20,
    strategy: Annotated[
        Literal["mixed", "uncertain", "random"], Field(description="How to pick items")
    ] = "mixed",
    seed: Annotated[int | None, Field(description="Random seed; the default is stable per job")] = None,
) -> LabelingSample:
    st = get_state()
    job = _load_job(job_id)
    meta = st.jobs.meta(job_id)
    raw = st.jobs.raw_results(job_id)
    if not raw:
        raise ToolError(f"job {job_id} has no finished items yet ({job.status}); check laya_job_status")

    temperatures: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    if job.schema_ref:
        try:
            sch = st.repo.get_schema(SchemaRef.parse(job.schema_ref))
            temperatures, thresholds = sch.temperatures, sch.thresholds
        except (NotFound, ValueError):
            pass
    results = {
        int(i): to_item_result(
            int(i), r, temperatures=temperatures, thresholds=thresholds, default_threshold=st.settings.default_threshold
        )
        for i, r in raw
    }
    min_conf = {i: min((a.confidence for a in res.answers.values()), default=1.0) for i, res in results.items()}
    rng = random.Random(seed if seed is not None else job_id)

    order = sorted(min_conf, key=lambda i: (min_conf[i], i))
    n = min(n, len(order))
    picked: list[tuple[int, Literal["uncertain", "random"]]] = []
    if strategy == "uncertain":
        picked = [(i, "uncertain") for i in order[:n]]
    elif strategy == "random":
        picked = [(i, "random") for i in sorted(rng.sample(order, n))]
    else:
        k = math.ceil(n / 2)
        unc = order[:k]
        rest = order[k:]
        picked = [(i, "uncertain") for i in unc] + [(i, "random") for i in sorted(rng.sample(rest, min(n - k, len(rest))))]

    states = _job_states(job_id, meta)
    raw_by_index = {int(i): r for i, r in raw}
    items = []
    for i, reason in picked:
        res = results[i]
        suggested = {
            qid: predicted_value(ans, temperatures.get(qid, 1.0))
            for qid, ans in (raw_by_index[i].get("answers") or {}).items()
        }
        items.append(
            LabelingItem(
                index=i,
                state=states.get(i),
                reason=reason,
                min_confidence=round(min_conf[i], 4),
                needs_review=res.needs_review,
                answers=res.answers,
                suggested_expected=suggested,
            )
        )
    note = "" if states else " States could not be recovered for this job: match items to your inputs by index."
    return LabelingSample(
        job_id=job_id,
        strategy=strategy,
        total_items=len(results),
        items=items,
        next_step=(
            "Have a human confirm or correct suggested_expected for each item, then call laya_save_dataset with "
            "[{state, expected}] (append=true) and re-run laya_evaluate." + note
        ),
    )
