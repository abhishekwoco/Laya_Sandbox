"""Schema library tools: save, inspect, list and promote versioned question sets."""
from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from ..db.repo import NotFound
from ..engine.runtime import InvalidQuestions
from ..models import EvalReport, Questions, SchemaInfo, SchemaRef, SchemaSummary, SchemaTargets, questions_to_laya
from ..models.library import valid_name
from ..state import current_team, get_state
from ..workbench.hooks import effective_min_examples

server = FastMCP("laya-library")

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

WORKFLOW = (
    "Workflow: laya_validate_questions -> laya_save_schema (draft) -> laya_save_dataset (~50-200 labeled "
    "examples) -> laya_evaluate (evaluated, thresholds recommended) -> laya_calibrate -> laya_promote_schema "
    "(trusted). Only a trusted schema's thresholds are verified against its accuracy targets on labeled data, "
    "so only rely on laya_decide answers without reviewing them when the schema is trusted."
)

TeamParam = Annotated[
    str | None, Field(description="Team namespace. Omit to use your X-Laya-Team header (or the server default).")
]
SchemaParam = Annotated[
    str,
    Field(
        description="Schema reference: 'team/name' (latest version), 'team/name@3' (version 3), "
        "or a bare 'name' in your own team."
    ),
]


# --------------------------------------------------------------------------- shared helpers
def resolve_schema(schema: str, team: str | None = None) -> SchemaInfo:
    """Parse a schema reference and load it, turning lookup problems into actionable ToolErrors."""
    try:
        ref = SchemaRef.parse(schema, default_team=current_team(team))
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    try:
        return get_state().repo.get_schema(ref)
    except NotFound as exc:
        raise ToolError(
            f"schema {ref} not found. laya_list_schemas shows the saved schemas (all_teams=true for other teams); "
            "create one with laya_save_schema."
        ) from exc


def parse_name(name: str, team: str | None, kind: str) -> tuple[str, str]:
    """'name' or 'team/name' -> (team, name), validated. Versions are never part of a name."""
    raw = name.strip().lower()
    if "@" in raw:
        raise ToolError(f"{kind} names carry no version ('{raw}'); versions are assigned automatically")
    if "/" in raw:
        t, _, n = raw.partition("/")
    else:
        t, n = current_team(team), raw
    if not valid_name(t) or not valid_name(n):
        raise ToolError(
            f"invalid {kind} name {raw!r}: use lowercase letters, digits, '-', '_' or '.', starting with a letter or digit"
        )
    return t, n


def schema_ref_of(info: SchemaInfo) -> SchemaRef:
    return SchemaRef(team=info.team, name=info.name, version=info.version)


# --------------------------------------------------------------------------- response models
class SaveSchemaResult(BaseModel):
    schema_info: SchemaInfo = Field(description="The saved draft version")
    warnings: list[str] = Field(default_factory=list, description="Non-blocking issues found while validating the questions")
    next_step: str


class SchemaList(BaseModel):
    team: str | None = Field(description="Team listed, or null when listing all teams")
    schemas: list[SchemaSummary]


class PromoteResult(BaseModel):
    schema_info: SchemaInfo
    report_id: int
    message: str


# --------------------------------------------------------------------------- tools
@server.tool(
    name="laya_save_schema",
    description=(
        "Save a named question set (a 'schema') for your team so it can be evaluated, calibrated and then used "
        "by laya_decide. Every save creates a NEW DRAFT version (team/name@N); earlier versions, their thresholds "
        "and reports are kept. Questions use Laya's format: {id: {type: choice|score|noul, instructions, criteria}}. "
        "Questions are validated (no inference); problems that would make Laya fail are errors, softer issues come "
        "back as warnings. `targets` sets the accuracy each question must reach and the minimum labeled examples "
        "needed before promotion (defaults: server target accuracy, 50 examples). " + WORKFLOW
    ),
    tags={"library"},
    annotations=WRITE,
)
async def laya_save_schema(
    name: Annotated[str, Field(description="Schema name, e.g. 'issue-triage' (or 'team/issue-triage')")],
    questions: Annotated[Questions, Field(description="Question id -> question. Keep ids short and stable (e.g. 'team', 'severity')")],
    description: Annotated[str, Field(description="What the schema decides and where it is used")] = "",
    targets: Annotated[
        SchemaTargets | None,
        Field(description="Promotion targets: min_accuracy (0-1), min_examples, per_question accuracy overrides"),
    ] = None,
    team: TeamParam = None,
) -> SaveSchemaResult:
    st = get_state()
    t, n = parse_name(name, team, "schema")
    if not questions:
        raise ToolError("a schema needs at least one question")
    if len(questions) > st.settings.max_questions:
        raise ToolError(
            f"{len(questions)} questions exceeds the limit of {st.settings.max_questions} per schema; "
            "split it into several schemas"
        )
    if targets is None:
        targets = SchemaTargets(
            min_accuracy=st.settings.default_target_accuracy, min_examples=st.settings.min_eval_examples
        )
    unknown = sorted(set(targets.per_question) - set(questions))
    if unknown:
        raise ToolError(f"targets.per_question names unknown question ids: {', '.join(unknown)}")
    try:
        warnings = st.engine.validate(questions_to_laya(questions))
    except InvalidQuestions as exc:
        raise ToolError(f"invalid questions: {exc}") from exc
    info = st.repo.save_schema(t, n, questions, description, targets)
    min_n = effective_min_examples(info, st.settings)
    return SaveSchemaResult(
        schema_info=info,
        warnings=list(warnings),
        next_step=(
            f"Saved {info.ref} as a draft. Next: laya_save_dataset with at least {min_n} labeled examples "
            f"(100-200 recommended), then laya_evaluate(schema='{info.ref}', dataset=...)."
        ),
    )


@server.tool(
    name="laya_get_schema",
    description=(
        "Show one version of a saved schema: its questions, status (draft/evaluated/trusted), per-question "
        "confidence thresholds and calibration temperatures, promotion targets and the id of its latest "
        "evaluation report (read it with laya_get_report)."
    ),
    tags={"library"},
    annotations=READ_ONLY,
)
async def laya_get_schema(schema: SchemaParam, team: TeamParam = None) -> SchemaInfo:
    return resolve_schema(schema, team)


@server.tool(
    name="laya_list_schemas",
    description=(
        "List saved schemas (latest version of each) with status and question ids. Defaults to your team; "
        "all_teams=true lists every team's schemas so you can reuse a trusted one instead of writing your own."
    ),
    tags={"library"},
    annotations=READ_ONLY,
)
async def laya_list_schemas(
    team: TeamParam = None,
    all_teams: Annotated[bool, Field(description="List schemas of every team")] = False,
) -> SchemaList:
    repo = get_state().repo
    if all_teams:
        return SchemaList(team=None, schemas=repo.list_schemas(None))
    t = current_team(team)
    return SchemaList(team=t, schemas=repo.list_schemas(t))


@server.tool(
    name="laya_promote_schema",
    description=(
        "Mark a schema version TRUSTED: its confident laya_decide answers can then be used without review. "
        "Refuses unless that version's latest evaluation report meets its targets: every question reaches the "
        "target accuracy at its recommended threshold, with enough labeled examples. On refusal the error lists "
        "the failing questions and what to do next. Evaluate (and ideally calibrate) first."
    ),
    tags={"library"},
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
)
async def laya_promote_schema(schema: SchemaParam, team: TeamParam = None) -> PromoteResult:
    st = get_state()
    info = resolve_schema(schema, team)
    min_n = effective_min_examples(info, st.settings)
    if info.latest_report_id is None:
        raise ToolError(
            f"{info.ref} has not been evaluated. Save a dataset with at least {min_n} labeled examples "
            f"(laya_save_dataset), run laya_evaluate(schema='{info.ref}', dataset=...), then promote."
        )
    try:
        report = st.repo.get_report(info.latest_report_id)
    except NotFound as exc:
        raise ToolError(f"report {info.latest_report_id} of {info.ref} is missing; re-run laya_evaluate") from exc
    if info.status == "trusted":
        return PromoteResult(schema_info=info, report_id=report.id or info.latest_report_id, message=f"{info.ref} is already trusted")

    problems = promotion_problems(report, info, min_n)
    if problems:
        raise ToolError(
            f"cannot promote {info.ref}: report {info.latest_report_id} does not meet the targets.\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\nNext: if ECE is high, run laya_calibrate; look at the confusion matrix and worst_misses in "
            f"laya_get_report({info.latest_report_id}) to fix wrong labels or sharpen criteria descriptions "
            "(changing questions means laya_save_schema -> new version); add labeled examples with laya_save_dataset; "
            "then laya_evaluate again and retry."
        )
    promoted = st.repo.update_schema_version(schema_ref_of(info), status="trusted")
    return PromoteResult(
        schema_info=promoted,
        report_id=info.latest_report_id,
        message=(
            f"{promoted.ref} is trusted: its thresholds met the accuracy targets on labeled data. laya_decide "
            "returns answers at or above each question's threshold as 'decided' and flags the rest 'needs_review'."
        ),
    )


def promotion_problems(report: EvalReport, info: SchemaInfo, min_examples: int) -> list[str]:
    problems: list[str] = []
    if report.n_examples < min_examples:
        problems.append(f"only {report.n_examples} evaluated examples; need at least {min_examples}")
    for qid in info.questions:
        m = report.per_question.get(qid)
        target = info.targets.per_question.get(qid, report.target_accuracy)
        if m is None or m.n == 0:
            problems.append(f"{qid}: no labeled examples in the evaluated dataset")
            continue
        if not m.meets_target:
            problems.append(
                f"{qid}: accuracy {m.accuracy:.1%} (n={m.n}, ECE {m.ece:.3f}); no confidence threshold reaches "
                f"the {target:.0%} target with at least 10 answers"
            )
        if m.n < min_examples:
            problems.append(f"{qid}: only {m.n} labeled answers; need at least {min_examples}")
    if not problems and not report.passes_targets:
        problems.append("the report does not pass its targets (see its notes)")
    return problems


@server.tool(
    name="laya_get_report",
    description=(
        "Read an evaluation report. Per question: accuracy, confusion matrix (rows = expected label, columns = "
        "predicted, in `labels` order), per-label precision/recall/F1, score MAE, ECE (calibration error; lower "
        "is better, > 0.05 suggests laya_calibrate), accuracy by confidence band, the recommended confidence "
        "threshold with the accuracy and coverage it gives, and whether the target is met. worst_misses are the "
        "most confident wrong answers: check them for labeling mistakes or unclear criteria."
    ),
    tags={"library", "workbench"},
    annotations=READ_ONLY,
)
async def laya_get_report(
    report_id: Annotated[int, Field(description="Report id from laya_get_schema (latest_report_id) or a job's result_ref")],
) -> EvalReport:
    try:
        return get_state().repo.get_report(report_id)
    except NotFound as exc:
        raise ToolError(
            f"report {report_id} not found; laya_get_schema shows a schema's latest_report_id"
        ) from exc
