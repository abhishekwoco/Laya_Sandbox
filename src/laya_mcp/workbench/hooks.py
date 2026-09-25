"""Glue between evaluate jobs, reports and schema versions.

`register_hooks(jobs)` must be called once at startup, after the JobService exists:
when an 'evaluate' job completes, its raw results are scored against the dataset, the report is
saved and the schema version is updated (latest report, recommended thresholds, status).
`laya_calibrate` reuses `make_report` + `apply_report` so both paths follow the same rules.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..models import EvalReport, LabeledExample, SchemaInfo, SchemaRef
from ..state import AppState, get_state
from .evaluation import build_report

if TYPE_CHECKING:
    from ..db.repo import Repo
    from ..jobs.service import JobService

log = logging.getLogger(__name__)


class DatasetChanged(ValueError):
    """The dataset's states no longer line up with the evaluated job items."""


def effective_min_examples(schema: SchemaInfo, settings: Settings) -> int:
    """The server minimum is a floor; a schema's own targets can only raise the bar."""
    return max(int(schema.targets.min_examples), int(settings.min_eval_examples))


def split_dataset_ref(dataset: str) -> tuple[str, str]:
    team, _, name = dataset.partition("/")
    if not team or not name:
        raise ValueError(f"invalid dataset reference {dataset!r}; use team/name")
    return team, name


def dataset_fingerprint(examples: list[LabeledExample]) -> str:
    """Hash of the example STATES (labels may be corrected later without invalidating a job)."""
    h = hashlib.sha256()
    for ex in examples:
        h.update(json.dumps(ex.state, sort_keys=True, ensure_ascii=False, default=str).encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def evaluated_examples(repo: "Repo", meta: dict[str, Any]) -> list[LabeledExample]:
    """Dataset examples as they were when the evaluate job was submitted (first n_examples),
    refusing if the states were replaced since then."""
    team, name = split_dataset_ref(meta["dataset"])
    examples = repo.get_dataset_examples(team, name)
    n = meta.get("n_examples")
    if n is not None:
        if len(examples) < int(n):
            raise DatasetChanged(
                f"dataset {meta['dataset']} has {len(examples)} examples but the job evaluated {n}; "
                "it was replaced since. Re-run laya_evaluate."
            )
        examples = examples[: int(n)]
        fp = meta.get("fingerprint")
        if fp and dataset_fingerprint(examples) != fp:
            raise DatasetChanged(
                f"dataset {meta['dataset']} was replaced since this job ran (its states changed); re-run laya_evaluate."
            )
    return examples


def make_report(
    schema: SchemaInfo,
    examples: list[LabeledExample],
    raw_results: list[tuple[int, dict[str, Any]]],
    *,
    dataset: str,
    job_id: str | None,
    temperatures: dict[str, float],
    settings: Settings,
) -> EvalReport:
    return build_report(
        schema,
        examples,
        raw_results,
        dataset=dataset,
        job_id=job_id,
        temperatures=temperatures,
        target_accuracy=schema.targets.min_accuracy,
        min_examples=effective_min_examples(schema, settings),
    )


def apply_report(repo: "Repo", schema: SchemaInfo, report_id: int, report: EvalReport) -> SchemaInfo:
    """Link the report to its schema version, store the recommended thresholds and set status:
    'evaluated', except a trusted version stays trusted while its new report still passes."""
    thresholds = {
        qid: m.recommended_threshold for qid, m in report.per_question.items() if m.recommended_threshold is not None
    }
    keep_trusted = schema.status == "trusted" and report.passes_targets
    ref = SchemaRef(team=schema.team, name=schema.name, version=schema.version)
    return repo.update_schema_version(
        ref,
        status=None if keep_trusted else "evaluated",
        thresholds=thresholds,
        latest_report_id=report_id,
    )


def handle_evaluate_complete(job_id: str, state: AppState | None = None) -> int:
    """Build, save and apply the report of a finished evaluate job. Returns the report id."""
    st = state or get_state()
    jobs = st.jobs
    meta = jobs.meta(job_id)
    ref = SchemaRef.parse(meta["schema_ref"])
    schema = st.repo.get_schema(ref)
    examples = evaluated_examples(st.repo, meta)
    raw = jobs.raw_results(job_id)
    report = make_report(
        schema,
        examples,
        raw,
        dataset=meta["dataset"],
        job_id=job_id,
        temperatures=schema.temperatures,
        settings=st.settings,
    )
    report_id = st.repo.save_report(report)
    apply_report(st.repo, schema, report_id, report)
    jobs.set_result_ref(job_id, {"report_id": report_id, "passes_targets": report.passes_targets})
    return report_id


def _on_evaluate_complete(job_id: str) -> None:
    try:
        handle_evaluate_complete(job_id)
    except Exception as exc:
        # Surface the problem where the agent polls (result_ref), then re-raise so the JobService
        # logs it and records it as the job's error (it catches hook exceptions).
        log.exception("building the evaluation report for job %s failed", job_id)
        try:
            get_state().jobs.set_result_ref(job_id, {"report_error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            log.exception("could not record the report error on job %s", job_id)
        raise


def register_hooks(jobs: "JobService") -> None:
    jobs.on_complete("evaluate", _on_evaluate_complete)
