"""Minimal in-memory stand-ins for Repo and JobService, implementing only what the library and
workbench tools call. Used while the real SQLModel repo / Huey job service are built; they follow
the contracts in db/repo.py and jobs/service.py. FakeJobs runs a job synchronously inside submit()
and then fires the on_complete hooks, like the real worker thread would."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from laya_mcp.db.repo import NotFound
from laya_mcp.models import (
    DatasetInfo,
    EvalReport,
    JobInfo,
    LabeledExample,
    Questions,
    SchemaInfo,
    SchemaRef,
    SchemaSummary,
    SchemaTargets,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeRepo:
    def __init__(self) -> None:
        self.schemas: dict[tuple[str, str], list[SchemaInfo]] = {}
        self.datasets: dict[tuple[str, str], dict[str, Any]] = {}
        self.reports: dict[int, EvalReport] = {}

    # schemas
    def save_schema(self, team: str, name: str, questions: Questions, description: str = "",
                    targets: SchemaTargets | None = None) -> SchemaInfo:
        versions = self.schemas.setdefault((team, name), [])
        info = SchemaInfo(
            team=team, name=name, version=len(versions) + 1, status="draft", description=description,
            questions=questions, targets=targets or SchemaTargets(), created_at=_now(),
        )
        versions.append(info)
        return info.model_copy(deep=True)

    def _find(self, ref: SchemaRef) -> tuple[list[SchemaInfo], int]:
        versions = self.schemas.get((ref.team, ref.name))
        if not versions:
            raise NotFound(f"schema {ref.team}/{ref.name} not found")
        v = ref.version or len(versions)
        if not 1 <= v <= len(versions):
            raise NotFound(f"schema {ref} not found")
        return versions, v - 1

    def get_schema(self, ref: SchemaRef) -> SchemaInfo:
        versions, i = self._find(ref)
        return versions[i].model_copy(deep=True)

    def list_schemas(self, team: str | None = None) -> list[SchemaSummary]:
        out = []
        for (t, n), versions in sorted(self.schemas.items()):
            if team is not None and t != team:
                continue
            last = versions[-1]
            out.append(SchemaSummary(
                ref=last.ref, status=last.status, description=last.description,
                question_ids=list(last.questions), versions=len(versions), updated_at=last.created_at,
            ))
        return out

    def update_schema_version(self, ref: SchemaRef, *, status=None, thresholds=None, temperatures=None,
                              latest_report_id=None) -> SchemaInfo:
        versions, i = self._find(ref)
        info = versions[i]
        upd: dict[str, Any] = {}
        if status is not None:
            upd["status"] = status
        if thresholds is not None:
            upd["thresholds"] = dict(thresholds)
        if temperatures is not None:
            upd["temperatures"] = dict(temperatures)
        if latest_report_id is not None:
            upd["latest_report_id"] = latest_report_id
        versions[i] = info.model_copy(update=upd, deep=True)
        return versions[i].model_copy(deep=True)

    # datasets
    def save_dataset(self, team: str, name: str, examples: list[LabeledExample], append: bool = True) -> DatasetInfo:
        ds = self.datasets.get((team, name))
        if ds is None or not append:
            ds = {"examples": [], "created_at": (ds or {}).get("created_at", _now())}
            self.datasets[(team, name)] = ds
        ds["examples"].extend(e.model_copy(deep=True) for e in examples)
        ds["updated_at"] = _now()
        return self._info(team, name)

    def _info(self, team: str, name: str) -> DatasetInfo:
        ds = self.datasets[(team, name)]
        qids = sorted({k for e in ds["examples"] for k in e.expected})
        return DatasetInfo(team=team, name=name, count=len(ds["examples"]), question_ids=qids,
                           created_at=ds["created_at"], updated_at=ds["updated_at"])

    def get_dataset_examples(self, team: str, name: str) -> list[LabeledExample]:
        if (team, name) not in self.datasets:
            raise NotFound(f"dataset {team}/{name} not found")
        return [e.model_copy(deep=True) for e in self.datasets[(team, name)]["examples"]]

    def list_datasets(self, team: str | None = None) -> list[DatasetInfo]:
        return [self._info(t, n) for (t, n) in sorted(self.datasets) if team is None or t == team]

    # reports
    def save_report(self, report: EvalReport) -> int:
        rid = len(self.reports) + 1
        self.reports[rid] = report.model_copy(update={"id": rid}, deep=True)
        return rid

    def get_report(self, report_id: int) -> EvalReport:
        if report_id not in self.reports:
            raise NotFound(f"report {report_id} not found")
        return self.reports[report_id].model_copy(deep=True)


class FakeJobs:
    def __init__(self, repo: FakeRepo, engine: Any) -> None:
        self.repo = repo
        self.engine = engine
        self.jobs: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Callable[[str], None]]] = {}

    def submit(self, *, team: str, kind: str, items: list[Any], questions: dict[str, dict[str, Any]],
               schema_ref: str | None = None, model: str | None = None, lang: str | None = None,
               append_to: str | None = None, finalize: bool = True, meta: dict[str, Any] | None = None) -> JobInfo:
        job_id = uuid.uuid4().hex[:12]
        raw = self.engine.predict_blocking(items, questions, model=model, lang=lang)
        self.jobs[job_id] = {
            "info": JobInfo(job_id=job_id, kind=kind, status="completed", team=team, total=len(items),
                            done=len(items), schema_ref=schema_ref, eta_seconds=0.0, created_at=_now(),
                            finished_at=_now()),
            "meta": dict(meta or {}),
            "raw": list(enumerate(raw)),
            "items": list(items),
        }
        errors = []
        for fn in self.hooks.get(kind, []):   # like the real service: hook failures become the job error
            try:
                fn(job_id)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            job = self.jobs[job_id]
            job["info"] = job["info"].model_copy(update={"error": "completion hook failed: " + "; ".join(errors)})
        return self.get(job_id)

    def _job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise NotFound(f"job {job_id} not found")
        return self.jobs[job_id]

    def get(self, job_id: str) -> JobInfo:
        return self._job(job_id)["info"].model_copy(deep=True)

    def meta(self, job_id: str) -> dict[str, Any]:
        return dict(self._job(job_id)["meta"])

    def raw_results(self, job_id: str) -> list[tuple[int, dict[str, Any]]]:
        return list(self._job(job_id)["raw"])

    def set_result_ref(self, job_id: str, ref: dict[str, Any]) -> None:
        job = self._job(job_id)
        job["info"] = job["info"].model_copy(update={"result_ref": dict(ref)})

    def on_complete(self, kind: str, fn: Callable[[str], None]) -> None:
        self.hooks.setdefault(kind, []).append(fn)
