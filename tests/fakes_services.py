"""Minimal in-memory stand-ins for Repo and JobService (the real ones are built by the data
workstream). They implement only the methods the server/tools workstream calls, following the
contracts in db/repo.py and jobs/service.py. Jobs run synchronously inside `submit`."""
from __future__ import annotations

import itertools
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from laya_mcp.db.repo import NotFound
from laya_mcp.engine.answers import to_item_result
from laya_mcp.models import (
    EvalReport,
    JobInfo,
    JobResultsPage,
    Questions,
    SchemaInfo,
    SchemaRef,
    SchemaSummary,
    SchemaTargets,
    UsageRow,
    UsageStats,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryRepo:
    def __init__(self, db_url: str = "memory://") -> None:
        self.schemas: dict[tuple[str, str], list[SchemaInfo]] = {}
        self.reports: dict[int, EvalReport] = {}
        self.usage: list[dict[str, Any]] = []
        self.migrated = False
        self._ids = itertools.count(1)

    def migrate(self) -> None:
        self.migrated = True

    # schemas
    def save_schema(self, team: str, name: str, questions: Questions, description: str = "",
                    targets: SchemaTargets | None = None) -> SchemaInfo:
        versions = self.schemas.setdefault((team, name), [])
        info = SchemaInfo(team=team, name=name, version=len(versions) + 1, status="draft", description=description,
                          questions=questions, targets=targets or SchemaTargets(), created_at=_now())
        versions.append(info)
        return info

    def get_schema(self, ref: SchemaRef) -> SchemaInfo:
        versions = self.schemas.get((ref.team, ref.name))
        if not versions:
            raise NotFound(f"schema {ref.team}/{ref.name} does not exist")
        if ref.version is None:
            return versions[-1]
        if not 1 <= ref.version <= len(versions):
            raise NotFound(f"schema {ref} has no version {ref.version}")
        return versions[ref.version - 1]

    def list_schemas(self, team: str | None = None) -> list[SchemaSummary]:
        out = []
        for (t, n), versions in self.schemas.items():
            if team and t != team:
                continue
            last = versions[-1]
            out.append(SchemaSummary(ref=last.ref, status=last.status, description=last.description,
                                     question_ids=list(last.questions), versions=len(versions),
                                     updated_at=last.created_at))
        return out

    def update_schema_version(self, ref: SchemaRef, *, status=None, thresholds=None, temperatures=None,
                              latest_report_id=None) -> SchemaInfo:
        info = self.get_schema(ref)
        upd = {k: v for k, v in dict(status=status, thresholds=thresholds, temperatures=temperatures,
                                     latest_report_id=latest_report_id).items() if v is not None}
        new = info.model_copy(update=upd)
        versions = self.schemas[(info.team, info.name)]
        versions[info.version - 1] = new
        return new

    # reports
    def save_report(self, report: EvalReport) -> int:
        rid = next(self._ids)
        self.reports[rid] = report.model_copy(update={"id": rid})
        return rid

    def get_report(self, report_id: int) -> EvalReport:
        if report_id not in self.reports:
            raise NotFound(f"report {report_id} does not exist")
        return self.reports[report_id]

    # usage
    def record_usage(self, *, team: str, tool: str, schema_ref: str | None, rows: int, latency_ms: int,
                     answers: int = 0, needs_review: int = 0, error: bool = False) -> None:
        self.usage.append(dict(team=team, tool=tool, schema_ref=schema_ref, rows=rows, latency_ms=latency_ms,
                               answers=answers, needs_review=needs_review, error=error, at=_now()))

    def usage_stats(self, since: datetime, team: str | None = None) -> UsageStats:
        groups: dict[tuple, list[dict]] = {}
        for u in self.usage:
            if u["at"] < since or (team and u["team"] != team):
                continue
            groups.setdefault((u["team"], u["tool"], u["schema_ref"]), []).append(u)
        rows = []
        for (t, tool, ref), us in sorted(groups.items(), key=lambda kv: str(kv[0])):
            lat = sorted(u["latency_ms"] for u in us)
            answers = sum(u["answers"] for u in us)
            rows.append(UsageRow(team=t, tool=tool, schema_ref=ref, calls=len(us), rows=sum(u["rows"] for u in us),
                                 p50_ms=float(statistics.median(lat)), p95_ms=float(lat[int(0.95 * (len(lat) - 1))]),
                                 needs_review_rate=(sum(u["needs_review"] for u in us) / answers) if answers else None,
                                 errors=sum(u["error"] for u in us)))
        return UsageStats(since=since, rows=rows)


class MemoryJobs:
    def __init__(self, settings: Any, repo: Any, engine: Any) -> None:
        self.settings, self.repo, self.engine = settings, repo, engine
        self.jobs: dict[str, dict[str, Any]] = {}
        self.hooks: dict[str, list[Callable[[str], None]]] = {}
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def on_complete(self, kind: str, fn: Callable[[str], None]) -> None:
        self.hooks.setdefault(kind, []).append(fn)

    def submit(self, *, team, kind, items, questions, schema_ref=None, model=None, lang=None, append_to=None,
               finalize=True, meta=None) -> JobInfo:
        if append_to is not None:
            job = self._job(append_to)
            if job["finalized"]:
                raise ValueError(f"job {append_to} is finalized; start a new job")
            questions = questions or job["questions"]
        else:
            if not questions:
                raise ValueError("a new job needs questions")
            job = dict(id=uuid.uuid4().hex[:12], kind=kind, team=team, questions=questions, schema_ref=schema_ref,
                       model=model, lang=lang, raw=[], status="running", finalized=False, cancelled=False,
                       meta=meta or {}, created_at=_now(), finished_at=None, result_ref={})
            self.jobs[job["id"]] = job
        for state in items:
            if job["cancelled"]:
                break
            job["raw"].append(self.engine.predict_blocking([state], job["questions"], model=model, lang=lang)[0])
        if finalize:
            job["finalized"] = True
            if not job["cancelled"]:
                job["status"], job["finished_at"] = "completed", _now()
                for fn in self.hooks.get(kind, []):
                    fn(job["id"])
        return self.get(job["id"])

    def _job(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise NotFound(f"job {job_id} does not exist")
        return self.jobs[job_id]

    def get(self, job_id: str) -> JobInfo:
        j = self._job(job_id)
        n = len(j["raw"])
        return JobInfo(job_id=j["id"], kind=j["kind"], status=j["status"], team=j["team"], total=n, done=n,
                       schema_ref=j["schema_ref"], eta_seconds=0.0 if j["status"] == "completed" else None,
                       created_at=j["created_at"], finished_at=j["finished_at"], result_ref=j["result_ref"])

    def meta(self, job_id: str) -> dict[str, Any]:
        return self._job(job_id)["meta"]

    def raw_results(self, job_id: str) -> list[tuple[int, dict[str, Any]]]:
        return list(enumerate(self._job(job_id)["raw"]))

    def set_result_ref(self, job_id: str, ref: dict[str, Any]) -> None:
        self._job(job_id)["result_ref"] = ref

    def cancel(self, job_id: str) -> JobInfo:
        j = self._job(job_id)
        if j["status"] in ("queued", "running"):
            j["cancelled"], j["status"], j["finished_at"] = True, "cancelled", _now()
        return self.get(job_id)

    def results(self, job_id, *, offset=0, limit=50, needs_review_only=False, question_id=None, value=None,
                detail="compact", temperatures=None, thresholds=None, default_threshold=None) -> JobResultsPage:
        j = self._job(job_id)
        items = [to_item_result(i, r, detail=detail, temperatures=temperatures, thresholds=thresholds,
                                default_threshold=default_threshold) for i, r in enumerate(j["raw"])]
        summary: dict[str, dict[str, int]] = {}
        for it in items:
            for qid, a in it.answers.items():
                key = str(a.level if a.type == "score" else a.value).lower()
                summary.setdefault(qid, {}).setdefault(key, 0)
                summary[qid][key] += 1

        def keep(it) -> bool:
            if needs_review_only and not it.needs_review:
                return False
            if question_id is not None:
                a = it.answers.get(question_id)
                if a is None:
                    return False
                if value is not None and str(a.level if a.type == "score" else a.value).lower() != value:
                    return False
            return True

        kept = [it for it in items if keep(it)]
        page = kept[offset: offset + limit]
        nxt = offset + limit if offset + limit < len(kept) else None
        return JobResultsPage(job=self.get(job_id), items=page, offset=offset, next_offset=nxt, summary=summary)


def install_optional_stubs(monkeypatch: Any) -> list[str]:
    """Make modules owned by other workstreams importable while they are being written:
    laya_mcp.tools.library / laya_mcp.tools.workbench (FastMCP sub-servers) and
    laya_mcp.workbench.hooks (register_hooks). Real modules are used whenever they import.
    Returns the names that were stubbed."""
    import importlib
    import sys
    import types

    from fastmcp import FastMCP

    stubbed = []
    for mod in ("laya_mcp.tools.library", "laya_mcp.tools.workbench", "laya_mcp.workbench.hooks"):
        try:
            importlib.import_module(mod)
            continue
        except ImportError:
            pass
        stub = types.ModuleType(mod)
        if mod.endswith("hooks"):
            stub.register_hooks = lambda jobs: None
        else:
            stub.server = FastMCP(f"stub-{mod.rsplit('.', 1)[1]}")
        monkeypatch.setitem(sys.modules, mod, stub)
        stubbed.append(mod)
    return stubbed
