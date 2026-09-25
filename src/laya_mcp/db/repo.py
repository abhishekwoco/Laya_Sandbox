"""Repository over SQLModel/SQLite. All methods are synchronous and thread-safe
(one Session per call); async tools call them via anyio.to_thread.run_sync or directly
(they are fast).

CONTRACT (implemented by the data workstream):
- Tables live in db/models.py (SQLModel); schema changes go through Alembic
  (db/migrations, shipped inside the package; alembic.ini at the project root for the CLI).
- `Repo.migrate()` runs `alembic upgrade head` programmatically (used at app startup and in tests
  with a temp sqlite file). It creates the database directory if missing.
- Schemas are versioned: every save_schema creates version N+1 with status 'draft'.
  Thresholds, temperatures and report links belong to a specific version.
- NotFound is raised for missing schemas/datasets/reports/jobs; ValueError for invalid input.
- Datetimes are returned as timezone-aware UTC.

SQLite setup: WAL journal, busy_timeout, foreign keys on, check_same_thread=False. The
driver's implicit transactions are disabled and every transaction is opened explicitly:
reads use BEGIN (deferred), writes use BEGIN IMMEDIATE so concurrent writers queue on the
busy timeout instead of failing with "database is locked" when upgrading a read lock.
"""
from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy import event, func
from sqlalchemy.engine import Engine, make_url
from sqlmodel import Session, delete, insert, select, update

from ..models import (
    DatasetInfo,
    EvalReport,
    LabeledExample,
    Question,
    Questions,
    SchemaInfo,
    SchemaRef,
    SchemaStatus,
    SchemaSummary,
    SchemaTargets,
    UsageRow,
    UsageStats,
)
from ..models.library import valid_name
from .models import (
    DatasetRecord,
    ExampleRecord,
    JobItemRecord,
    JobRecord,
    ReportRecord,
    SchemaRecord,
    SchemaVersionRecord,
    UsageEventRecord,
    utcnow,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

ACTIVE_JOB_STATUSES = ("queued", "running")
SCHEMA_STATUSES = ("draft", "evaluated", "trusted")
MAX_ERROR_CHARS = 2000

ItemClaim = Literal["run", "skip", "stop"]


class NotFound(LookupError):
    pass


class JobClosed(ValueError):
    """Items cannot be appended: the job is finalized or no longer queued/running."""


# helpers --------------------------------------------------------------------------------------
def _aware(dt: datetime | None) -> datetime | None:
    """Aware UTC; naive inputs are taken to be UTC already."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _json_default(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if hasattr(o, "tolist"):          # numpy arrays and scalars
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _check_name(value: str, what: str) -> str:
    v = (value or "").strip().lower()
    if not valid_name(v):
        raise ValueError(
            f"invalid {what} {value!r}: use lowercase letters, digits, '_', '-' or '.', "
            "starting with a letter or digit"
        )
    return v


def _dump_questions(questions: Questions | dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, q in questions.items():
        q = q if isinstance(q, Question) else Question.model_validate(q)
        out[str(qid)] = q.model_dump(mode="json", exclude_none=True)
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default method)."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    v = sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)
    return round(float(v), 1)


def create_sqlite_engine(url: str) -> Engine:
    """Engine with the pragmas and explicit transaction control described in the module doc."""
    u = make_url(url)
    if u.get_backend_name() != "sqlite":
        raise ValueError(f"only sqlite URLs are supported, got {u.get_backend_name()!r}")
    db = u.database
    if db and db != ":memory:" and not db.startswith("file:"):
        Path(db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    engine = sa.create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 30},
        json_serializer=_dumps,
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.isolation_level = None          # no implicit BEGIN from the driver; see _on_begin
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Any) -> None:
        mode = conn.get_execution_options().get("sqlite_begin", "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")

    return engine


class Repo:
    def __init__(self, db_url: str) -> None:
        self.url = db_url
        self.engine = create_sqlite_engine(db_url)
        self._write_engine = self.engine.execution_options(sqlite_begin="IMMEDIATE")

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def _read(self) -> Iterator[Session]:
        with Session(self.engine, expire_on_commit=False) as s:
            yield s

    @contextmanager
    def _write(self) -> Iterator[Session]:
        with Session(self._write_engine, expire_on_commit=False) as s:
            yield s
            s.commit()

    def migrate(self) -> None:
        """`alembic upgrade head` on this repo's database (creates it and its directory)."""
        from alembic import command
        from alembic.config import Config

        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        with self.engine.connect() as conn:
            # Batch migrations on SQLite recreate tables; FK enforcement must be off while they
            # run (the pragma is a no-op inside a transaction, so set it on the raw connection).
            raw = conn.connection.driver_connection
            raw.execute("PRAGMA foreign_keys=OFF")
            try:
                with conn.begin():
                    cfg.attributes["connection"] = conn
                    command.upgrade(cfg, "head")
            finally:
                raw.execute("PRAGMA foreign_keys=ON")

    # schemas ------------------------------------------------------------------
    @staticmethod
    def _schema_info(row: SchemaRecord, ver: SchemaVersionRecord) -> SchemaInfo:
        return SchemaInfo(
            team=row.team,
            name=row.name,
            version=ver.version,
            status=ver.status,  # type: ignore[arg-type]
            description=row.description,
            questions={qid: Question.model_validate(q) for qid, q in ver.questions.items()},
            targets=SchemaTargets.model_validate(ver.targets or {}),
            thresholds=dict(ver.thresholds or {}),
            temperatures=dict(ver.temperatures or {}),
            latest_report_id=ver.latest_report_id,
            created_at=_aware(ver.created_at),
        )

    @staticmethod
    def _find_schema(s: Session, team: str, name: str) -> SchemaRecord | None:
        return s.exec(
            select(SchemaRecord).where(SchemaRecord.team == team, SchemaRecord.name == name)
        ).first()

    def _find_version(self, s: Session, ref: SchemaRef) -> tuple[SchemaRecord, SchemaVersionRecord]:
        team, name = ref.team.strip().lower(), ref.name.strip().lower()
        row = self._find_schema(s, team, name)
        if row is None:
            raise NotFound(f"schema {team}/{name} not found")
        q = select(SchemaVersionRecord).where(SchemaVersionRecord.schema_id == row.id)
        if ref.version is not None:
            q = q.where(SchemaVersionRecord.version == ref.version)
        else:
            q = q.order_by(SchemaVersionRecord.version.desc()).limit(1)  # type: ignore[union-attr]
        ver = s.exec(q).first()
        if ver is None:
            raise NotFound(f"schema {team}/{name} has no version {ref.version}")
        return row, ver

    def save_schema(
        self,
        team: str,
        name: str,
        questions: Questions,
        description: str = "",
        targets: SchemaTargets | None = None,
    ) -> SchemaInfo:
        """Create the schema or add version N+1 (status 'draft'). An empty description keeps the
        existing one; targets=None carries the previous version's targets forward (or defaults).
        Thresholds and temperatures start empty: they belong to the version they were fit on."""
        team, name = _check_name(team, "team"), _check_name(name, "schema name")
        qdump = _dump_questions(questions)
        if not qdump:
            raise ValueError("a schema needs at least one question")
        now = utcnow()
        with self._write() as s:
            row = self._find_schema(s, team, name)
            prev: SchemaVersionRecord | None = None
            if row is None:
                row = SchemaRecord(team=team, name=name, description=description or "", created_at=now, updated_at=now)
                s.add(row)
                s.flush()
            else:
                if description:
                    row.description = description
                row.updated_at = now
                s.add(row)
                prev = s.exec(
                    select(SchemaVersionRecord)
                    .where(SchemaVersionRecord.schema_id == row.id)
                    .order_by(SchemaVersionRecord.version.desc())  # type: ignore[union-attr]
                    .limit(1)
                ).first()
            if targets is None:
                targets = SchemaTargets.model_validate(prev.targets) if prev else SchemaTargets()
            ver = SchemaVersionRecord(
                schema_id=row.id,  # type: ignore[arg-type]
                version=(prev.version + 1) if prev else 1,
                questions=qdump,
                targets=targets.model_dump(mode="json"),
                status="draft",
                thresholds={},
                temperatures={},
                created_at=now,
            )
            s.add(ver)
            s.flush()
            return self._schema_info(row, ver)

    def get_schema(self, ref: SchemaRef) -> SchemaInfo:
        """ref.version None -> latest version."""
        with self._read() as s:
            row, ver = self._find_version(s, ref)
            return self._schema_info(row, ver)

    def list_schemas(self, team: str | None = None) -> list[SchemaSummary]:
        """One row per schema (ref 'team/name'); status, description and question ids are those
        of the latest version, `versions` is the latest version number."""
        latest = (
            select(SchemaVersionRecord.schema_id, func.max(SchemaVersionRecord.version).label("v"))
            .group_by(SchemaVersionRecord.schema_id)
            .subquery()
        )
        stmt = (
            select(SchemaRecord, SchemaVersionRecord)
            .join(latest, latest.c.schema_id == SchemaRecord.id)
            .join(
                SchemaVersionRecord,
                sa.and_(SchemaVersionRecord.schema_id == SchemaRecord.id, SchemaVersionRecord.version == latest.c.v),
            )
            .order_by(SchemaRecord.team, SchemaRecord.name)
        )
        if team:
            stmt = stmt.where(SchemaRecord.team == team.strip().lower())
        with self._read() as s:
            return [
                SchemaSummary(
                    ref=f"{row.team}/{row.name}",
                    status=ver.status,  # type: ignore[arg-type]
                    description=row.description,
                    question_ids=list(ver.questions),
                    versions=ver.version,
                    updated_at=_aware(row.updated_at),
                )
                for row, ver in s.exec(stmt).all()
            ]

    def update_schema_version(
        self,
        ref: SchemaRef,
        *,
        status: SchemaStatus | None = None,
        thresholds: dict[str, float] | None = None,
        temperatures: dict[str, float] | None = None,
        latest_report_id: int | None = None,
    ) -> SchemaInfo:
        """Update one version (ref.version None -> latest). thresholds/temperatures REPLACE the
        stored dicts; keys must be question ids of that version."""
        if status is not None and status not in SCHEMA_STATUSES:
            raise ValueError(f"invalid schema status {status!r}; use one of {', '.join(SCHEMA_STATUSES)}")
        with self._write() as s:
            row, ver = self._find_version(s, ref)
            qids = set(ver.questions)
            if thresholds is not None:
                bad = sorted(set(thresholds) - qids)
                if bad:
                    raise ValueError(f"thresholds for unknown question ids {bad} in {row.team}/{row.name}@{ver.version}")
                if any(not 0.0 <= float(v) <= 1.0 for v in thresholds.values()):
                    raise ValueError("thresholds must be between 0 and 1")
                ver.thresholds = {k: float(v) for k, v in thresholds.items()}
            if temperatures is not None:
                bad = sorted(set(temperatures) - qids)
                if bad:
                    raise ValueError(f"temperatures for unknown question ids {bad} in {row.team}/{row.name}@{ver.version}")
                if any(not (math.isfinite(float(v)) and float(v) > 0) for v in temperatures.values()):
                    raise ValueError("temperatures must be positive numbers")
                ver.temperatures = {k: float(v) for k, v in temperatures.items()}
            if status is not None:
                ver.status = status
            if latest_report_id is not None:
                ver.latest_report_id = latest_report_id
            row.updated_at = utcnow()
            s.add(ver)
            s.add(row)
            s.flush()
            return self._schema_info(row, ver)

    # datasets -----------------------------------------------------------------
    @staticmethod
    def _dataset_info(ds: DatasetRecord, count: int) -> DatasetInfo:
        return DatasetInfo(
            team=ds.team,
            name=ds.name,
            count=count,
            question_ids=list(ds.question_ids or []),
            created_at=_aware(ds.created_at),
            updated_at=_aware(ds.updated_at),
        )

    @staticmethod
    def _count_examples(s: Session, dataset_id: int) -> int:
        return s.exec(select(func.count()).select_from(ExampleRecord).where(ExampleRecord.dataset_id == dataset_id)).one()

    def save_dataset(self, team: str, name: str, examples: list[LabeledExample], append: bool = True) -> DatasetInfo:
        """append=False replaces the dataset's examples."""
        team, name = _check_name(team, "team"), _check_name(name, "dataset name")
        exs = [e if isinstance(e, LabeledExample) else LabeledExample.model_validate(e) for e in examples]
        now = utcnow()
        with self._write() as s:
            ds = s.exec(select(DatasetRecord).where(DatasetRecord.team == team, DatasetRecord.name == name)).first()
            if ds is None:
                ds = DatasetRecord(team=team, name=name, question_ids=[], created_at=now, updated_at=now)
                s.add(ds)
                s.flush()
            if append:
                last = s.exec(select(func.max(ExampleRecord.idx)).where(ExampleRecord.dataset_id == ds.id)).one()
                start = 0 if last is None else last + 1
                qids = list(ds.question_ids or [])
            else:
                s.exec(delete(ExampleRecord).where(ExampleRecord.dataset_id == ds.id))
                start, qids = 0, []
            rows = []
            for i, e in enumerate(exs):
                dumped = e.model_dump(mode="json")
                for qid in dumped["expected"]:
                    if qid not in qids:
                        qids.append(qid)
                rows.append({"dataset_id": ds.id, "idx": start + i, "state": dumped["state"], "expected": dumped["expected"]})
            if rows:
                s.exec(insert(ExampleRecord), params=rows)
            ds.question_ids = qids
            ds.updated_at = now
            s.add(ds)
            s.flush()
            return self._dataset_info(ds, self._count_examples(s, ds.id))  # type: ignore[arg-type]

    def get_dataset_examples(self, team: str, name: str) -> list[LabeledExample]:
        team, name = team.strip().lower(), name.strip().lower()
        with self._read() as s:
            ds = s.exec(select(DatasetRecord).where(DatasetRecord.team == team, DatasetRecord.name == name)).first()
            if ds is None:
                raise NotFound(f"dataset {team}/{name} not found")
            rows = s.exec(
                select(ExampleRecord.state, ExampleRecord.expected)
                .where(ExampleRecord.dataset_id == ds.id)
                .order_by(ExampleRecord.idx)
            ).all()
            return [LabeledExample(state=state, expected=expected) for state, expected in rows]

    def list_datasets(self, team: str | None = None) -> list[DatasetInfo]:
        stmt = (
            select(DatasetRecord, func.count(ExampleRecord.id))
            .join(ExampleRecord, ExampleRecord.dataset_id == DatasetRecord.id, isouter=True)
            .group_by(DatasetRecord.id)
            .order_by(DatasetRecord.team, DatasetRecord.name)
        )
        if team:
            stmt = stmt.where(DatasetRecord.team == team.strip().lower())
        with self._read() as s:
            return [self._dataset_info(ds, n) for ds, n in s.exec(stmt).all()]

    # reports ------------------------------------------------------------------
    def save_report(self, report: EvalReport) -> int:
        """Stores the report (schema_ref inside it is 'team/name@version'); returns its id."""
        rec = ReportRecord(
            schema_ref=report.schema_ref,
            dataset=report.dataset,
            job_id=report.job_id,
            created_at=_aware(report.created_at),
            body=report.model_dump(mode="json", exclude={"id"}),
        )
        with self._write() as s:
            s.add(rec)
            s.flush()
            return rec.id  # type: ignore[return-value]

    def get_report(self, report_id: int) -> EvalReport:
        with self._read() as s:
            rec = s.get(ReportRecord, report_id)
            if rec is None:
                raise NotFound(f"report {report_id} not found")
            return EvalReport.model_validate({**rec.body, "id": rec.id})

    def list_reports(self, schema_ref: str | None = None, limit: int = 20) -> list[EvalReport]:
        """Newest first. schema_ref 'team/name@3' matches that version, 'team/name' every version."""
        stmt = select(ReportRecord).order_by(ReportRecord.id.desc()).limit(limit)  # type: ignore[union-attr]
        if schema_ref:
            ref = schema_ref.strip().lower()
            if "@" in ref:
                stmt = stmt.where(ReportRecord.schema_ref == ref)
            else:
                stmt = stmt.where(ReportRecord.schema_ref.like(ref.replace("%", r"\%").replace("_", r"\_") + "@%", escape="\\"))  # type: ignore[union-attr]
        with self._read() as s:
            return [EvalReport.model_validate({**r.body, "id": r.id}) for r in s.exec(stmt).all()]

    # usage --------------------------------------------------------------------
    def record_usage(
        self,
        *,
        team: str,
        tool: str,
        schema_ref: str | None,
        rows: int,
        latency_ms: int,
        answers: int = 0,
        needs_review: int = 0,
        error: bool = False,
    ) -> None:
        with self._write() as s:
            s.add(
                UsageEventRecord(
                    ts=utcnow(), team=team, tool=tool, schema_ref=schema_ref, rows=int(rows),
                    latency_ms=int(latency_ms), answers=int(answers), needs_review=int(needs_review), error=bool(error),
                )
            )

    def usage_stats(self, since: datetime, team: str | None = None) -> UsageStats:
        """Grouped by (team, tool, schema_ref): calls, rows, p50/p95 latency, needs_review/answers, errors."""
        E = UsageEventRecord
        stmt = select(E.team, E.tool, E.schema_ref, E.rows, E.latency_ms, E.answers, E.needs_review, E.error).where(
            E.ts >= _aware(since)
        )
        if team:
            stmt = stmt.where(E.team == team.strip().lower())
        groups: dict[tuple[str, str, str | None], list[tuple[int, int, int, int, bool]]] = defaultdict(list)
        with self._read() as s:
            for t, tool, ref, rows, lat, ans, nr, err in s.exec(stmt).all():
                groups[(t, tool, ref)].append((rows, lat, ans, nr, err))
        out: list[UsageRow] = []
        for (t, tool, ref), evs in groups.items():
            lat = sorted(float(e[1]) for e in evs)
            answers = sum(e[2] for e in evs)
            out.append(
                UsageRow(
                    team=t,
                    tool=tool,
                    schema_ref=ref,
                    calls=len(evs),
                    rows=sum(e[0] for e in evs),
                    p50_ms=_percentile(lat, 50),
                    p95_ms=_percentile(lat, 95),
                    needs_review_rate=round(sum(e[3] for e in evs) / answers, 4) if answers else None,
                    errors=sum(1 for e in evs if e[4]),
                )
            )
        out.sort(key=lambda r: (-r.calls, r.team, r.tool, r.schema_ref or ""))
        return UsageStats(since=since, rows=out)

    # jobs (persistence only; orchestration lives in jobs/service.py) ------------
    @staticmethod
    def _insert_items(s: Session, job_id: str, start: int, states: list[Any]) -> None:
        if states:
            s.exec(
                insert(JobItemRecord),
                params=[{"job_id": job_id, "idx": start + i, "state": st, "status": "pending"} for i, st in enumerate(states)],
            )

    def create_job(
        self,
        *,
        kind: str,
        team: str,
        questions: dict[str, Any],
        states: list[Any],
        schema_ref: str | None = None,
        model: str | None = None,
        lang: str | None = None,
        meta: dict[str, Any] | None = None,
        finalized: bool = True,
    ) -> JobRecord:
        """Insert a job and its items (idx 0..n-1, status pending) in one transaction."""
        job = JobRecord(
            id=uuid.uuid4().hex, kind=kind, team=team, status="queued", schema_ref=schema_ref,
            questions=dict(questions), model=model, lang=lang, meta=dict(meta or {}), result_ref={},
            total=len(states), done=0, failed=0, finalized=finalized, created_at=utcnow(),
        )
        with self._write() as s:
            s.add(job)
            s.flush()
            self._insert_items(s, job.id, 0, states)
        return job

    def append_job_items(
        self, job_id: str, states: list[Any], *, finalize: bool, meta: dict[str, Any] | None = None
    ) -> tuple[JobRecord, list[int]]:
        """Append items to an open job; returns (job, new item indexes). Raises JobClosed when the
        job was finalized or is no longer queued/running."""
        with self._write() as s:
            job = s.get(JobRecord, job_id)
            if job is None:
                raise NotFound(f"job {job_id} not found")
            if job.finalized:
                raise JobClosed(f"job {job_id} is already finalized; submit the items as a new job")
            if job.status not in ACTIVE_JOB_STATUSES:
                raise JobClosed(f"job {job_id} is {job.status}; submit the items as a new job")
            start = job.total
            self._insert_items(s, job_id, start, states)
            job.total = start + len(states)
            if finalize:
                job.finalized = True
            if meta:
                job.meta = {**(job.meta or {}), **meta}
            s.add(job)
            s.flush()
            return job, list(range(start, start + len(states)))

    def get_job(self, job_id: str) -> JobRecord:
        with self._read() as s:
            job = s.get(JobRecord, job_id)
            if job is None:
                raise NotFound(f"job {job_id} not found")
            return job

    def list_jobs(self, *, team: str | None = None, statuses: tuple[str, ...] | None = None, limit: int = 50) -> list[JobRecord]:
        """Newest first."""
        stmt = select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)  # type: ignore[union-attr]
        if team:
            stmt = stmt.where(JobRecord.team == team)
        if statuses:
            stmt = stmt.where(JobRecord.status.in_(statuses))  # type: ignore[attr-defined]
        with self._read() as s:
            return list(s.exec(stmt).all())

    def active_jobs(self) -> list[JobRecord]:
        """Queued/running jobs, oldest first."""
        stmt = (
            select(JobRecord)
            .where(JobRecord.status.in_(ACTIVE_JOB_STATUSES))  # type: ignore[attr-defined]
            .order_by(JobRecord.created_at, JobRecord.id)
        )
        with self._read() as s:
            return list(s.exec(stmt).all())

    def pending_item_indexes(self, job_id: str) -> list[int]:
        with self._read() as s:
            return list(
                s.exec(
                    select(JobItemRecord.idx)
                    .where(JobItemRecord.job_id == job_id, JobItemRecord.status == "pending")
                    .order_by(JobItemRecord.idx)
                ).all()
            )

    def pending_items_count(self) -> int:
        """Queue depth: pending items across queued/running jobs."""
        with self._read() as s:
            return s.exec(
                select(func.count())
                .select_from(JobItemRecord)
                .join(JobRecord, JobRecord.id == JobItemRecord.job_id)
                .where(JobItemRecord.status == "pending", JobRecord.status.in_(ACTIVE_JOB_STATUSES))  # type: ignore[attr-defined]
            ).one()

    def start_job_item(self, job_id: str, idx: int) -> tuple[ItemClaim, Any]:
        """Worker check before running one item: ('stop', None) when the job is gone or no longer
        active, ('skip', None) when the item is not pending, else ('run', state). Moves a queued
        job to running."""
        with self._write() as s:
            status = s.exec(select(JobRecord.status).where(JobRecord.id == job_id)).first()
            if status is None or status not in ACTIVE_JOB_STATUSES:
                return "stop", None
            item = s.exec(
                select(JobItemRecord).where(JobItemRecord.job_id == job_id, JobItemRecord.idx == idx)
            ).first()
            if item is None or item.status != "pending":
                return "skip", None
            if status == "queued":
                s.exec(
                    update(JobRecord)
                    .where(JobRecord.id == job_id, JobRecord.status == "queued")
                    .values(status="running")
                    .execution_options(synchronize_session=False)
                )
            return "run", item.state

    def finish_job_item(self, job_id: str, idx: int, *, raw: dict[str, Any] | None = None, error: str | None = None) -> bool:
        """Store one item's outcome and bump the job counters. Only a PENDING item transitions
        (a cancelled item's late result is discarded); returns whether it did."""
        with self._write() as s:
            res = s.exec(
                update(JobItemRecord)
                .where(JobItemRecord.job_id == job_id, JobItemRecord.idx == idx, JobItemRecord.status == "pending")
                .values(
                    status="failed" if error is not None else "done",
                    raw=None if error is not None else raw,
                    error=error[:MAX_ERROR_CHARS] if error is not None else None,
                )
                .execution_options(synchronize_session=False)
            )
            if res.rowcount != 1:
                return False
            s.exec(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .values(done=JobRecord.done + 1, failed=JobRecord.failed + (1 if error is not None else 0))
                .execution_options(synchronize_session=False)
            )
            return True

    def claim_job_completion(self, job_id: str) -> JobRecord | None:
        """Atomically claim a finalized job whose items are all processed (sets finished_at).
        Exactly one caller gets the job back. If every item failed the job is marked 'failed'
        here; otherwise the caller runs completion hooks and then calls complete_job()."""
        with self._write() as s:
            res = s.exec(
                update(JobRecord)
                .where(
                    JobRecord.id == job_id,
                    JobRecord.finalized.is_(True),  # type: ignore[attr-defined]
                    JobRecord.done >= JobRecord.total,
                    JobRecord.status.in_(ACTIVE_JOB_STATUSES),  # type: ignore[attr-defined]
                    JobRecord.finished_at.is_(None),  # type: ignore[union-attr]
                )
                .values(finished_at=utcnow())
                .execution_options(synchronize_session=False)
            )
            if res.rowcount != 1:
                return None
            job = s.get(JobRecord, job_id)
            assert job is not None
            if job.total > 0 and job.failed >= job.total:
                first = s.exec(
                    select(JobItemRecord.error)
                    .where(JobItemRecord.job_id == job_id, JobItemRecord.status == "failed")
                    .order_by(JobItemRecord.idx)
                    .limit(1)
                ).first()
                job.status = "failed"
                job.error = f"all {job.total} items failed; first error: {first}"[:MAX_ERROR_CHARS]
                s.add(job)
                s.flush()
            return job

    def complete_job(self, job_id: str, *, error: str | None = None) -> None:
        """Mark a claimed job completed (error = completion hook failure, if any)."""
        values: dict[str, Any] = {"status": "completed"}
        if error:
            values["error"] = error[:MAX_ERROR_CHARS]
        with self._write() as s:
            s.exec(
                update(JobRecord)
                .where(JobRecord.id == job_id, JobRecord.status.in_(ACTIVE_JOB_STATUSES))  # type: ignore[attr-defined]
                .values(**values)
                .execution_options(synchronize_session=False)
            )

    def release_completion_claims(self) -> int:
        """Startup recovery: a crash between claim_job_completion and complete_job leaves an
        active job with finished_at set; clear it so the job can be completed again."""
        with self._write() as s:
            res = s.exec(
                update(JobRecord)
                .where(JobRecord.status.in_(ACTIVE_JOB_STATUSES), JobRecord.finished_at.is_not(None))  # type: ignore[attr-defined,union-attr]
                .values(finished_at=None)
                .execution_options(synchronize_session=False)
            )
            return res.rowcount

    def cancel_job(self, job_id: str) -> JobRecord:
        """Cancel a queued/running job and its pending items. No-op for finished jobs and for
        jobs already claimed for completion."""
        with self._write() as s:
            job = s.get(JobRecord, job_id)
            if job is None:
                raise NotFound(f"job {job_id} not found")
            if job.status in ACTIVE_JOB_STATUSES and job.finished_at is None:
                s.exec(
                    update(JobItemRecord)
                    .where(JobItemRecord.job_id == job_id, JobItemRecord.status == "pending")
                    .values(status="cancelled")
                    .execution_options(synchronize_session=False)
                )
                job.status = "cancelled"
                job.finished_at = utcnow()
                s.add(job)
                s.flush()
            return job

    def set_job_result_ref(self, job_id: str, ref: dict[str, Any]) -> None:
        with self._write() as s:
            res = s.exec(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .values(result_ref=dict(ref))
                .execution_options(synchronize_session=False)
            )
            if res.rowcount != 1:
                raise NotFound(f"job {job_id} not found")

    def job_raw_results(self, job_id: str) -> list[tuple[int, dict[str, Any]]]:
        """[(idx, raw)] for items with status 'done', in index order."""
        with self._read() as s:
            rows = s.exec(
                select(JobItemRecord.idx, JobItemRecord.raw)
                .where(JobItemRecord.job_id == job_id, JobItemRecord.status == "done")
                .order_by(JobItemRecord.idx)
            ).all()
            return [(idx, raw) for idx, raw in rows]

    def job_states(self, job_id: str, indices: list[int] | None = None) -> dict[int, Any]:
        """{idx: state} for a job's items (all, or only `indices`)."""
        with self._read() as s:
            q = select(JobItemRecord.idx, JobItemRecord.state).where(JobItemRecord.job_id == job_id)
            if indices is not None:
                q = q.where(JobItemRecord.idx.in_(list(indices)))
            return {idx: state for idx, state in s.exec(q.order_by(JobItemRecord.idx)).all()}

    def job_failures(self, job_id: str) -> list[tuple[int, str]]:
        """[(idx, error)] for failed items, in index order."""
        with self._read() as s:
            rows = s.exec(
                select(JobItemRecord.idx, JobItemRecord.error)
                .where(JobItemRecord.job_id == job_id, JobItemRecord.status == "failed")
                .order_by(JobItemRecord.idx)
            ).all()
            return [(idx, err or "") for idx, err in rows]
