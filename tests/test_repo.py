"""Repo: migrations, schema library, datasets, reports, usage stats (SQLite in a temp dir)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import SQLModel

from laya_mcp.config import Settings
from laya_mcp.db import models as db_models  # noqa: F401  (tables on SQLModel.metadata)
from laya_mcp.db.repo import NotFound, Repo
from laya_mcp.models import (
    EvalReport,
    LabeledExample,
    Question,
    QuestionMetrics,
    SchemaRef,
    SchemaTargets,
)
from laya_mcp.models.library import Miss

TABLES = {"schemas", "schema_versions", "datasets", "examples", "reports", "usage_events", "jobs", "job_items"}

QUESTIONS = {
    "team": Question(type="choice", instructions="Which team owns this?", criteria={"backend": "server side", "frontend": None}),
    "bug": Question(type="noul", instructions="Is this a bug?"),
}


@pytest.fixture
def repo(settings: Settings):
    r = Repo(settings.db_url)
    r.migrate()
    yield r
    r.close()


# migrations -------------------------------------------------------------------------------------
def test_migrate_creates_db_in_missing_directory(tmp_path):
    settings = Settings(data_dir=tmp_path / "does" / "not" / "exist")
    repo = Repo(settings.db_url)
    repo.migrate()
    repo.migrate()  # idempotent
    db = settings.data_dir / "laya_mcp.db"
    assert db.exists()
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
        assert TABLES <= names
        assert conn.execute("select version_num from alembic_version").fetchone() == ("0001",)
        assert conn.execute("pragma journal_mode").fetchone()[0] == "wal"
    repo.close()


def test_migrations_match_models(repo):
    """Guard: a model change without a new Alembic revision fails here."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with repo.engine.connect() as conn:
        mc = MigrationContext.configure(conn, opts={"compare_type": True, "render_as_batch": True})
        assert compare_metadata(mc, SQLModel.metadata) == []


def test_foreign_keys_enforced(repo):
    with pytest.raises(Exception):
        with repo._write() as s:
            s.add(db_models.ExampleRecord(dataset_id=999, idx=0, state="x", expected={}))


# schemas ----------------------------------------------------------------------------------------
def test_schema_versioning(repo):
    v1 = repo.save_schema("Dev", "Issue-Triage", QUESTIONS, description="triage incoming issues",
                          targets=SchemaTargets(min_accuracy=0.85, min_examples=20))
    assert (v1.team, v1.name, v1.version, v1.status) == ("dev", "issue-triage", 1, "draft")
    assert v1.ref == "dev/issue-triage@1"
    assert v1.created_at.tzinfo is not None
    assert v1.questions["team"].criteria == {"backend": "server side", "frontend": None}

    repo.update_schema_version(SchemaRef.parse("dev/issue-triage@1"), status="evaluated",
                               thresholds={"team": 0.7}, temperatures={"bug": 1.5})

    v2 = repo.save_schema("dev", "issue-triage", {"bug": {"type": "noul", "instructions": "Is it a regression?"}})
    assert v2.version == 2 and v2.status == "draft"
    assert v2.description == "triage incoming issues"              # empty description keeps the old one
    assert v2.targets.min_accuracy == 0.85                           # targets carried forward
    assert v2.thresholds == {} and v2.temperatures == {}             # calibration belongs to v1

    latest = repo.get_schema(SchemaRef.parse("dev/issue-triage"))
    assert latest.version == 2 and list(latest.questions) == ["bug"]
    old = repo.get_schema(SchemaRef.parse("dev/issue-triage@1"))
    assert old.status == "evaluated" and old.thresholds == {"team": 0.7} and old.temperatures == {"bug": 1.5}
    assert set(old.questions) == {"team", "bug"}

    with pytest.raises(NotFound):
        repo.get_schema(SchemaRef.parse("dev/issue-triage@7"))
    with pytest.raises(NotFound):
        repo.get_schema(SchemaRef.parse("dev/nope"))


def test_list_schemas(repo):
    repo.save_schema("dev", "a", QUESTIONS, description="first")
    repo.save_schema("dev", "a", QUESTIONS)
    repo.save_schema("support", "b", {"bug": QUESTIONS["bug"]})
    repo.update_schema_version(SchemaRef.parse("support/b"), status="trusted")

    rows = repo.list_schemas()
    assert [r.ref for r in rows] == ["dev/a", "support/b"]
    a, b = rows
    assert a.versions == 2 and a.status == "draft" and a.description == "first"
    assert a.question_ids == ["team", "bug"]
    assert b.status == "trusted" and b.versions == 1
    assert [r.ref for r in repo.list_schemas(team="support")] == ["support/b"]
    assert repo.list_schemas(team="nobody") == []


def test_update_schema_version_validation(repo):
    repo.save_schema("dev", "s", QUESTIONS)
    ref = SchemaRef.parse("dev/s")
    with pytest.raises(ValueError, match="unknown question ids"):
        repo.update_schema_version(ref, thresholds={"nope": 0.5})
    with pytest.raises(ValueError, match="between 0 and 1"):
        repo.update_schema_version(ref, thresholds={"team": 1.5})
    with pytest.raises(ValueError, match="positive"):
        repo.update_schema_version(ref, temperatures={"team": 0})
    with pytest.raises(ValueError, match="invalid schema status"):
        repo.update_schema_version(ref, status="golden")  # type: ignore[arg-type]
    info = repo.update_schema_version(ref, latest_report_id=12, status="trusted")
    assert info.latest_report_id == 12 and info.status == "trusted"
    with pytest.raises(NotFound):
        repo.update_schema_version(SchemaRef.parse("dev/s@2"), status="trusted")


def test_invalid_names_rejected(repo):
    with pytest.raises(ValueError, match="invalid schema name"):
        repo.save_schema("dev", "has space", QUESTIONS)
    with pytest.raises(ValueError, match="invalid team"):
        repo.save_schema("dev/x", "ok", QUESTIONS)
    with pytest.raises(ValueError, match="at least one question"):
        repo.save_schema("dev", "empty", {})


# datasets ---------------------------------------------------------------------------------------
def test_datasets_append_and_replace(repo):
    info = repo.save_dataset("dev", "issues", [
        LabeledExample(state="server down", expected={"team": "backend"}),
        LabeledExample(state={"title": "button", "body": "misaligned"}, expected={"team": "frontend", "bug": True}),
    ])
    assert info.count == 2 and info.question_ids == ["team", "bug"]

    info = repo.save_dataset("dev", "issues", [LabeledExample(state=[{"role": "user", "text": "hi"}], expected={"sev": 2})])
    assert info.count == 3 and info.question_ids == ["team", "bug", "sev"]
    exs = repo.get_dataset_examples("dev", "issues")
    assert [e.state for e in exs] == ["server down", {"title": "button", "body": "misaligned"}, [{"role": "user", "text": "hi"}]]
    assert exs[1].expected == {"team": "frontend", "bug": True}
    assert exs[2].expected == {"sev": 2}

    info = repo.save_dataset("dev", "issues", [LabeledExample(state="only", expected={"bug": False})], append=False)
    assert info.count == 1 and info.question_ids == ["bug"]
    assert repo.get_dataset_examples("dev", "issues") == [LabeledExample(state="only", expected={"bug": False})]

    repo.save_dataset("support", "tickets", [])
    listed = {(d.team, d.name): d.count for d in repo.list_datasets()}
    assert listed == {("dev", "issues"): 1, ("support", "tickets"): 0}
    assert [d.name for d in repo.list_datasets(team="support")] == ["tickets"]
    with pytest.raises(NotFound):
        repo.get_dataset_examples("dev", "missing")


# reports ----------------------------------------------------------------------------------------
def _report(schema_ref: str = "dev/issue-triage@1", job_id: str | None = "abc") -> EvalReport:
    return EvalReport(
        schema_ref=schema_ref,
        dataset="dev/issues",
        job_id=job_id,
        created_at=datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        n_examples=60,
        target_accuracy=0.9,
        calibrated=False,
        per_question={
            "team": QuestionMetrics(type="choice", n=60, accuracy=0.93, ece=0.04, labels=["backend", "frontend"],
                                    confusion=[[30, 2], [2, 26]], recommended_threshold=0.8, meets_target=True),
        },
        overall_accuracy=0.93,
        passes_targets=True,
        worst_misses=[Miss(example_index=3, question_id="team", expected="backend", predicted="frontend", confidence=0.97)],
        model_counts={"english": 60},
    )


def test_reports_round_trip(repo):
    report = _report()
    rid = repo.save_report(report)
    loaded = repo.get_report(rid)
    assert loaded.id == rid
    assert loaded.model_dump(exclude={"id"}) == report.model_dump(exclude={"id"})

    rid2 = repo.save_report(_report("dev/issue-triage@2", job_id=None))
    repo.save_report(_report("dev/other@1"))
    assert [r.id for r in repo.list_reports("dev/issue-triage")] == [rid2, rid]
    assert [r.id for r in repo.list_reports("dev/issue-triage@1")] == [rid]
    assert len(repo.list_reports()) == 3
    with pytest.raises(NotFound):
        repo.get_report(9999)


# usage ------------------------------------------------------------------------------------------
def test_usage_stats_percentiles_and_grouping(repo):
    for i in range(101):                                   # latencies 0, 10, ..., 1000
        repo.record_usage(team="dev", tool="laya_decide", schema_ref="dev/triage@1", rows=3,
                          latency_ms=i * 10, answers=3, needs_review=1 if i % 2 else 0, error=(i % 25 == 0))
    repo.record_usage(team="dev", tool="laya_classify", schema_ref=None, rows=2, latency_ms=100)
    repo.record_usage(team="dev", tool="laya_classify", schema_ref=None, rows=2, latency_ms=200)
    repo.record_usage(team="sales", tool="laya_classify", schema_ref=None, rows=1, latency_ms=50, answers=2, needs_review=2)

    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    stats = repo.usage_stats(since)
    assert stats.since == since
    by = {(r.team, r.tool, r.schema_ref): r for r in stats.rows}
    decide = by[("dev", "laya_decide", "dev/triage@1")]
    assert stats.rows[0] is decide or stats.rows[0] == decide              # most calls first
    assert (decide.calls, decide.rows, decide.errors) == (101, 303, 5)
    assert (decide.p50_ms, decide.p95_ms) == (500.0, 950.0)
    assert decide.needs_review_rate == round(50 / 303, 4)
    classify = by[("dev", "laya_classify", None)]
    assert (classify.calls, classify.p50_ms, classify.p95_ms, classify.needs_review_rate) == (2, 150.0, 195.0, None)
    assert by[("sales", "laya_classify", None)].needs_review_rate == 1.0

    assert {r.team for r in repo.usage_stats(since, team="sales").rows} == {"sales"}
    assert repo.usage_stats(datetime.now(timezone.utc) + timedelta(minutes=1)).rows == []
    # naive `since` is taken as UTC
    assert len(repo.usage_stats(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5)).rows) == 3


def test_concurrent_writers_do_not_lock(repo):
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for k in range(25):
                repo.record_usage(team="dev", tool=f"t{n}", schema_ref=None, rows=1, latency_ms=k)
                if k % 5 == 0:
                    repo.save_dataset("dev", f"ds{n}", [LabeledExample(state=str(k), expected={"q": k})])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sum(r.calls for r in repo.usage_stats(datetime.now(timezone.utc) - timedelta(minutes=5)).rows) == 200
