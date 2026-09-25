"""SQLModel tables. Schema changes go through Alembic (db/migrations); never call create_all.

Conventions:
- Timestamps are timezone-aware UTC. SQLModel maps `datetime` to its UTCDateTime type, which
  rejects naive values on write and returns aware UTC on read (SQLite stores them without offset).
- JSON columns hold plain JSON (Pydantic models are dumped with mode="json" before storing).
- Constraint/index names follow NAMING_CONVENTION so future batch migrations on SQLite can
  refer to them by name.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlmodel import Field, SQLModel

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
SQLModel.metadata.naming_convention = NAMING_CONVENTION


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json(nullable: bool = False) -> Any:
    return sa.JSON(none_as_null=True) if nullable else sa.JSON()


# schema library -------------------------------------------------------------------------------
class SchemaRecord(SQLModel, table=True):
    __tablename__ = "schemas"
    __table_args__ = (sa.UniqueConstraint("team", "name"),)

    id: int | None = Field(default=None, primary_key=True)
    team: str = Field(index=True)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class SchemaVersionRecord(SQLModel, table=True):
    __tablename__ = "schema_versions"
    __table_args__ = (sa.UniqueConstraint("schema_id", "version"),)

    id: int | None = Field(default=None, primary_key=True)
    schema_id: int = Field(foreign_key="schemas.id")     # (schema_id, version) unique index covers lookups
    version: int
    questions: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    targets: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    status: str = "draft"                                  # draft | evaluated | trusted
    thresholds: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    temperatures: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    latest_report_id: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


# datasets -------------------------------------------------------------------------------------
class DatasetRecord(SQLModel, table=True):
    __tablename__ = "datasets"
    __table_args__ = (sa.UniqueConstraint("team", "name"),)

    id: int | None = Field(default=None, primary_key=True)
    team: str = Field(index=True)
    name: str
    question_ids: list[str] = Field(default_factory=list, sa_type=_json())
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ExampleRecord(SQLModel, table=True):
    __tablename__ = "examples"
    __table_args__ = (sa.Index("ix_examples_dataset_id_idx", "dataset_id", "idx", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="datasets.id")
    idx: int
    state: Any = Field(sa_type=_json())
    expected: dict[str, Any] = Field(default_factory=dict, sa_type=_json())


# evaluation reports ---------------------------------------------------------------------------
class ReportRecord(SQLModel, table=True):
    __tablename__ = "reports"

    id: int | None = Field(default=None, primary_key=True)
    schema_ref: str = Field(index=True)                    # team/name@version
    dataset: str
    job_id: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    body: dict[str, Any] = Field(default_factory=dict, sa_type=_json())   # EvalReport (mode="json", no id)


# usage ----------------------------------------------------------------------------------------
class UsageEventRecord(SQLModel, table=True):
    __tablename__ = "usage_events"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    team: str
    tool: str
    schema_ref: str | None = None
    rows: int = 0
    latency_ms: int = 0
    answers: int = 0
    needs_review: int = 0
    error: bool = False


# jobs -----------------------------------------------------------------------------------------
class JobRecord(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(primary_key=True)                      # uuid4 hex
    kind: str                                              # batch | evaluate
    team: str = Field(index=True)
    status: str = Field(default="queued", index=True)      # queued | running | completed | failed | cancelled
    schema_ref: str | None = None
    questions: dict[str, Any] = Field(default_factory=dict, sa_type=_json())   # laya wire format
    model: str | None = None
    lang: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    result_ref: dict[str, Any] = Field(default_factory=dict, sa_type=_json())
    total: int = 0
    done: int = 0                                          # processed items (succeeded + failed)
    failed: int = 0
    finalized: bool = False
    created_at: datetime = Field(default_factory=utcnow, index=True)
    # Set when the job is claimed for completion (hooks running) and kept as the finish time.
    finished_at: datetime | None = None
    error: str | None = None


class JobItemRecord(SQLModel, table=True):
    __tablename__ = "job_items"
    __table_args__ = (sa.Index("ix_job_items_job_id_idx", "job_id", "idx", unique=True),)

    id: int | None = Field(default=None, primary_key=True)
    job_id: str = Field(foreign_key="jobs.id")
    idx: int
    state: Any = Field(sa_type=_json())
    status: str = "pending"                                # pending | done | failed | cancelled
    raw: dict[str, Any] | None = Field(default=None, sa_type=_json(nullable=True))   # UNCALIBRATED laya result
    error: str | None = None
