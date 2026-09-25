"""Batch jobs, server status and usage statistics."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .results import ItemResult

JobKind = Literal["batch", "evaluate"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobInfo(BaseModel):
    job_id: str
    kind: JobKind
    status: JobStatus
    team: str
    total: int
    done: int
    failed: int = 0
    schema_ref: str | None = None
    eta_seconds: float | None = None
    created_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    result_ref: dict[str, Any] = Field(default_factory=dict, description="e.g. {'report_id': 7} for evaluate jobs")


class JobResultsPage(BaseModel):
    job: JobInfo
    items: list[ItemResult]
    offset: int
    next_offset: int | None
    summary: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="question id -> answer value -> count, across the whole job (noul as 'true'/'false', score as rounded level)",
    )


class EngineStatus(BaseModel):
    laya_version: str
    device: str
    threads: int | None
    loaded: list[str]
    available: list[str]
    busy: bool
    waiting: int
    rss_gb: float | None
    uptime_s: float
    ms_per_row: float | None = Field(default=None, description="Rolling average inference cost per row")


class UsageRow(BaseModel):
    team: str
    tool: str
    schema_ref: str | None
    calls: int
    rows: int
    p50_ms: float
    p95_ms: float
    needs_review_rate: float | None
    errors: int


class UsageStats(BaseModel):
    since: datetime
    rows: list[UsageRow]
