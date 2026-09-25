"""Ops tools: laya_status, laya_load_models, laya_unload_models, laya_detect_language, laya_usage_stats."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import laya
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from .. import __version__
from ..models import EngineStatus, State, UsageStats
from ..state import current_team, get_state
from ._core import READ_ONLY, ModelName, Stopwatch, run_sync, tool_errors

server = FastMCP("laya-ops")


class Limits(BaseModel):
    sync_row_budget: int = Field(description="Max items x questions in one synchronous call")
    max_questions: int
    max_state_chars: int
    max_items_per_call: int = Field(description="Max items per laya_classify_batch call")
    default_threshold: float
    request_timeout_s: float


class ServerStatus(BaseModel):
    server_version: str
    team: str = Field(description="Team this client is identified as (X-Laya-Team header)")
    engine: EngineStatus
    limits: Limits
    seconds_per_row: float = Field(description="Current cost estimate per row (one question on one item)")
    pending_batch_items: int | None = Field(
        default=None, description="Batch/evaluate items waiting in the job queue (all teams)"
    )


class ModelsResult(BaseModel):
    loaded: list[str]
    latency_ms: int


class LanguageInfo(BaseModel):
    script: str = Field(description="Dominant script: latin, han, devanagari, arabic, ... or unknown")
    language: str | None = Field(description="Best-effort ISO 639-1 code for Latin-script text; None when undecided")
    language_undecided: bool
    is_english: bool = Field(description="True when the English checkpoint can read this text")
    recommended_model: str = Field(description="Checkpoint auto-routing would pick: english or multilingual")
    non_latin_fraction: float
    script_profile: dict[str, float]


@server.tool(
    name="laya_status",
    description=(
        "Server health and limits: loaded checkpoints, device, torch threads, whether inference is busy and how "
        "many requests are waiting, batch items queued, memory, current cost per row, your team, and the request "
        "limits (synchronous row budget, max questions, max items per batch call). Check it before planning a "
        "large workload."
    ),
    tags={"ops"},
    annotations={**READ_ONLY, "title": "Laya status"},
)
async def laya_status() -> ServerStatus:
    st = get_state()
    s = st.settings
    pending = None
    counter = getattr(st.repo, "pending_items_count", None)
    if callable(counter):
        try:
            pending = int(await run_sync(counter))
        except Exception:
            pending = None
    return ServerStatus(
        server_version=__version__,
        team=current_team(),
        engine=st.engine.status(),
        limits=Limits(
            sync_row_budget=s.sync_row_budget,
            max_questions=s.max_questions,
            max_state_chars=s.max_state_chars,
            max_items_per_call=s.max_items_per_call,
            default_threshold=s.default_threshold,
            request_timeout_s=s.request_timeout_s,
        ),
        seconds_per_row=round(st.engine.estimate_seconds(1), 3),
        pending_batch_items=pending,
    )


@server.tool(
    name="laya_load_models",
    description=(
        "Load Laya checkpoints into memory ahead of use (english is preloaded; multilingual and typed-decisions "
        "load lazily on first use, ~10 s each). At most 2 stay resident; loading a third evicts the least recently "
        "used. Only needed to avoid a cold start before a latency-sensitive run."
    ),
    tags={"ops"},
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False,
                 "title": "Load models"},
)
async def laya_load_models(
    models: Annotated[list[ModelName], Field(min_length=1, description="Checkpoints to load.")],
) -> ModelsResult:
    st = get_state()
    sw = Stopwatch()
    with tool_errors():
        loaded = await st.engine.load(list(dict.fromkeys(models)))
    return ModelsResult(loaded=loaded, latency_ms=sw.ms)


@server.tool(
    name="laya_unload_models",
    description=(
        "Free memory by unloading checkpoints (all of them when `models` is omitted). They reload automatically, "
        "with a cold-start delay, on the next request that needs them. Rarely needed; it slows other teams' calls."
    ),
    tags={"ops"},
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False,
                 "title": "Unload models"},
)
async def laya_unload_models(
    models: Annotated[list[ModelName] | None, Field(description="Checkpoints to unload; omit for all.")] = None,
) -> ModelsResult:
    st = get_state()
    sw = Stopwatch()
    with tool_errors():
        loaded = await st.engine.unload(list(dict.fromkeys(models)) if models else None)
    return ModelsResult(loaded=loaded, latency_ms=sw.ms)


@server.tool(
    name="laya_detect_language",
    description=(
        "Detect the script and (best-effort) language of some text or a JSON state, and which checkpoint "
        "auto-routing would use. Instant, no model call. Useful to pass `lang`/`model` explicitly or to spot "
        "mixed-language data before a batch."
    ),
    tags={"ops"},
    annotations={**READ_ONLY, "title": "Detect language"},
)
async def laya_detect_language(
    text: Annotated[State, Field(description="Text, or a JSON object/list whose string values are inspected.")],
) -> LanguageInfo:
    a: dict[str, Any] = laya.detect_language(text)
    english = bool(a.get("is_english"))
    return LanguageInfo(
        script=str(a.get("script", "unknown")),
        language=a.get("language"),
        language_undecided=bool(a.get("language_undecided", a.get("language") is None)),
        is_english=english,
        recommended_model="english" if english else "multilingual",
        non_latin_fraction=float(a.get("non_latin_fraction", 0.0)),
        script_profile={k: round(float(v), 4) for k, v in (a.get("script_profile") or {}).items()},
    )


@server.tool(
    name="laya_usage_stats",
    description=(
        "Usage per team, tool and schema over the last `since_hours`: calls, rows, p50/p95 latency, needs-review "
        "rate and errors. Use it to see what delegating to Laya costs and how often answers need a human/agent "
        "decision. Counts only; no content is stored."
    ),
    tags={"ops"},
    annotations={**READ_ONLY, "title": "Usage stats"},
)
async def laya_usage_stats(
    since_hours: Annotated[float, Field(gt=0, le=24 * 90, description="Look-back window in hours (max 90 days).")] = 24,
    team: Annotated[str | None, Field(description="Only this team; omit for all teams.")] = None,
) -> UsageStats:
    st = get_state()
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    with tool_errors():
        try:
            return await run_sync(st.repo.usage_stats, since, team=team.strip().lower() if team else None)
        except LookupError as e:
            raise ToolError(f"No usage data: {e}") from e
