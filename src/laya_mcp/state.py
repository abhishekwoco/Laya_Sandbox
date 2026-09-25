"""Process-wide application state and per-request helpers shared by all tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastmcp.server.dependencies import get_http_headers

from .config import Settings

if TYPE_CHECKING:
    from .db.repo import Repo
    from .engine.runtime import InferenceEngine
    from .jobs.service import JobService


@dataclass
class AppState:
    settings: Settings
    engine: "InferenceEngine"
    repo: "Repo"
    jobs: "JobService"


_state: AppState | None = None


def set_state(state: AppState | None) -> None:
    global _state
    _state = state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("laya-mcp is not started: AppState has not been set")
    return _state


def current_team(explicit: str | None = None) -> str:
    """Team for this call: explicit argument > X-Laya-Team header > Settings.default_team."""
    settings = get_state().settings
    if explicit:
        return explicit.strip().lower()
    try:
        headers = get_http_headers(include={settings.team_header})
    except Exception:  # not inside an HTTP request (in-memory client, tests)
        headers = {}
    team = headers.get(settings.team_header)
    return team.strip().lower() if team else settings.default_team
