"""Shared fixtures. Workstreams add their own fixtures in their own test modules."""
from pathlib import Path

import pytest

from laya_mcp.config import Settings
from tests.fakes import FakeEngine, FakeRouter


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, preload="english", warmup=False, threads=None)


@pytest.fixture
def fake_router() -> FakeRouter:
    return FakeRouter()


@pytest.fixture
def fake_engine(settings: Settings, fake_router: FakeRouter) -> FakeEngine:
    return FakeEngine(settings, fake_router)
