"""Huey wiring for laya-mcp jobs: SqliteHuey factory, task registration, in-process consumer.

Dispatch only: Huey carries "process these item indexes of job X" messages. Job state and the
per-item results live in the SQL tables (db/repo.py), which are the source of truth; Huey's
result store is disabled.

In-process consumer (why a Consumer subclass): huey 3.4's `Consumer.start()` spawns the worker
and scheduler threads and then installs SIGINT/SIGTERM/SIGHUP handlers. `signal.signal()` only
works in the main thread and would replace uvicorn's own handlers, and `Consumer.run()` blocks
the caller and can `os.execl()` the process on SIGHUP. `InProcessConsumer` keeps huey's Worker
and Scheduler thread loops (dequeue/execute, retry scheduling) and its worker health check, but
installs no signal handlers and runs the health check from a daemon supervisor thread. The
server lifespan calls `JobService.start()/stop()`, which drive it. Worker threads live in the
server process, so they share the loaded Laya model with the interactive tools.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from huey import SqliteHuey
from huey.api import TaskWrapper
from huey.constants import WORKER_THREAD
from huey.consumer import Consumer, ConsumerStopped

from ..config import Settings

logger = logging.getLogger("laya_mcp.jobs")

HUEY_NAME = "laya-jobs"
PROCESS_TASK_NAME = "laya_process_job_items"


def create_huey(settings: Settings, *, immediate: bool = False) -> SqliteHuey:
    """SqliteHuey on Settings.jobs_db_path. immediate=True (tests) executes tasks synchronously
    inside enqueue() on in-memory storage and never touches jobs.db."""
    if not immediate:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
    huey_logger = logging.getLogger("huey")
    if huey_logger.level == logging.NOTSET:
        # huey logs every task execution at INFO; keep the server log readable unless configured.
        huey_logger.setLevel(logging.WARNING)
    return SqliteHuey(
        name=HUEY_NAME,
        filename=str(settings.jobs_db_path),
        immediate=immediate,
        results=False,        # results live in the job_items table
        utc=True,
        strict_fifo=True,     # AUTOINCREMENT ids: chunks run in enqueue order
        timeout=30,           # sqlite busy timeout (s)
    )


@dataclass
class JobTasks:
    process_items: TaskWrapper


def register_tasks(huey: SqliteHuey, process_items: Callable[[str, list[int]], None]) -> JobTasks:
    """Register the job tasks on this Huey instance. `process_items(job_id, idxs)` must be
    idempotent (it skips items that are no longer pending), which makes huey's retries and
    restart re-enqueueing safe."""

    @huey.task(name=PROCESS_TASK_NAME, retries=2, retry_delay=2)
    def _process(job_id: str, idxs: list[int]) -> None:
        process_items(job_id, list(idxs))

    return JobTasks(process_items=_process)


class InProcessConsumer(Consumer):
    """huey Consumer that runs as threads inside the server process (see module doc)."""

    def __init__(self, huey: SqliteHuey, workers: int = 1, *, poll_interval: float = 1.0, **kw: Any) -> None:
        super().__init__(
            huey,
            workers=max(1, workers),
            periodic=False,
            initial_delay=min(0.1, poll_interval),
            backoff=1.15,
            max_delay=poll_interval,     # idle polling backs off to at most this many seconds
            scheduler_interval=1,
            worker_type=WORKER_THREAD,
            check_worker_health=True,
            health_check_interval=10,
            **kw,
        )
        self._supervisor: threading.Thread | None = None

    def _set_signal_handlers(self) -> None:
        """No signal handlers: we may not be on the main thread, and the server owns signals."""

    def start(self) -> None:
        super().start()
        self._supervisor = threading.Thread(target=self._supervise, name="laya-jobs-supervisor", daemon=True)
        self._supervisor.start()

    def _supervise(self) -> None:
        health_ts = time.monotonic()
        while True:
            try:
                health_ts = self.loop(health_ts)     # waits on the stop flag, restarts dead workers
            except ConsumerStopped:
                return
            except Exception:  # pragma: no cover - defensive
                logger.exception("job consumer supervisor error")
                time.sleep(1.0)

    def shutdown(self, timeout: float | None = 30.0) -> None:
        """Set the stop flag and wait (up to `timeout`) for workers to finish their current task."""
        self.shutdown_timeout = timeout
        self.stop(graceful=True)
        if self._supervisor is not None:
            self._supervisor.join(timeout=5)
        self.huey.notify_interrupted_tasks()
