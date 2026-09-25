"""JobService: async batch and evaluation jobs.

CONTRACT (implemented by the data workstream):
- Huey with SqliteHuey(filename=Settings.jobs_db_path) dispatches work; the consumer runs as
  Settings.job_workers worker THREADS inside the server process (so they share the loaded
  model). Job state and per-item raw results are stored in SQLModel tables via Repo (not in
  Huey's result store) so results can be paginated and survive restarts.
- Items are processed in small groups; each state goes through
  `engine.predict_blocking([state], questions, model=..., lang=...)`, and its raw result is
  stored UNCALIBRATED (calibration is applied when results are read).
- Jobs can be built in chunks: submit_batch(..., append_to=job_id, finalize=False) keeps the job
  open. A job completes when finalized and every item is done. Cancel stops remaining items.
- On startup, unfinished items of queued/running jobs are re-enqueued.
- Completion hooks: `on_complete(kind, fn)` registers fn(job_id) called (in the worker thread)
  after a job of that kind completes; used by the workbench to build evaluation reports.

Implementation notes:
- Consumer: see jobs/queue.py (`InProcessConsumer`: huey's own Worker/Scheduler threads, no
  signal handlers). `immediate=True` (tests) runs every task synchronously inside submit().
- Counters: JobInfo.done counts PROCESSED items (succeeded + failed); JobInfo.failed is the
  failed subset. Progress = done / total. Failed items are excluded from results(); their
  errors are available from failures(job_id).
- A failing item never fails the job; the job ends 'failed' only when every item failed
  (hooks are not run then). Items are checked for cancellation/shutdown between items, so a
  cancel or stop waits for at most one in-flight prediction.
- Completion: exactly one worker claims a finished job, runs the kind's hooks, THEN marks it
  'completed', so a poller never sees a completed evaluate job without its result_ref. Hook
  exceptions are logged and stored in JobInfo.error; the job still completes.
- ETA = engine.estimate_seconds(rows), rows = this job's unprocessed items x its questions
  plus the unprocessed rows of older queued/running jobs (they are served first).
- start() treats the SQL tables as the source of truth: it flushes jobs.db's queue and
  re-enqueues every pending item of queued/running jobs. One JobService per jobs.db.
- Shutdown order in the app lifespan: jobs.stop() before engine.stop(). An item interrupted by
  shutdown stays pending (it is not counted as failed) and runs again after restart.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable

from ..config import Settings
from ..db.models import JobRecord
from ..db.repo import ACTIVE_JOB_STATUSES, NotFound, Repo
from ..engine.answers import to_item_result
from ..engine.runtime import InferenceEngine
from ..models import Answer, Detail, ItemResult, JobInfo, JobKind, JobResultsPage, State
from .queue import InProcessConsumer, create_huey, register_tasks

logger = logging.getLogger("laya_mcp.jobs")

JOB_KINDS: tuple[str, ...] = ("batch", "evaluate")


def answer_key(ans: Answer) -> str:
    """Summary/filter key of an answer: choice label, noul 'true'/'false', score rounded level."""
    if ans.type == "noul":
        return "true" if ans.value is True or str(ans.value).lower() == "true" else "false"
    if ans.type == "score":
        return str(int(math.floor(float(ans.value) + 0.5)))
    return str(ans.value)


class JobService:
    def __init__(
        self,
        settings: Settings,
        repo: Repo,
        engine: InferenceEngine,
        *,
        immediate: bool = False,
        chunk_size: int = 4,
        poll_interval: float = 1.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        """immediate=True executes tasks synchronously on submit (tests). chunk_size = items per
        huey task. poll_interval = max idle polling delay of the worker threads (s)."""
        self.settings = settings
        self.repo = repo
        self.engine = engine
        self.immediate = immediate
        self.chunk_size = max(1, int(chunk_size))
        self.poll_interval = poll_interval
        self.shutdown_timeout = shutdown_timeout
        self.huey = create_huey(settings, immediate=immediate)
        self._tasks = register_tasks(self.huey, self._process_items)
        self._hooks: dict[str, list[Callable[[str], None]]] = {k: [] for k in JOB_KINDS}
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._consumer: InProcessConsumer | None = None
        self._started = False

    # lifecycle -------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._started

    def start(self) -> None:
        """Start in-process Huey consumer threads and re-enqueue unfinished work."""
        with self._lock:
            if self._started:
                return
            self._stopping.clear()
            if not self.immediate:
                # jobs.db only carries dispatch messages; drop stale ones and rebuild from the tables.
                self.huey.storage.flush_queue()
                self.huey.storage.flush_schedule()
            self.repo.release_completion_claims()
            requeued = self._recover()
            if not self.immediate:
                self._consumer = InProcessConsumer(
                    self.huey, workers=self.settings.job_workers, poll_interval=self.poll_interval
                )
                self._consumer.start()
            self._started = True
        logger.info(
            "job service started (%s, %d worker thread(s)); re-enqueued %d pending item(s)",
            "immediate" if self.immediate else "in-process consumer",
            self.settings.job_workers,
            requeued,
        )

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            consumer, self._consumer = self._consumer, None
            self._started = False
        if consumer is not None:
            consumer.shutdown(timeout=self.shutdown_timeout)
        if not self.immediate:
            self.huey.storage.close()
        logger.info("job service stopped")

    def _recover(self) -> int:
        n = 0
        for job in self.repo.active_jobs():
            idxs = self.repo.pending_item_indexes(job.id)
            if idxs:
                self._enqueue(job.id, idxs)
                n += len(idxs)
            elif job.finalized:
                self._tasks.process_items(job.id, [])     # nothing left: completes it in a worker
        return n

    # submission ------------------------------------------------------------------------------
    def submit(
        self,
        *,
        team: str,
        kind: JobKind,
        items: list[State],
        questions: dict[str, dict[str, Any]],
        schema_ref: str | None = None,
        model: str | None = None,
        lang: str | None = None,
        append_to: str | None = None,
        finalize: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> JobInfo:
        """Create a job (or append items to an open one) and enqueue its items. `meta` is stored
        with the job (e.g. {'dataset': 'dev/issues'} for evaluate jobs).

        Appending: the job must be open (not finalized) and still queued/running (JobClosed, a
        ValueError, otherwise); appended items are answered with the job's original questions,
        model and lang, so `questions` may be empty or must equal the job's. `meta` is merged."""
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}; use one of {', '.join(JOB_KINDS)}")
        items = list(items)
        if append_to:
            job = self.repo.get_job(append_to)
            if job.kind != kind:
                raise ValueError(f"job {append_to} is a {job.kind} job, not {kind}")
            if questions and questions != job.questions:
                raise ValueError(
                    f"questions differ from those of job {append_to}; appended items use the job's "
                    "original questions (omit questions, or submit a new job)"
                )
            job, idxs = self.repo.append_job_items(append_to, items, finalize=finalize, meta=meta)
        else:
            if not questions:
                raise ValueError("a job needs at least one question")
            job = self.repo.create_job(
                kind=kind, team=team, questions=questions, states=items, schema_ref=schema_ref,
                model=model, lang=lang, meta=meta, finalized=finalize,
            )
            idxs = list(range(len(items)))
        self._enqueue(job.id, idxs)
        if finalize:
            self._maybe_complete(job.id)       # e.g. finalizing with no new items
        return self.get(job.id)

    def _enqueue(self, job_id: str, idxs: list[int]) -> None:
        for i in range(0, len(idxs), self.chunk_size):
            self._tasks.process_items(job_id, idxs[i : i + self.chunk_size])

    # queries ---------------------------------------------------------------------------------
    def get(self, job_id: str) -> JobInfo:
        return self._info(self.repo.get_job(job_id))

    def meta(self, job_id: str) -> dict[str, Any]:
        return dict(self.repo.get_job(job_id).meta or {})

    def list_jobs(self, team: str | None = None, limit: int = 20) -> list[JobInfo]:
        """Newest first (ops / status views)."""
        return [self._info(j) for j in self.repo.list_jobs(team=team, limit=limit)]

    def queue_depth(self) -> int:
        """Pending items across queued/running jobs."""
        return self.repo.pending_items_count()

    def failures(self, job_id: str) -> list[tuple[int, str]]:
        """[(item index, error)] for failed items, in index order."""
        self.repo.get_job(job_id)
        return self.repo.job_failures(job_id)

    def _info(self, job: JobRecord) -> JobInfo:
        active = job.status in ACTIVE_JOB_STATUSES
        return JobInfo(
            job_id=job.id,
            kind=job.kind,  # type: ignore[arg-type]
            status=job.status,  # type: ignore[arg-type]
            team=job.team,
            total=job.total,
            done=job.done,
            failed=job.failed,
            schema_ref=job.schema_ref,
            eta_seconds=round(self._eta_seconds(job), 1) if active else None,
            created_at=job.created_at,
            finished_at=None if active else job.finished_at,
            error=job.error,
            result_ref=dict(job.result_ref or {}),
        )

    def _eta_seconds(self, job: JobRecord) -> float:
        rows = max(0, job.total - job.done) * max(1, len(job.questions))
        for other in self.repo.active_jobs():
            if other.id != job.id and (other.created_at, other.id) < (job.created_at, job.id):
                rows += max(0, other.total - other.done) * max(1, len(other.questions))
        return float(self.engine.estimate_seconds(rows)) if rows else 0.0

    def results(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
        needs_review_only: bool = False,
        question_id: str | None = None,
        value: str | None = None,
        detail: Detail = "compact",
        temperatures: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        default_threshold: float | None = None,
    ) -> JobResultsPage:
        """Paginated normalised results (via engine.answers.to_item_result) + whole-job summary.
        Filters apply before pagination. value compares against str(Answer.value) lowercased
        (noul 'true'/'false', score rounded level).

        default_threshold None -> Settings.default_threshold, so needs_review is always
        meaningful. question_id narrows both filters to that question; value without question_id
        matches any question. Only successfully processed items are returned (see failures())."""
        job = self.repo.get_job(job_id)
        offset, limit = max(0, int(offset)), max(1, int(limit))
        if default_threshold is None:
            default_threshold = self.settings.default_threshold
        want = value.strip().lower() if value is not None else None
        norm = dict(temperatures=temperatures, thresholds=thresholds, default_threshold=default_threshold)

        counts: dict[str, dict[str, int]] = {}
        matched: list[tuple[int, dict[str, Any], ItemResult]] = []
        for idx, raw in self.repo.job_raw_results(job_id):
            item = to_item_result(idx, raw, detail="compact", **norm)
            for qid, ans in item.answers.items():
                c = counts.setdefault(qid, {})
                key = answer_key(ans)
                c[key] = c.get(key, 0) + 1
            if self._matches(item, needs_review_only, question_id, want):
                matched.append((idx, raw, item))

        page = matched[offset : offset + limit]
        items = [it if detail == "compact" else to_item_result(idx, raw, detail=detail, **norm) for idx, raw, it in page]
        summary = {qid: dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) for qid, c in counts.items()}
        return JobResultsPage(
            job=self._info(job),
            items=items,
            offset=offset,
            next_offset=offset + limit if offset + limit < len(matched) else None,
            summary=summary,
        )

    @staticmethod
    def _matches(item: ItemResult, needs_review_only: bool, question_id: str | None, want: str | None) -> bool:
        if needs_review_only:
            if question_id is not None:
                if question_id not in item.needs_review:
                    return False
            elif not item.needs_review:
                return False
        if want is not None:
            if question_id is not None:
                ans = item.answers.get(question_id)
                return ans is not None and answer_key(ans).lower() == want
            return any(answer_key(a).lower() == want for a in item.answers.values())
        return True

    def raw_results(self, job_id: str) -> list[tuple[int, dict[str, Any]]]:
        """[(item index, raw laya result)] for all completed items, in index order."""
        self.repo.get_job(job_id)
        return self.repo.job_raw_results(job_id)

    def states(self, job_id: str, indices: list[int] | None = None) -> dict[int, State]:
        """{item index: submitted state}, e.g. to show items for human labelling."""
        self.repo.get_job(job_id)
        return self.repo.job_states(job_id, indices)

    # control ---------------------------------------------------------------------------------
    def cancel(self, job_id: str) -> JobInfo:
        """Cancel remaining items (no-op for finished jobs). Results of processed items stay readable."""
        job = self.repo.cancel_job(job_id)
        return self._info(job)

    def set_result_ref(self, job_id: str, ref: dict[str, Any]) -> None:
        self.repo.set_job_result_ref(job_id, ref)

    def on_complete(self, kind: JobKind, fn: Callable[[str], None]) -> None:
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        self._hooks[kind].append(fn)

    # worker side -----------------------------------------------------------------------------
    def _process_items(self, job_id: str, idxs: list[int]) -> None:
        """Huey task body (worker thread, or the caller's thread in immediate mode). Idempotent."""
        try:
            job = self.repo.get_job(job_id)
        except NotFound:
            logger.warning("job %s no longer exists; dropping %d queued item(s)", job_id, len(idxs))
            return
        if job.status not in ACTIVE_JOB_STATUSES:
            return
        for idx in idxs:
            if self._stopping.is_set():
                return                          # left pending; re-enqueued by the next start()
            claim, state = self.repo.start_job_item(job_id, idx)
            if claim == "stop":
                return
            if claim == "skip":
                continue
            try:
                raw = self.engine.predict_blocking([state], job.questions, model=job.model, lang=job.lang)[0]
            except Exception as exc:
                if self._stopping.is_set():
                    logger.info("job %s item %d interrupted by shutdown; will retry after restart", job_id, idx)
                    return
                logger.warning("job %s item %d failed: %s: %s", job_id, idx, type(exc).__name__, exc)
                self.repo.finish_job_item(job_id, idx, error=f"{type(exc).__name__}: {exc}")
            else:
                self.repo.finish_job_item(job_id, idx, raw=raw)
        self._maybe_complete(job_id)

    def _maybe_complete(self, job_id: str) -> None:
        job = self.repo.claim_job_completion(job_id)
        if job is None:
            return
        if job.status == "failed":
            logger.warning("job %s failed: %s", job_id, job.error)
            return
        error = self._run_hooks(job)
        self.repo.complete_job(job_id, error=error)
        logger.info("job %s (%s) completed: %d items, %d failed", job_id, job.kind, job.total, job.failed)

    def _run_hooks(self, job: JobRecord) -> str | None:
        errors: list[str] = []
        for fn in list(self._hooks.get(job.kind, [])):
            try:
                fn(job.id)
            except Exception as exc:
                logger.exception("completion hook %r failed for job %s", fn, job.id)
                errors.append(f"{getattr(fn, '__name__', 'hook')}: {type(exc).__name__}: {exc}")
        return ("completion hook failed: " + "; ".join(errors)) if errors else None
