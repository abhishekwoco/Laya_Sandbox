"""JobService: lifecycle, chunked building, results, cancel, failures, hooks, restart recovery.

Most tests use Huey immediate mode (tasks run inside submit()); the threaded tests run the real
in-process consumer (SqliteHuey on jobs.db + worker threads) against FakeEngine.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

from laya_mcp.config import Settings
from laya_mcp.db.repo import JobClosed, NotFound, Repo
from laya_mcp.jobs.service import JobService, answer_key
from laya_mcp.models import Answer
from tests.fakes import FakeEngine, FakeRouter

Q: dict[str, dict[str, Any]] = {
    "team": {"type": "choice", "instructions": "Which team owns this?",
             "criteria": {"backend": "server database", "frontend": "button layout"}},
    "bug": {"type": "noul", "instructions": "Is this a bug?"},
    "sev": {"type": "score", "instructions": "How severe is it?", "criteria": ["trivial cosmetic", "major outage"]},
}
# 6 backend / bug / severe items followed by 4 frontend / not-a-bug / cosmetic items
ITEMS = [f"backend server outage bug #{i}" for i in range(6)] + [f"frontend button cosmetic glitch #{i}" for i in range(4)]


def wait_for(pred: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for condition")


class FlakyEngine(FakeEngine):
    """Fails any state containing 'boom'; records which threads ran predictions."""

    def __init__(self, settings: Settings, router: FakeRouter | None = None) -> None:
        super().__init__(settings, router)
        self.threads: set[str] = set()

    def predict_blocking(self, states, questions, *, model=None, lang=None):
        self.threads.add(threading.current_thread().name)
        if any("boom" in str(s) for s in states):
            raise RuntimeError("model exploded")
        return super().predict_blocking(states, questions, model=model, lang=lang)


@pytest.fixture
def repo(settings: Settings):
    r = Repo(settings.db_url)
    r.migrate()
    yield r
    r.close()


@pytest.fixture
def jobs(settings: Settings, repo: Repo, fake_engine: FakeEngine):
    svc = JobService(settings, repo, fake_engine, immediate=True)
    svc.start()
    yield svc
    svc.stop()


def threaded(settings: Settings, repo: Repo, engine: Any, **kw: Any) -> JobService:
    return JobService(settings, repo, engine, poll_interval=0.05, shutdown_timeout=10, **kw)


# lifecycle --------------------------------------------------------------------------------------
def test_submit_runs_to_completion_and_stores_raw_results(jobs, fake_router):
    info = jobs.submit(team="dev", kind="batch", items=ITEMS, questions=Q, schema_ref="dev/triage@1", model="english")
    assert (info.status, info.total, info.done, info.failed) == ("completed", 10, 10, 0)
    assert info.kind == "batch" and info.team == "dev" and info.schema_ref == "dev/triage@1"
    assert info.eta_seconds is None and info.finished_at is not None and info.created_at.tzinfo is not None
    assert fake_router.calls == 10                                   # one prediction per item

    raw = jobs.raw_results(info.job_id)
    assert [i for i, _ in raw] == list(range(10))
    # stored uncalibrated, exactly as the engine returned it
    assert raw[0][1] == fake_router.predict(ITEMS[0], Q, model="english")
    assert raw[0][1]["routing"]["model"] == "english"


def test_chunked_build_append_and_finalize(jobs):
    info = jobs.submit(team="dev", kind="batch", items=ITEMS[:3], questions=Q, finalize=False)
    assert info.status == "running" and (info.done, info.total) == (3, 3)      # processed, but still open

    info = jobs.submit(team="dev", kind="batch", items=ITEMS[3:5], questions={}, append_to=info.job_id, finalize=False)
    assert info.status == "running" and info.total == 5
    with pytest.raises(ValueError, match="questions differ"):
        jobs.submit(team="dev", kind="batch", items=["x"], questions={"other": Q["bug"]}, append_to=info.job_id)
    with pytest.raises(ValueError, match="is a batch job"):
        jobs.submit(team="dev", kind="evaluate", items=["x"], questions={}, append_to=info.job_id)

    info = jobs.submit(team="dev", kind="batch", items=ITEMS[5:], questions=Q, append_to=info.job_id,
                       finalize=True, meta={"source": "chunk-3"})
    assert (info.status, info.total, info.done) == ("completed", 10, 10)
    assert [i for i, _ in jobs.raw_results(info.job_id)] == list(range(10))
    assert jobs.meta(info.job_id) == {"source": "chunk-3"}

    with pytest.raises(JobClosed):
        jobs.submit(team="dev", kind="batch", items=["late"], questions={}, append_to=info.job_id)


def test_finalize_with_no_new_items_completes(jobs):
    info = jobs.submit(team="dev", kind="batch", items=ITEMS[:2], questions=Q, finalize=False)
    assert info.status == "running"
    info = jobs.submit(team="dev", kind="batch", items=[], questions={}, append_to=info.job_id, finalize=True)
    assert info.status == "completed" and info.total == 2


def test_invalid_submissions_and_not_found(jobs):
    with pytest.raises(ValueError, match="unknown job kind"):
        jobs.submit(team="dev", kind="bogus", items=["x"], questions=Q)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one question"):
        jobs.submit(team="dev", kind="batch", items=["x"], questions={})
    for call in (jobs.get, jobs.meta, jobs.raw_results, jobs.cancel, jobs.failures,
                 lambda j: jobs.results(j), lambda j: jobs.set_result_ref(j, {"a": 1})):
        with pytest.raises(NotFound):
            call("0" * 32)
    with pytest.raises(NotFound):
        jobs.submit(team="dev", kind="batch", items=["x"], questions={}, append_to="0" * 32)


# results ----------------------------------------------------------------------------------------
def test_results_pagination_filters_and_summary(jobs):
    job_id = jobs.submit(team="dev", kind="batch", items=ITEMS, questions=Q).job_id

    page = jobs.results(job_id, limit=4)
    assert [i.index for i in page.items] == [0, 1, 2, 3] and page.offset == 0 and page.next_offset == 4
    assert page.summary == {"team": {"backend": 6, "frontend": 4}, "bug": {"true": 6, "false": 4}, "sev": {"1": 6, "0": 4}}
    assert page.job.status == "completed"
    last = jobs.results(job_id, offset=8, limit=4)
    assert [i.index for i in last.items] == [8, 9] and last.next_offset is None
    assert last.summary == page.summary                                   # whole-job summary, not per page

    a0 = page.items[0].answers
    assert a0["team"].value == "backend" and a0["bug"].value is True and a0["bug"].p_true == 0.9
    assert a0["sev"].level == 1 and a0["team"].probabilities is None      # compact by default

    # value filter (case-insensitive), narrowed to a question, applied before pagination
    fe = jobs.results(job_id, question_id="team", value="FRONTEND", limit=3)
    assert [i.index for i in fe.items] == [6, 7, 8] and fe.next_offset == 3
    fe2 = jobs.results(job_id, question_id="team", value="frontend", offset=3, limit=3)
    assert [i.index for i in fe2.items] == [9] and fe2.next_offset is None
    assert [i.index for i in jobs.results(job_id, value="true").items] == list(range(6))     # any question
    assert [i.index for i in jobs.results(job_id, question_id="sev", value="0").items] == [6, 7, 8, 9]
    assert jobs.results(job_id, question_id="team", value="nobody").items == []

    # needs_review filter uses the thresholds passed in
    nr = jobs.results(job_id, needs_review_only=True, thresholds={"bug": 0.95}, default_threshold=0.0, limit=100)
    assert len(nr.items) == 10 and all(i.needs_review == ["bug"] for i in nr.items)
    assert all(i.answers["bug"].status == "needs_review" and i.answers["team"].status == "decided" for i in nr.items)
    assert jobs.results(job_id, needs_review_only=True, default_threshold=0.0).items == []
    assert jobs.results(job_id, needs_review_only=True, question_id="team",
                        thresholds={"bug": 0.95}, default_threshold=0.0).items == []
    # default_threshold None falls back to Settings.default_threshold, so statuses are always set
    assert all(a.status in ("decided", "needs_review") for a in jobs.results(job_id).items[0].answers.values())


def test_results_detail_and_calibration(jobs):
    job_id = jobs.submit(team="dev", kind="batch", items=ITEMS, questions=Q).job_id
    full = jobs.results(job_id, detail="full", limit=1).items[0]
    assert set(full.answers["team"].probabilities) == {"backend", "frontend"}
    assert full.answers["sev"].legend == {"0": "trivial cosmetic", "1": "major outage"}

    plain = jobs.results(job_id, limit=1).items[0].answers["team"].confidence
    softened = jobs.results(job_id, limit=1, temperatures={"team": 3.0}).items[0].answers["team"].confidence
    assert softened < plain
    # raw results stay uncalibrated
    assert jobs.raw_results(job_id)[0][1]["answers"]["team"]["probabilities"]["backend"] == pytest.approx(plain, abs=1e-3)


def test_answer_key():
    assert answer_key(Answer(type="noul", value=True, confidence=0.9)) == "true"
    assert answer_key(Answer(type="noul", value=False, confidence=0.9)) == "false"
    assert answer_key(Answer(type="score", value=2.5, confidence=0.5)) == "3"
    assert answer_key(Answer(type="score", value=1.49, confidence=0.5)) == "1"
    assert answer_key(Answer(type="choice", value="Backend", confidence=0.9)) == "Backend"


# failures ---------------------------------------------------------------------------------------
def test_item_failures_are_counted_not_fatal(settings, repo):
    jobs = JobService(settings, repo, FlakyEngine(settings), immediate=True)
    jobs.start()
    items = ["backend server", "boom 1", "frontend button", "boom 2"]
    info = jobs.submit(team="dev", kind="batch", items=items, questions=Q)
    assert (info.status, info.done, info.failed, info.error) == ("completed", 4, 2, None)
    assert jobs.failures(info.job_id) == [(1, "RuntimeError: model exploded"), (3, "RuntimeError: model exploded")]
    page = jobs.results(info.job_id)
    assert [i.index for i in page.items] == [0, 2]                       # failed items are not results
    assert sum(page.summary["team"].values()) == 2
    jobs.stop()


def test_job_fails_only_when_every_item_fails(jobs):
    hook_calls: list[str] = []
    jobs.on_complete("batch", hook_calls.append)
    bad_q = {"q": {"type": "essay", "instructions": "Write something"}}      # FakeRouter rejects the type
    info = jobs.submit(team="dev", kind="batch", items=["a", "b", "c"], questions=bad_q)
    assert (info.status, info.done, info.failed) == ("failed", 3, 3)
    assert "all 3 items failed" in info.error and "unknown type" in info.error
    assert hook_calls == []                                               # hooks only for completed jobs


# hooks ------------------------------------------------------------------------------------------
def test_completion_hooks_per_kind_and_errors(jobs):
    seen: list[tuple[str, str]] = []

    def build_report(job_id: str) -> None:
        seen.append(("evaluate", job_id))
        jobs.set_result_ref(job_id, {"report_id": 7})

    def broken(job_id: str) -> None:
        raise RuntimeError("report store offline")

    jobs.on_complete("evaluate", build_report)
    jobs.on_complete("batch", lambda j: seen.append(("batch", j)))

    ev = jobs.submit(team="dev", kind="evaluate", items=ITEMS[:2], questions=Q, meta={"dataset": "dev/issues"})
    assert seen == [("evaluate", ev.job_id)]
    assert ev.status == "completed" and ev.result_ref == {"report_id": 7} and ev.error is None
    assert jobs.meta(ev.job_id) == {"dataset": "dev/issues"}

    jobs.on_complete("evaluate", broken)
    ev2 = jobs.submit(team="dev", kind="evaluate", items=ITEMS[:1], questions=Q)
    assert ev2.status == "completed"                                      # hook failure doesn't fail the job
    assert "report store offline" in ev2.error and ev2.result_ref == {"report_id": 7}
    assert seen[-1] == ("evaluate", ev2.job_id)

    b = jobs.submit(team="dev", kind="batch", items=ITEMS[:1], questions=Q)
    assert seen[-1] == ("batch", b.job_id) and len(seen) == 3
    with pytest.raises(ValueError):
        jobs.on_complete("nope", print)  # type: ignore[arg-type]


# cancel -----------------------------------------------------------------------------------------
def test_cancel_stops_remaining_items(settings, repo):
    router = FakeRouter(delay_s=0.03)
    jobs = threaded(settings, repo, FakeEngine(settings, router))
    jobs.start()
    try:
        info = jobs.submit(team="dev", kind="batch", items=[f"backend item {i}" for i in range(60)], questions=Q)
        wait_for(lambda: jobs.get(info.job_id).done >= 3)
        cancelled = jobs.cancel(info.job_id)
        assert cancelled.status == "cancelled" and cancelled.finished_at is not None and cancelled.eta_seconds is None
        time.sleep(0.3)                                                   # let in-flight/queued chunks drain
        after = jobs.get(info.job_id)
        assert after.status == "cancelled" and after.done < 60
        assert after.done <= cancelled.done + 1                            # at most the in-flight item finished
        assert repo.pending_item_indexes(info.job_id) == []
        assert router.calls <= after.done + 1
        page = jobs.results(info.job_id, limit=100)
        assert len(page.items) == after.done                               # processed results stay readable
        # cancelling a finished job is a no-op
        done_job = jobs.submit(team="dev", kind="batch", items=["x"], questions=Q)
        wait_for(lambda: jobs.get(done_job.job_id).status == "completed")
        assert jobs.cancel(done_job.job_id).status == "completed"
    finally:
        jobs.stop()


# threaded consumer ------------------------------------------------------------------------------
def test_real_threaded_consumer(settings, repo):
    settings.job_workers = 2
    engine = FlakyEngine(settings)
    jobs = threaded(settings, repo, engine)
    hook_threads: list[str] = []
    jobs.on_complete("batch", lambda j: hook_threads.append(threading.current_thread().name))
    jobs.start()
    try:
        a = jobs.submit(team="dev", kind="batch", items=ITEMS, questions=Q)
        b = jobs.submit(team="sales", kind="batch", items=ITEMS[:5], questions={"bug": Q["bug"]})
        assert a.status in ("queued", "running") and a.eta_seconds is not None
        wait_for(lambda: all(jobs.get(j).status == "completed" for j in (a.job_id, b.job_id)))
        assert jobs.get(a.job_id).done == 10 and jobs.get(b.job_id).done == 5
        assert jobs.results(a.job_id).summary["team"] == {"backend": 6, "frontend": 4}
        assert settings.jobs_db_path.exists()
        assert engine.threads and all(t.startswith("Worker-") for t in engine.threads)
        assert len(hook_threads) == 2                                        # hooks ran before 'completed'
        assert all(t.startswith("Worker-") for t in hook_threads)          # ... in worker threads
        assert jobs.queue_depth() == 0 and len(jobs.huey) == 0
        assert {j.job_id for j in jobs.list_jobs()} == {a.job_id, b.job_id}
        assert [j.job_id for j in jobs.list_jobs(team="sales")] == [b.job_id]
    finally:
        jobs.stop()
    assert not any(t.name.startswith("Worker-") and t.is_alive() for t in threading.enumerate())


def test_eta_counts_older_jobs_and_queue_depth(settings, repo):
    jobs = threaded(settings, repo, FakeEngine(settings))            # not started: nothing is processed
    a = jobs.submit(team="dev", kind="batch", items=ITEMS, questions=Q)                  # 10 x 3 rows
    b = jobs.submit(team="dev", kind="batch", items=ITEMS[:4], questions={"bug": Q["bug"]})   # 4 x 1 rows
    assert (a.status, b.status) == ("queued", "queued")
    assert a.eta_seconds == pytest.approx(30 * 0.001, abs=0.05)
    assert jobs.get(b.job_id).eta_seconds == pytest.approx((30 + 4) * 0.001, abs=0.05)
    assert jobs.queue_depth() == 14
    jobs.stop()


# restart recovery -------------------------------------------------------------------------------
def test_restart_recovers_pending_items(settings, repo):
    # "Old process": jobs submitted, one item processed, then the process dies before its consumer ran.
    old = threaded(settings, repo, FakeEngine(settings))
    j1 = old.submit(team="dev", kind="batch", items=ITEMS, questions=Q)
    j2 = old.submit(team="dev", kind="evaluate", items=ITEMS[:3], questions=Q, finalize=False)
    assert len(old.huey) > 0                                         # dispatch messages sit in jobs.db
    claim, state = repo.start_job_item(j1.job_id, 0)
    assert claim == "run"
    repo.finish_job_item(j1.job_id, 0, raw=FakeRouter().predict(state, Q))
    old.stop()
    assert old.get(j1.job_id).status == "running" and old.get(j1.job_id).done == 1

    # "New process": start() flushes stale messages and re-enqueues pending items from the tables.
    router = FakeRouter()
    new = threaded(settings, repo, FakeEngine(settings, router))
    completed: list[str] = []
    new.on_complete("batch", completed.append)
    new.on_complete("evaluate", completed.append)
    new.start()
    try:
        wait_for(lambda: new.get(j1.job_id).status == "completed")
        info = new.get(j1.job_id)
        assert (info.done, info.total, info.failed) == (10, 10, 0)
        wait_for(lambda: new.get(j2.job_id).done == 3)
        assert new.get(j2.job_id).status == "running"                  # still open: not finalized
        assert router.calls == 9 + 3                                   # nothing processed twice
        new.submit(team="dev", kind="evaluate", items=[], questions={}, append_to=j2.job_id, finalize=True)
        wait_for(lambda: new.get(j2.job_id).status == "completed")
        assert sorted(completed) == sorted([j1.job_id, j2.job_id])
        assert len(new.huey) == 0
    finally:
        new.stop()


def test_restart_after_stop_mid_job(settings, repo):
    router = FakeRouter(delay_s=0.03)
    first = threaded(settings, repo, FakeEngine(settings, router))
    first.start()
    info = first.submit(team="dev", kind="batch", items=[f"backend {i}" for i in range(30)], questions=Q)
    wait_for(lambda: first.get(info.job_id).done >= 2)
    first.stop()                                                      # waits for the in-flight item only
    stopped = first.get(info.job_id)
    assert stopped.status == "running" and 2 <= stopped.done < 30 and stopped.failed == 0

    second = threaded(settings, repo, FakeEngine(settings, FakeRouter()))
    second.start()
    try:
        wait_for(lambda: second.get(info.job_id).status == "completed")
        final = second.get(info.job_id)
        assert (final.done, final.failed) == (30, 0)
        assert [i for i, _ in second.raw_results(info.job_id)] == list(range(30))
    finally:
        second.stop()


def test_restart_completes_job_claimed_before_crash(settings, repo, fake_engine):
    job = repo.create_job(kind="evaluate", team="dev", questions=Q, states=["backend server"], finalized=True)
    _, state = repo.start_job_item(job.id, 0)
    repo.finish_job_item(job.id, 0, raw=FakeRouter().predict(state, Q))
    assert repo.claim_job_completion(job.id) is not None              # crash while hooks were running
    assert repo.claim_job_completion(job.id) is None                  # claim is exclusive
    assert repo.get_job(job.id).status == "running"

    jobs = JobService(settings, repo, fake_engine, immediate=True)
    hooks: list[str] = []
    jobs.on_complete("evaluate", hooks.append)
    jobs.start()
    assert hooks == [job.id]
    assert jobs.get(job.id).status == "completed"
    jobs.stop()
