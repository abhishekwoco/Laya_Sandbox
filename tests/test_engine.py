"""InferenceEngine with FakeRouter: ordering, scheduling, limits, validation and status."""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from laya_mcp.config import Settings
from laya_mcp.engine import budget
from laya_mcp.engine.runtime import (
    DEFAULT_MS_PER_ROW,
    KNOWN_MODELS,
    BudgetExceeded,
    EngineBusy,
    InferenceEngine,
    InvalidQuestions,
)
from tests.fakes import FakeRouter, fake_answer

Q = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {"refund": "money back for a charge", "bug": "a software defect or crash"},
    },
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}


class TrackingRouter(FakeRouter):
    """FakeRouter that records call order and the peak number of concurrent passes."""

    def __init__(self, delay_s: float = 0.0) -> None:
        super().__init__(delay_s)
        self.log: list[Any] = []
        self.active = 0
        self.max_active = 0
        self._lk = threading.Lock()

    def predict(self, state, questions, model=None, task=None, lang=None, lang_guess=None):
        with self._lk:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.log.append(state)
        try:
            return super().predict(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
        finally:
            with self._lk:
                self.active -= 1


def with_settings(settings: Settings, **update: Any) -> Settings:
    return settings.model_copy(update=update)


async def started(settings: Settings, router: Any | None = None, **update: Any) -> InferenceEngine:
    eng = InferenceEngine(with_settings(settings, **update) if update else settings, router or FakeRouter())
    await eng.start()
    return eng


async def wait_until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


@pytest.fixture
def restore_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


# --------------------------------------------------------------------------- results
async def test_predict_returns_one_raw_result_per_state_in_order(settings):
    eng = await started(settings)
    states = ["please refund me", {"message": "the app has a bug"}, "urgent: call me back"]
    out = await eng.predict(states, Q)
    assert len(out) == 3
    for s, r in zip(states, out):
        assert r["answers"] == {qid: fake_answer(s, qid, q) for qid, q in Q.items()}
        assert r["routing"]["model"] == "english"
        assert set(r) >= {"model", "answers", "usage", "routing"}
    assert [r["answers"]["intent"]["choice"] for r in out[:2]] == ["refund", "bug"]
    assert out[2]["answers"]["urgent"]["noul"] == 0.9


async def test_predict_blocking_matches_predict(settings):
    eng = await started(settings)
    states = ["refund please", "a crash in the app"]
    assert await asyncio.to_thread(eng.predict_blocking, states, Q) == await eng.predict(states, Q)


async def test_empty_states_and_model_alias(settings):
    router = TrackingRouter()
    eng = await started(settings, router)
    assert await eng.predict([], Q) == []
    assert router.calls == 0
    out = await eng.predict(["hello"], Q, model="en")   # laya alias -> english
    assert out[0]["routing"]["model"] == "english"
    with pytest.raises(ValueError, match="unknown model"):
        await eng.predict(["hello"], Q, model="gpt-9")


async def test_not_started_and_stopped(settings):
    eng = InferenceEngine(settings, FakeRouter())
    with pytest.raises(RuntimeError, match="not started"):
        await eng.predict(["x"], Q)
    with pytest.raises(RuntimeError, match="not started"):
        eng.predict_blocking(["x"], Q)
    await eng.start()
    await eng.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await eng.predict(["x"], Q)


async def test_warmup_runs_once_and_is_not_timed(settings):
    router = TrackingRouter()
    eng = await started(settings, router, warmup=True)
    assert router.calls == 1
    assert eng.status().ms_per_row is None


async def test_thread_cap_applies_in_the_threads_that_run_passes(settings, restore_threads):
    """torch.set_num_threads is per OS thread under OpenMP, so the cap is applied in whichever
    thread runs a pass (asyncio worker threads, job worker threads)."""
    seen: list[int] = []

    class Recording(FakeRouter):
        def predict(self, state, questions, **kw):
            seen.append(torch.get_num_threads())
            return super().predict(state, questions, **kw)

    torch.set_num_threads(3)
    eng = await started(settings, Recording(), threads=2, warmup=True)
    await eng.predict(["interactive"], Q)
    await asyncio.to_thread(eng.predict_blocking, ["batch"], Q)
    assert seen == [2, 2, 2]


# --------------------------------------------------------------------------- scheduling
async def test_exactly_one_pass_at_a_time(settings):
    router = TrackingRouter(delay_s=0.03)
    eng = await started(settings, router, max_waiting_requests=16)
    batch = asyncio.create_task(asyncio.to_thread(eng.predict_blocking, [f"b{i}" for i in range(5)], Q))
    results = await asyncio.gather(*(eng.predict([f"i{i}a", f"i{i}b"], Q) for i in range(6)))
    await batch
    assert all(len(r) == 2 for r in results)
    assert router.calls == 5 + 12
    assert router.max_active == 1


async def test_engine_busy_when_waiting_limit_reached(settings):
    router = TrackingRouter(delay_s=0.4)
    eng = await started(settings, router, max_waiting_requests=1)
    first = asyncio.create_task(eng.predict(["first"], Q))
    await wait_until(lambda: eng.status().busy)
    second = asyncio.create_task(eng.predict(["second"], Q))
    await wait_until(lambda: eng.status().waiting == 1)
    with pytest.raises(EngineBusy) as ei:
        await eng.predict(["third"], Q)
    assert ei.value.waiting == 1 and ei.value.retry_after_s >= 1
    assert "laya_classify_batch" in str(ei.value)
    await asyncio.gather(first, second)
    assert router.log == ["first", "second"]
    assert eng.status().waiting == 0


async def test_zero_waiting_limit_still_admits_an_idle_engine(settings):
    router = TrackingRouter(delay_s=0.3)
    eng = await started(settings, router, max_waiting_requests=0)
    first = asyncio.create_task(eng.predict(["first"], Q))
    await wait_until(lambda: eng.status().busy)
    with pytest.raises(EngineBusy):
        await eng.predict(["second"], Q)
    await first
    assert len(await eng.predict(["third"], Q)) == 1


async def test_interactive_call_jumps_ahead_of_batch_work(settings):
    router = TrackingRouter(delay_s=0.25)
    eng = await started(settings, router)
    batch_states = [f"batch {i}" for i in range(4)]
    batch = asyncio.create_task(asyncio.to_thread(eng.predict_blocking, batch_states, Q))
    await wait_until(lambda: len(router.log) >= 1)          # first batch state in flight
    t0 = time.monotonic()
    await eng.predict(["interactive"], Q)
    waited = time.monotonic() - t0
    batch_out = await batch
    assert router.log.index("interactive") == 1              # right after the in-flight state
    assert len(batch_out) == 4 and router.log[2:] == batch_states[1:]
    assert waited < 0.25 * 2 + 0.3                           # at most one batch pass + its own


async def test_timeout_while_queued_never_runs(settings):
    router = TrackingRouter(delay_s=0.5)
    eng = await started(settings, router)
    first = asyncio.create_task(eng.predict(["slow"], Q))
    await wait_until(lambda: router.log == ["slow"])
    with pytest.raises(TimeoutError, match="did not answer within"):
        await eng.predict(["late"], Q, timeout=0.1)
    assert eng.status().waiting == 0
    await first
    await asyncio.sleep(0.1)
    assert router.log == ["slow"]


async def test_timeout_of_running_pass_finishes_in_background(settings):
    router = TrackingRouter(delay_s=0.4)
    eng = await started(settings, router, request_timeout_s=0.1)   # default timeout from Settings
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        await eng.predict(["slow"], Q)
    assert time.monotonic() - t0 < 0.35
    assert eng.status().busy                                  # the pass is still running
    await wait_until(lambda: not eng.status().busy)
    assert router.log == ["slow"]
    out = await eng.predict(["next"], Q, timeout=5)           # engine is usable again
    assert out[0]["routing"]["model"] == "english"


async def test_multi_state_call_stops_after_timeout(settings):
    router = TrackingRouter(delay_s=0.2)
    eng = await started(settings, router)
    with pytest.raises(TimeoutError):
        await eng.predict(["s1", "s2", "s3", "s4"], Q, timeout=0.3)
    await wait_until(lambda: not eng.status().busy)
    await asyncio.sleep(0.3)
    assert router.log == ["s1", "s2"]


async def test_cancelled_caller_does_not_block_batch_work(settings):
    router = TrackingRouter(delay_s=0.3)
    eng = await started(settings, router)
    first = asyncio.create_task(eng.predict(["first"], Q))
    await wait_until(lambda: eng.status().busy)
    queued = asyncio.create_task(eng.predict(["queued"], Q))
    await wait_until(lambda: eng.status().waiting == 1)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert eng.status().waiting == 0
    await first
    out = await asyncio.wait_for(asyncio.to_thread(eng.predict_blocking, ["batch"], Q), timeout=5)
    assert len(out) == 1 and "queued" not in router.log


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "questions, fragment",
    [
        ({}, "no questions"),
        ({"q": "just text"}, "definition must be a dict"),
        ({"q": {"type": "rank", "instructions": "x"}}, "unknown type"),
        ({"q": {"type": "noul"}}, "no 'instructions'"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": []}}, "at least one criterion"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": "a,b"}}, "takes 'criteria' as a dict"),
        ({"q": {"type": "score", "instructions": "x", "criteria": {"0": "low"}}}, "as a list of level"),
        ({"q": {"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}}, "noul question takes"),
        ({"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b", "a"]}}, "appears twice"),
        ({"q": {"type": "noul", "instructions": "   "}}, "non-empty text"),
    ],
)
async def test_invalid_questions_fail_fast(settings, questions, fragment):
    router = TrackingRouter()
    eng = await started(settings, router)
    with pytest.raises(InvalidQuestions, match=fragment):
        eng.validate(questions)
    with pytest.raises(InvalidQuestions, match=fragment):
        await eng.predict(["x"], questions)
    with pytest.raises(InvalidQuestions, match=fragment):
        eng.predict_blocking(["x"], questions)
    assert router.calls == 0


async def test_laya_value_errors_become_invalid_questions(settings):
    class Raising(FakeRouter):
        message = "question 'topic' options exceed head_max_len=192"

        def predict(self, state, questions, **kw):
            raise ValueError(self.message)

    router = Raising()
    eng = await started(settings, router)
    with pytest.raises(InvalidQuestions, match="options exceed head_max_len=192; remove labels"):
        await eng.predict(["x"], Q)
    router.message = "something unrelated"
    with pytest.raises(ValueError, match="unrelated") as ei:
        await eng.predict(["x"], Q)
    assert not isinstance(ei.value, InvalidQuestions)


async def test_validate_clean_questions_has_no_warnings(settings):
    eng = await started(settings)
    assert eng.validate(Q) == []


def test_validate_before_start_is_structural_only(settings):
    eng = InferenceEngine(settings, FakeRouter())
    warnings = eng.validate(Q)
    assert warnings == ["token budget not checked: checkpoint 'english' is not loaded, so only the structure was validated"]


def _choice(criteria, instructions="Which one?"):
    return {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}


@pytest.mark.parametrize(
    "questions, fragment",
    [
        (_choice({f"label{i}": f"category number {i} of many" for i in range(14)}), "has 14 labels"),
        (_choice({"bug": "a defect in the software product", "defect": "a defect in the software products"}),
         "near-identical descriptions"),
        (_choice({"bug": "Broken feature.", "defect": "broken feature"}), "identical descriptions"),
        (_choice({"bug": "a software defect", "other": None}), "'other' have no description while others do"),
        (_choice(["bug", "feature", "question"]), "labels have no descriptions"),
        (_choice({"Bug": "a crash", "bug": "a defect"}), "differ only in case"),
        (_choice({"only": "the one option"}), "single label"),
        (_choice({"a": "first", "b": "second"}, instructions="x" * 301), "instructions are 301 characters"),
        ({"q": {"type": "noul", "instructions": "Is it spam?", "criteria": {"yes": "spam", "true": "spam"}}},
         "'yes' are ignored"),
        ({"q": {"type": "score", "instructions": "How bad?", "criteria": ["fine", "", "terrible"]}},
         "score levels 1 have no description"),
        ({"q": {"type": "score", "instructions": "How bad?", "criteria": ["not bad at all", "not bad at all!"]}},
         "level 0 and level 1 have identical"),
    ],
)
async def test_validate_warnings(settings, questions, fragment):
    eng = await started(settings)
    warnings = eng.validate(questions)
    assert any(fragment in w for w in warnings), warnings


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


async def test_validate_token_budget_with_loaded_tokenizer(settings):
    """FakeRouter's agent: whitespace tokenizer, max_len 512, head_max_len 192."""
    eng = await started(settings)
    near = _choice({f"l{i}": _words(30, f"w{i}_") for i in range(5)})          # 5 x 32 = 160 of 176 tokens
    assert any("of the 176-token option budget" in w for w in eng.validate(near))
    cut = _choice({f"l{i}": _words(30, f"w{i}_") for i in range(8)})           # 256 > 176: cut to 22 each
    assert any("cut to 22 tokens" in w for w in eng.validate(cut))
    long_desc = _choice({"a": _words(60, "x"), "b": "short"})
    assert any("longer than 48 tokens" in w for w in eng.validate(long_desc))
    with pytest.raises(InvalidQuestions, match="options exceed head_max_len=192"):
        eng.validate(_choice([f"l{i}" for i in range(300)]))                  # markers past max_len


async def test_validate_warns_about_question_count(settings):
    eng = await started(settings, max_questions=2)
    qs = {f"q{i}": {"type": "noul", "instructions": f"Is statement {i} true?"} for i in range(3)}
    assert any("3 questions exceed the per-call limit of 2" in w for w in eng.validate(qs))


# --------------------------------------------------------------------------- budget
def test_check_sync_budget(settings):
    eng = InferenceEngine(settings.model_copy(update={"sync_row_budget": 40}), FakeRouter())
    eng.check_sync_budget(10, 4, ["short"] * 10)                               # exactly 40 rows
    with pytest.raises(BudgetExceeded, match="44 rows exceeds the synchronous budget of 40") as ei:
        eng.check_sync_budget(11, 4)
    assert "laya_classify_batch" in str(ei.value)
    with pytest.raises(BudgetExceeded, match="21 questions exceed the limit of 20"):
        eng.check_sync_budget(1, 21)
    with pytest.raises(BudgetExceeded, match=r"item 1 \(20,001 chars\)"):
        eng.check_sync_budget(2, 1, ["ok", "x" * 20_001])
    big = {"text": "y" * 19_995}                                               # JSON adds 12 chars
    with pytest.raises(BudgetExceeded, match="item 0"):
        eng.check_sync_budget(1, 1, [big])


def test_budget_helpers():
    assert budget.rows_for(3, Q) == 6
    assert budget.rows_for(3, 4) == 12
    assert budget.rows_for(0, Q) == 0
    assert budget.state_chars({"a": "é"}) == len('{"a": "é"}')
    assert budget.oversized_states(["abc", "abcdef", {"k": "v"}], 5) == [(1, 6), (2, 10)]


# --------------------------------------------------------------------------- estimates / status
async def test_estimate_seconds_uses_rolling_ms_per_row(settings):
    eng = await started(settings, TrackingRouter(delay_s=0.05))
    assert eng.estimate_seconds(10) == pytest.approx(10 * DEFAULT_MS_PER_ROW / 1000)
    await eng.predict(["a", "b", "c"], Q)                   # 3 passes x 2 rows, ~25 ms per row
    ms = eng.status().ms_per_row
    assert ms is not None and 20 <= ms <= 250
    assert eng.estimate_seconds(4) == pytest.approx(4 * eng.ms_per_row / 1000)


async def test_pass_that_loads_its_model_is_not_timed(settings):
    router = TrackingRouter(delay_s=0.02)
    eng = await started(settings, router, preload="")
    assert router.loaded == []
    await eng.predict(["a"], Q)                            # loads english inside the pass
    assert eng.ms_per_row is None
    await eng.predict(["b"], Q)
    assert eng.ms_per_row is not None


async def test_status(settings):
    router = TrackingRouter(delay_s=0.3)
    eng = await started(settings, router)
    st = eng.status()
    assert st.loaded == ["english"] and st.available == list(KNOWN_MODELS)
    assert st.laya_version == "0.3.7" and st.device == "cpu"
    assert st.threads == torch.get_num_threads()
    assert not st.busy and st.waiting == 0 and st.ms_per_row is None
    assert st.rss_gb is not None and st.rss_gb > 0.01 and st.uptime_s >= 0
    task = asyncio.create_task(eng.predict(["x"], Q))
    await wait_until(lambda: eng.status().busy)
    await task
    assert eng.status().ms_per_row is not None


# --------------------------------------------------------------------------- models
async def test_predict_all_models_restores_resident_set(settings):
    router = TrackingRouter()
    eng = await started(settings, router)
    out = await eng.predict_all_models("please refund me", Q)
    assert list(out) == list(KNOWN_MODELS)
    assert {m: r["routing"]["model"] for m, r in out.items()} == {m: m for m in KNOWN_MODELS}
    assert all(r["answers"]["intent"]["choice"] == "refund" for r in out.values())
    assert router.loaded == ["english"]


async def test_predict_all_models_reports_a_checkpoint_that_fails_to_load(settings):
    class Offline(FakeRouter):
        def predict(self, state, questions, model=None, **kw):
            if model == "typed-decisions":
                raise OSError("not in the local cache and HF_HUB_OFFLINE=1")
            return super().predict(state, questions, model=model, **kw)

    eng = await started(settings, Offline())
    out = await eng.predict_all_models("hello", Q)
    assert out["typed-decisions"]["answers"] == {} and "HF_HUB_OFFLINE" in out["typed-decisions"]["error"]
    assert out["english"]["answers"] and out["multilingual"]["answers"]


async def test_load_unload_and_tokenizer(settings):
    router = FakeRouter()
    eng = await started(settings, router)
    assert await eng.load(["multi"]) == ["english", "multilingual"]
    assert await eng.unload(["multilingual"]) == ["english"]
    with pytest.raises(ValueError, match="unknown model"):
        await eng.load(["bogus"])
    assert eng.tokenizer().encode("two words") and eng.tokenizer("en") is not None
    assert await eng.unload() == []


# --------------------------------------------------------------------------- quantization hook
class _TinyAgent:
    def __init__(self, device: Any) -> None:
        self.model = torch.nn.Module()
        self.model.encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        self.model.head = torch.nn.Linear(2, 2)
        self.device = device
        self.tok = None
        self.cfg = {"max_len": 512, "head_max_len": 192}


class _TinyRouter(FakeRouter):
    def __init__(self, device: Any) -> None:
        super().__init__()
        self.device = device
        self.agents: dict[str, _TinyAgent] = {}

    def load(self, name):
        super().load(name)
        return self.agents.setdefault(name, _TinyAgent(self.device))


def _is_quantized(module: torch.nn.Module) -> bool:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    return isinstance(module, DynamicQuantizedLinear)


async def test_quantize_hook_quantizes_encoder_linears_once(settings):
    router = _TinyRouter(torch.device("cpu"))
    eng = await started(settings, router, quantize=True)
    agent = router.agents["english"]
    assert _is_quantized(agent.model.encoder[0]) and _is_quantized(agent.model.encoder[2])
    assert not _is_quantized(agent.model.head)                 # decision head stays fp32
    assert agent.model.encoder(torch.randn(3, 4)).shape == (3, 2)
    await eng.predict(["x"], Q)                                # load() again: no double quantization
    await eng.load(["multilingual"])
    assert _is_quantized(router.agents["multilingual"].model.encoder[0])


async def test_quantize_skips_non_cpu_and_disabled(settings):
    gpu = _TinyRouter(SimpleNamespace(type="cuda"))
    await started(settings, gpu, quantize=True)
    assert not _is_quantized(gpu.agents["english"].model.encoder[0])
    off = _TinyRouter(torch.device("cpu"))
    await started(settings, off, quantize=False)
    assert not _is_quantized(off.agents["english"].model.encoder[0])
    await started(settings, FakeRouter(), quantize=True)       # agents without a torch model: no-op
