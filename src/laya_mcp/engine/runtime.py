"""InferenceEngine: the only object that touches laya.Router.

CONTRACT (implemented by the engine workstream; tools and jobs depend on exactly this):

- One shared `laya.Router` (device, max_loaded from Settings; torch threads capped to
  Settings.threads via torch.set_num_threads before any model loads).
- Exactly ONE forward pass runs at a time (a threading lock). On this 8-core CPU host
  concurrent passes only fight for cores.
- Two priorities:
    * interactive (async `predict`): used by tools. If more than
      Settings.max_waiting_requests callers are already waiting -> raise EngineBusy.
    * batch (sync `predict_blocking`): used by job worker threads. Before taking the
      lock for each state it yields while any interactive caller is waiting, so an
      interactive call waits for at most one in-flight row group.
- Every state is one Router.predict call (Laya already batches all questions of a state
  into one forward pass).
- Raw result per state = exactly what `Router.predict` returns:
      {"model": ..., "answers": {qid: {...}}, "usage": {...}, "routing": {"model": "english", "reason": "..."}}
  Answers are NOT calibrated here; see engine/answers.py.
- Tracks a rolling average ms per row (row = one question on one state) for ETAs and status.

IMPLEMENTATION NOTES

- The "lock" is a slot guarded by a `threading.Condition`. Interactive callers are counted as
  waiting from admission until their worker thread holds the slot; a batch caller only takes the
  slot when it is free AND no interactive caller is waiting. Interactive callers queue FIFO on an
  asyncio.Lock, so only one of them occupies a worker thread at a time; an interactive call holds
  the slot for all of its states (bounded by Settings.sync_row_budget).
- Admission: at most `max_waiting_requests` interactive calls wait at once; the next one gets
  EngineBusy (retryable, with an ETA). A call arriving at an idle engine is always admitted.
- Timeouts (`request_timeout_s`, or `timeout=`) cover queueing plus inference. A caller that
  times out while still queued never runs. A forward pass that is already running cannot be
  interrupted: it finishes in its worker thread (holding the slot until then) and its result is
  discarded; a multi-state call stops after the current state.
- Model loads are not serialised with inference (laya.Router has its own lifecycle lock), except
  lazy loads triggered by a prediction, which happen inside that prediction's slot.
- `Settings.quantize`: every Agent the Router loads gets int8 dynamic quantization of its
  encoder's nn.Linear layers (CPU only). The decision head stays fp32 because
  nn.TransformerEncoderLayer's inference fast path reads Linear weights directly and fails on
  quantized modules. See docs/perf.md for when this is worth enabling.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from ..config import Settings
from ..models import EngineStatus, State
from . import budget

log = logging.getLogger("laya_mcp.engine")

T = TypeVar("T")

DEFAULT_MS_PER_ROW = 1000.0     # measured (docs/perf.md): english, 6 threads, ~0.5 s short to ~1.6 s long state per question row
EMA_ALPHA = 0.2
# Per-channel int8 weights: the less inaccurate of the two variants measured, and still far from
# fp32 (74% argmax agreement on the probe set), which is why Settings.quantize defaults to off.
QUANTIZE_PER_CHANNEL = True
WARMUP_STATE = "Warm-up request: please confirm the classifier is ready."
WARMUP_QUESTIONS: dict[str, dict[str, Any]] = {
    "ready": {"type": "noul", "instructions": "Is this a warm-up request?"},
}


class EngineBusy(RuntimeError):
    """Too many interactive requests waiting. Retryable."""

    def __init__(self, retry_after_s: float, waiting: int):
        super().__init__(
            f"Laya is busy ({waiting} requests waiting); retry in ~{retry_after_s:.0f}s "
            "or use laya_classify_batch for large workloads."
        )
        self.retry_after_s = retry_after_s
        self.waiting = waiting


class BudgetExceeded(ValueError):
    """Synchronous request is larger than the configured row budget or input caps."""


class InvalidQuestions(ValueError):
    """Question set failed validation. Message names the question and what to fix."""


KNOWN_MODELS = ("english", "multilingual", "typed-decisions")


# --------------------------------------------------------------------------- helpers
def quantize_dynamic_int8(model: Any, *, per_channel: bool = QUANTIZE_PER_CHANNEL) -> Any:
    """In-place int8 dynamic quantization of the nn.Linear layers of `model.encoder` (or of
    `model` itself when it has no encoder). CPU only. Returns the model."""
    import torch
    from torch.ao.quantization import default_dynamic_qconfig, per_channel_dynamic_qconfig, quantize_dynamic

    target = getattr(model, "encoder", None)
    if not isinstance(target, torch.nn.Module):
        target = model
    qconfig = per_channel_dynamic_qconfig if per_channel else default_dynamic_qconfig
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization deprecation notices
        quantize_dynamic(target, {torch.nn.Linear: qconfig}, inplace=True)
    return model


def _normalise_model(model: str | None) -> str | None:
    """Canonical checkpoint name (laya aliases accepted); ValueError names the valid choices."""
    if model is None:
        return None
    try:
        from laya.router import normalise_name
    except Exception:  # pragma: no cover - laya is a hard dependency
        key = str(model).strip().lower()
        if key not in KNOWN_MODELS:
            raise ValueError("unknown model %r; choose one of %s" % (model, list(KNOWN_MODELS))) from None
        return key
    return normalise_name(model)


def _as_invalid(e: ValueError) -> Exception:
    """laya raises ValueError('question ...') for questions it cannot answer."""
    msg = str(e)
    if not msg.startswith("question "):
        return e
    if "exceed head_max_len" in msg and "split" not in msg:
        msg += "; remove labels, shorten their descriptions or split the question"
    return InvalidQuestions(msg)


def _laya_version() -> str:
    try:
        from importlib.metadata import version

        return version("laya")
    except Exception:  # pragma: no cover
        return "unknown"


def _rss_gb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**30, 3)
    except Exception:  # pragma: no cover
        return None


def _offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


class _Ticket:
    """One admitted interactive request."""

    __slots__ = ("rows", "waiting", "cancelled", "finished")

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.waiting = True
        self.cancelled = False
        self.finished = False


class _Cancelled(Exception):
    """The caller gave up (timeout/cancellation) before its turn came."""


# --------------------------------------------------------------------------- engine
class InferenceEngine:
    def __init__(self, settings: Settings, router: Any | None = None) -> None:
        """`router` may be injected (tests pass a fake with the laya.Router interface:
        predict(state, questions, model=None, lang=None) -> dict, preload(names), unload(name),
        load(name) -> Agent-like with `.tok`, `.cfg`, and `.loaded` property)."""
        self.settings = settings
        self._router: Any | None = router
        self._owns_router = router is None
        self._cv = threading.Condition()
        self._running = False          # a forward pass holds the slot
        self._closed = False
        self._started = False
        self._waiting = 0              # interactive callers admitted but not yet holding the slot
        self._queued_rows = 0          # rows of admitted interactive calls not yet finished
        self._batch_waiting = 0
        self._ms_per_row: float | None = None
        self._passes = 0
        self._t0 = time.monotonic()
        self._async_lock: asyncio.Lock | None = None
        self._async_lock_loop: asyncio.AbstractEventLoop | None = None
        self._hook_lock = threading.Lock()
        self._device: str | None = settings.device
        self._available: list[str] | None = None

    # lifecycle ---------------------------------------------------------------
    async def start(self) -> None:
        """Build the Router if not injected, apply thread cap, preload Settings.preload_models
        and run one warm-up prediction (Settings.warmup). Runs blocking work in a thread."""
        await asyncio.to_thread(self._start_blocking)

    def _start_blocking(self) -> None:
        s = self.settings
        self._apply_thread_cap()
        budget._laya_agent_cls()  # import laya.agent (and torch) here, not on the event loop later
        if self._router is None:
            from laya import Router

            self._router = Router(device=s.device, max_loaded=s.max_loaded)
        self._install_load_hook(self._router)
        names = [_normalise_model(m) for m in s.preload_models]
        if names:
            t0 = time.perf_counter()
            self._router.preload(names)
            log.info("preloaded %s in %.1fs", names, time.perf_counter() - t0)
        if s.warmup and names:
            self._acquire_batch()
            try:
                for m in names:
                    t0 = time.perf_counter()
                    self._forward(WARMUP_STATE, WARMUP_QUESTIONS, m, None, record=False)
                    log.info("warm-up on %s took %.0f ms", m, (time.perf_counter() - t0) * 1000)
            finally:
                self._release()
        self._available = self._detect_available()
        self._started = True

    async def stop(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        await asyncio.to_thread(self._drain, 30.0)
        if self._owns_router and self._router is not None:
            await asyncio.to_thread(self._router.unload)
        self._started = False

    def _drain(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("engine stop: a forward pass is still running after %.0fs", timeout)
                    return
                self._cv.wait(remaining)

    def _require_started(self) -> Any:
        if self._closed:
            raise RuntimeError("the Laya engine is stopped")
        if self._router is None or not self._started:
            raise RuntimeError("the Laya engine is not started")
        return self._router

    # validation ---------------------------------------------------------------
    def validate(self, questions: dict[str, dict[str, Any]], model: str = "english") -> list[str]:
        """Validate a Laya-format question dict without inference.
        Raises InvalidQuestions (message names the question and the fix; reuse laya's
        Agent._check_question wording). Returns a list of warnings (option budget close to
        head_max_len, > 12 choice labels, duplicate/near-duplicate label descriptions, missing
        descriptions, overly long instructions). Uses the loaded tokenizer when available,
        otherwise only structural checks."""
        name = _normalise_model(model) or "english"
        tok = cfg = None
        router = self._router
        if router is not None and name in router.loaded:
            agent = router.load(name)
            tok, cfg = getattr(agent, "tok", None), getattr(agent, "cfg", None)
        try:
            try:
                warnings_ = budget.validate_questions(
                    questions, tok=tok, cfg=cfg, max_questions=self.settings.max_questions
                )
            except ValueError:
                raise
            except Exception as e:  # tokenizer oddity: keep the structural result
                log.warning("token budget check failed for %s: %r", name, e)
                warnings_ = budget.validate_questions(questions, max_questions=self.settings.max_questions)
                tok = None
        except ValueError as e:
            raise InvalidQuestions(str(e)) from e
        if tok is None or cfg is None:
            warnings_.append(
                "token budget not checked: checkpoint %r is not loaded, so only the structure was validated" % name
            )
        return warnings_

    def _check_questions(self, questions: Any) -> None:
        try:
            budget.check_structure(questions)
        except ValueError as e:
            raise InvalidQuestions(str(e)) from e

    def check_sync_budget(self, n_items: int, n_questions: int, states: list[State] | None = None) -> None:
        """Raise BudgetExceeded when n_items * n_questions > Settings.sync_row_budget,
        n_questions > Settings.max_questions, or any serialized state > Settings.max_state_chars.
        The message must tell the agent what to do (e.g. 'use laya_classify_batch')."""
        s = self.settings
        if n_questions > s.max_questions:
            raise BudgetExceeded(
                f"{n_questions} questions exceed the limit of {s.max_questions} per call. Keep only the "
                f"questions you need, or split them across calls (each question costs one row per item)."
            )
        rows = budget.rows_for(n_items, n_questions)
        if rows > s.sync_row_budget:
            raise BudgetExceeded(
                f"{n_items} items x {n_questions} questions = {rows} rows exceeds the synchronous budget of "
                f"{s.sync_row_budget} rows (~{self.estimate_seconds(rows):.0f}s of inference). Use "
                f"laya_classify_batch for this workload, or send fewer items or questions per call."
            )
        if states:
            big = budget.oversized_states(states, s.max_state_chars)
            if big:
                where = ", ".join(f"item {i} ({n:,} chars)" for i, n in big[:5])
                raise BudgetExceeded(
                    f"{where} exceed the limit of {s.max_state_chars:,} characters per state. Send concise, "
                    f"pre-extracted content: the english checkpoint reads only about 500 tokens (~2,000 "
                    f"characters) of each state, so split long text into separate items."
                )

    # inference ----------------------------------------------------------------
    async def predict(
        self,
        states: list[State],
        questions: dict[str, dict[str, Any]],
        *,
        model: str | None = None,
        lang: str | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """Interactive priority. Returns one raw laya result per state, in order.
        Raises EngineBusy, TimeoutError, InvalidQuestions (laya ValueError on bad questions)."""
        self._require_started()
        self._check_questions(questions)
        name = _normalise_model(model)
        states = list(states)
        if not states:
            return []

        def work(ticket: _Ticket) -> list[dict[str, Any]]:
            out = []
            for state in states:
                if ticket.cancelled:
                    raise _Cancelled()
                out.append(self._forward(state, questions, name, lang))
            return out

        return await self._interactive(budget.rows_for(len(states), questions), timeout, work)

    def predict_blocking(
        self,
        states: list[State],
        questions: dict[str, dict[str, Any]],
        *,
        model: str | None = None,
        lang: str | None = None,
    ) -> list[dict[str, Any]]:
        """Batch priority, for job worker threads. Same output as predict().
        Must not be called on the event loop thread (it blocks)."""
        self._require_started()
        self._check_questions(questions)
        name = _normalise_model(model)
        out = []
        for state in states:
            self._acquire_batch()
            try:
                out.append(self._forward(state, questions, name, lang))
            finally:
                self._release()
        return out

    async def predict_all_models(self, state: State, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Run one state on every checkpoint in KNOWN_MODELS (loading lazily). {model: raw result}.

        Resident checkpoints run first. A checkpoint loaded only for this call is unloaded right
        after its pass (the Router's max_loaded cap still applies), and preloaded checkpoints that
        were evicted are reloaded, so the server ends in the same warm state. A checkpoint that
        cannot be loaded (e.g. not downloaded while HF_HUB_OFFLINE=1) yields a result with empty
        `answers` and an `error` string instead of failing the whole comparison."""
        self._require_started()
        self._check_questions(questions)
        rows = budget.rows_for(len(KNOWN_MODELS), questions)
        return await self._interactive(rows, None, lambda ticket: self._compare(ticket, state, questions))

    def _compare(self, ticket: _Ticket, state: State, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        router = self._router
        resident = set(router.loaded)
        out: dict[str, dict[str, Any]] = {}
        for m in sorted(KNOWN_MODELS, key=lambda m: m not in resident):
            if ticket.cancelled:
                raise _Cancelled()
            transient = m not in resident
            try:
                out[m] = self._forward(state, questions, m, None)
            except InvalidQuestions:
                raise
            except Exception as e:
                log.warning("compare: checkpoint %s failed: %r", m, e)
                out[m] = {
                    "model": "laya-rl-agent",
                    "answers": {},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "routing": {"model": m, "reason": "explicit model=%r" % m},
                    "error": f"{type(e).__name__}: {e}",
                }
            finally:
                if transient and m in router.loaded:
                    router.unload(m)
        for m in self.settings.preload_models:
            name = _normalise_model(m)
            if name in resident and name not in router.loaded:
                router.load(name)
        return {m: out[m] for m in KNOWN_MODELS if m in out}

    # models / status ------------------------------------------------------------
    async def load(self, models: list[str]) -> list[str]:
        """Load checkpoints; returns the loaded list afterwards."""
        router = self._require_started()
        names = [_normalise_model(m) for m in models]

        def work() -> list[str]:
            for n in names:
                router.load(n)
            self._available = self._detect_available()
            return list(router.loaded)

        return await asyncio.to_thread(work)

    async def unload(self, models: list[str] | None = None) -> list[str]:
        router = self._require_started()
        names = None if models is None else [_normalise_model(m) for m in models]

        def work() -> list[str]:
            if names is None:
                router.unload()
            else:
                for n in names:
                    router.unload(n)
            return list(router.loaded)

        return await asyncio.to_thread(work)

    def tokenizer(self, model: str = "english") -> Any:
        """Tokenizer of a checkpoint (loads it if needed). Used for chunking and budget checks."""
        router = self._require_started()
        return router.load(_normalise_model(model) or "english").tok

    def estimate_seconds(self, rows: int) -> float:
        """rows * rolling ms_per_row (fallback DEFAULT_MS_PER_ROW before any measurement)."""
        ms = self._ms_per_row if self._ms_per_row is not None else DEFAULT_MS_PER_ROW
        return max(0, rows) * ms / 1000.0

    @property
    def ms_per_row(self) -> float | None:
        return self._ms_per_row

    @property
    def batch_waiting(self) -> int:
        """Job worker threads currently waiting for the slot."""
        return self._batch_waiting

    def status(self) -> EngineStatus:
        router = self._router
        with self._cv:
            busy, waiting = self._running, self._waiting
        return EngineStatus(
            laya_version=_laya_version(),
            device=self._device or self._default_device(),
            threads=self._threads(),
            loaded=list(router.loaded) if router is not None else [],
            available=list(self._available) if self._available is not None else list(KNOWN_MODELS),
            busy=busy,
            waiting=waiting,
            rss_gb=_rss_gb(),
            uptime_s=round(time.monotonic() - self._t0, 1),
            ms_per_row=round(self._ms_per_row, 1) if self._ms_per_row is not None else None,
        )

    # internals: scheduling ----------------------------------------------------------
    def _interactive_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    def _admit(self, rows: int) -> _Ticket:
        with self._cv:
            if self._closed:
                raise RuntimeError("the Laya engine is stopped")
            limit = max(0, int(self.settings.max_waiting_requests))
            if (self._running or self._waiting) and self._waiting >= limit:
                eta = max(1.0, self.estimate_seconds(self._queued_rows + rows))
                raise EngineBusy(retry_after_s=eta, waiting=self._waiting)
            self._waiting += 1
            self._queued_rows += rows
            return _Ticket(rows)

    def _leave_waiting(self, ticket: _Ticket) -> None:
        """Caller must hold self._cv."""
        if ticket.waiting:
            ticket.waiting = False
            self._waiting -= 1
            self._cv.notify_all()

    def _finish(self, ticket: _Ticket, cancelled: bool) -> None:
        with self._cv:
            if cancelled:
                ticket.cancelled = True
            self._leave_waiting(ticket)
            if not ticket.finished:
                ticket.finished = True
                self._queued_rows -= ticket.rows
            self._cv.notify_all()

    def _acquire_interactive(self, ticket: _Ticket) -> None:
        with self._cv:
            while self._running and not self._closed and not ticket.cancelled:
                self._cv.wait()
            self._leave_waiting(ticket)
            if self._closed:
                raise RuntimeError("the Laya engine is stopped")
            if ticket.cancelled:
                raise _Cancelled()
            self._running = True

    def _acquire_batch(self) -> None:
        with self._cv:
            self._batch_waiting += 1
            try:
                while (self._running or self._waiting > 0) and not self._closed:
                    self._cv.wait()
                if self._closed:
                    raise RuntimeError("the Laya engine is stopped")
                self._running = True
            finally:
                self._batch_waiting -= 1

    def _release(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()

    def _with_slot(self, ticket: _Ticket, work: Callable[[_Ticket], T]) -> T:
        self._acquire_interactive(ticket)
        try:
            return work(ticket)
        finally:
            self._release()

    async def _interactive(self, rows: int, timeout: float | None, work: Callable[[_Ticket], T]) -> T:
        limit = self.settings.request_timeout_s if timeout is None else timeout
        ticket = self._admit(rows)
        completed = False
        cm = asyncio.timeout(limit if limit and limit > 0 else None)
        try:
            async with cm:
                async with self._interactive_lock():
                    result = await asyncio.to_thread(self._with_slot, ticket, work)
            completed = True
            return result
        except TimeoutError:
            if not cm.expired():
                raise
            raise TimeoutError(
                f"Laya did not answer within {limit:.0f}s ({rows} rows, queueing included). A forward pass "
                f"that had already started finishes in the background and its result is discarded. Retry "
                f"later, send fewer items or questions, or use laya_classify_batch."
            ) from None
        finally:
            self._finish(ticket, cancelled=not completed)

    # internals: inference -------------------------------------------------------------
    def _forward(
        self, state: State, questions: dict[str, dict[str, Any]], model: str | None, lang: str | None,
        *, record: bool = True,
    ) -> dict[str, Any]:
        """One Router.predict call. Caller holds the slot."""
        router = self._router
        self._apply_thread_cap()
        resident = set(router.loaded)
        t0 = time.perf_counter()
        try:
            result = router.predict(state, questions, model=model, lang=lang)
        except ValueError as e:
            raise _as_invalid(e) from e
        ms = (time.perf_counter() - t0) * 1000.0
        self._passes += 1
        chosen = (result.get("routing") or {}).get("model")
        # a pass that had to load its checkpoint first says nothing about inference cost
        if record and questions and chosen in resident:
            self._record(ms / len(questions))
        return result

    def _record(self, ms_per_row: float) -> None:
        prev = self._ms_per_row
        self._ms_per_row = ms_per_row if prev is None else (1 - EMA_ALPHA) * prev + EMA_ALPHA * ms_per_row

    def _apply_thread_cap(self) -> None:
        n = self.settings.threads
        if not n:
            return
        import torch

        if torch.get_num_threads() != n:
            torch.set_num_threads(n)

    def _threads(self) -> int | None:
        if self.settings.threads:
            return self.settings.threads
        try:
            import torch

            return torch.get_num_threads()
        except Exception:  # pragma: no cover
            return None

    def _default_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "cpu"

    # internals: model loading ----------------------------------------------------------
    def _install_load_hook(self, router: Any) -> None:
        """Wrap router.load (used by preload and predict too) to record the device and, when
        Settings.quantize is on, quantize each newly built Agent once."""
        if getattr(router, "_laya_mcp_hooked", False):
            return
        original = router.load

        def load(name: str, _original: Callable[[str], Any] = original) -> Any:
            agent = _original(name)
            self._after_load(agent, name)
            return agent

        router.load = load
        router._laya_mcp_hooked = True

    def _after_load(self, agent: Any, name: str) -> None:
        with self._hook_lock:
            if getattr(agent, "_laya_mcp_seen", False):
                return
            device = getattr(agent, "device", None)
            if device is not None:
                self._device = str(device)
            if self.settings.quantize:
                self._quantize(agent, name, device)
            try:
                agent._laya_mcp_seen = True
            except Exception:  # pragma: no cover - agents that refuse attributes get re-checked
                pass

    def _quantize(self, agent: Any, name: str, device: Any) -> None:
        model = getattr(agent, "model", None)
        try:
            import torch
        except Exception:  # pragma: no cover
            return
        if not isinstance(model, torch.nn.Module):
            return
        if getattr(device, "type", str(device or "cpu")) != "cpu":
            log.info("quantize: %s runs on %s, int8 dynamic quantization is CPU-only; skipped", name, device)
            return
        t0 = time.perf_counter()
        quantize_dynamic_int8(model)
        log.info("quantized %s to int8 (dynamic, encoder Linear layers) in %.1fs", name, time.perf_counter() - t0)

    def _detect_available(self) -> list[str]:
        """Checkpoints that can be served now: all of them when downloads are allowed, otherwise
        the resident ones plus those present in the local Hugging Face cache."""
        router = self._router
        loaded = set(router.loaded) if router is not None else set()
        if not self._owns_router or not _offline():
            return list(KNOWN_MODELS)
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:  # pragma: no cover
            return [m for m in KNOWN_MODELS if m in loaded]
        out = []
        for name in KNOWN_MODELS:
            if name in loaded:
                out.append(name)
                continue
            spec = getattr(router, "models", {}).get(name)
            repo, sub = (tuple(spec) + (None,))[:2] if isinstance(spec, (tuple, list)) else (spec, None)
            if not repo:
                continue
            if os.path.isdir(repo):
                out.append(name)
                continue
            fname = f"{sub}/model.safetensors" if sub else "model.safetensors"
            try:
                if isinstance(try_to_load_from_cache(repo, fname), str):
                    out.append(name)
            except Exception:
                pass
        return out
