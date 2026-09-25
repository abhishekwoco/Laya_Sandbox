"""Deterministic stand-ins for laya.Router and InferenceEngine.

FakeRouter mimics laya.Router.predict output exactly (shapes from laya/agent.py:399-424), so
it can be injected into the real InferenceEngine. Answers are keyword-driven so tests can
assert on them: a choice label (or its description) appearing in the state text gets most of
the probability; a noul is true when the instructions' last word appears in the state; a score
picks the highest level whose description word appears.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

import numpy as np

from laya_mcp.models import EngineStatus


def _text(state: Any) -> str:
    return (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)).lower()


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _jitter(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return (h[0] / 255.0) * 0.5


def fake_answer(state: Any, qid: str, q: dict[str, Any]) -> dict[str, Any]:
    text = _text(state)
    t = q["type"]
    crit = q.get("criteria")
    if t == "choice":
        labels = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        z = np.array([_jitter(text, qid, lab) for lab in labels])
        for i, lab in enumerate(labels):
            desc = (crit.get(lab) if isinstance(crit, dict) else None) or ""
            words = [lab.lower()] + [w for w in desc.lower().replace(",", " ").split() if len(w) > 3]
            if any(w in text for w in words):
                z[i] += 4.0
        p = _softmax(z)
        conf = float(p.max())
        return {
            "type": "choice",
            "choice": labels[int(p.argmax())],
            "probabilities": {lab: round(float(v), 4) for lab, v in zip(labels, p)},
            "confidence": round(conf, 4),
            "action": {"act_probability": 0.5},
        }
    if t == "score":
        levels = list(crit)
        z = np.array([_jitter(text, qid, str(i)) for i in range(len(levels))])
        for i, desc in enumerate(levels):
            if any(w in text for w in desc.lower().split() if len(w) > 3):
                z[i] += 3.0
        p = _softmax(z)
        return {
            "type": "score",
            "score": round(float((np.arange(len(levels)) * p).sum()), 4),
            "legend": {str(i): c for i, c in enumerate(levels)},
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": round(float(p.max()), 4),
            "action": {"act_probability": 0.5},
        }
    # noul
    key = q["instructions"].lower().rstrip("?. ").split()[-1]
    p_true = 0.9 if key in text else 0.1 + _jitter(text, qid) * 0.4
    return {
        "type": "noul",
        "noul": round(p_true, 4),
        "confidence": round(max(p_true, 1 - p_true), 4),
        "action": {"act_probability": 0.5},
    }


class _FakeTok:
    """Whitespace 'tokenizer' with the subset of the HF API that chunking/budget code uses."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [abs(hash(w)) % 30000 + 1 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join("tok" for _ in ids)

    def __call__(self, text: str, **kw: Any) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


class _FakeAgent:
    def __init__(self) -> None:
        self.tok = _FakeTok()
        self.cfg = {"max_len": 512, "head_max_len": 192}


class FakeRouter:
    """Implements the slice of laya.Router that InferenceEngine uses."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self._loaded: list[str] = []
        self.calls = 0

    @property
    def loaded(self) -> list[str]:
        return list(self._loaded)

    def load(self, name: str) -> _FakeAgent:
        if name not in self._loaded:
            self._loaded.append(name)
        return _FakeAgent()

    def preload(self, names: list[str] | None = None) -> None:
        for n in names or ["english", "multilingual", "typed-decisions"]:
            self.load(n)

    def unload(self, name: str | None = None) -> None:
        self._loaded = [] if name is None else [n for n in self._loaded if n != name]

    def predict(self, state: Any, questions: dict[str, Any], model: str | None = None,
                task: str | None = None, lang: str | None = None, lang_guess: Any = None) -> dict[str, Any]:
        for qid, q in questions.items():
            if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
                raise ValueError(f"question {qid!r}: unknown type {q.get('type') if isinstance(q, dict) else q!r}")
            if "instructions" not in q:
                raise ValueError(f"question {qid!r}: no 'instructions'; add the text the model should answer")
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        chosen = model or ("multilingual" if any(ord(c) > 0x900 for c in _text(state)) else "english")
        self.load(chosen)
        answers = {qid: fake_answer(state, qid, q) for qid, q in questions.items()}
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 10 * len(questions), "output_tokens": 0},
            "routing": {"model": chosen, "reason": "fake router"},
        }


class FakeEngine:
    """Implements the InferenceEngine contract (engine/runtime.py) on top of FakeRouter,
    without the real locking/priority logic. For tool tests while the real engine is built;
    once the real engine exists prefer InferenceEngine(settings, router=FakeRouter())."""

    def __init__(self, settings: Any, router: FakeRouter | None = None) -> None:
        self.settings = settings
        self.router = router or FakeRouter()
        self._t0 = time.time()

    async def start(self) -> None:
        self.router.preload(self.settings.preload_models)

    async def stop(self) -> None:
        pass

    def validate(self, questions: dict[str, dict[str, Any]], model: str = "english") -> list[str]:
        from laya_mcp.engine.runtime import InvalidQuestions

        for qid, q in questions.items():
            if q.get("type") == "choice" and not q.get("criteria"):
                raise InvalidQuestions(f"question {qid!r}: a choice question needs at least one criterion")
            if q.get("type") == "score" and not isinstance(q.get("criteria"), list):
                raise InvalidQuestions(f"question {qid!r}: a score question takes 'criteria' as a list")
        return []

    def check_sync_budget(self, n_items: int, n_questions: int, states: list[Any] | None = None) -> None:
        from laya_mcp.engine.runtime import BudgetExceeded

        if n_items * n_questions > self.settings.sync_row_budget:
            raise BudgetExceeded(
                f"{n_items} items x {n_questions} questions = {n_items * n_questions} rows exceeds the "
                f"synchronous budget of {self.settings.sync_row_budget}; use laya_classify_batch"
            )

    async def predict(self, states, questions, *, model=None, lang=None, timeout=None):
        return [self.router.predict(s, questions, model=model, lang=lang) for s in states]

    def predict_blocking(self, states, questions, *, model=None, lang=None):
        return [self.router.predict(s, questions, model=model, lang=lang) for s in states]

    async def predict_all_models(self, state, questions):
        return {m: self.router.predict(state, questions, model=m) for m in ("english", "multilingual", "typed-decisions")}

    async def load(self, models):
        for m in models:
            self.router.load(m)
        return self.router.loaded

    async def unload(self, models=None):
        for m in models or [None]:
            self.router.unload(m)
        return self.router.loaded

    def tokenizer(self, model: str = "english"):
        return _FakeTok()

    def estimate_seconds(self, rows: int) -> float:
        return rows * 0.001

    def status(self) -> EngineStatus:
        return EngineStatus(
            laya_version="0.3.7", device="cpu", threads=self.settings.threads, loaded=self.router.loaded,
            available=["english", "multilingual", "typed-decisions"], busy=False, waiting=0,
            rss_gb=None, uptime_s=time.time() - self._t0, ms_per_row=1.0,
        )
