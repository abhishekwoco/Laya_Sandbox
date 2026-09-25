"""Turn raw laya answers into normalised `Answer`s, applying calibration and thresholds.

Raw laya answer shapes (laya/agent.py:399-424):
  choice: {"type":"choice","choice": label,"probabilities":{label:p},"confidence":c,...}
  score:  {"type":"score","score": expected,"legend":{"0":desc},"probabilities":{"0":p},...}
  noul:   {"type":"noul","noul": p_true,"confidence": max(p,1-p),...}

`Answer.confidence` is the probability of the chosen answer (top probability), not laya's
entropy-based `confidence` for choice/score: only a probability can be calibrated, measured
with ECE and compared against a decision threshold. For noul the two coincide.

Calibration is post-hoc temperature scaling of the returned probabilities:
  p' = softmax(log(p) / T), noul via the 2-vector [1 - p_true, p_true]. T = 1 is the identity.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import Answer, Detail, ItemResult
from ..models.library import ExpectedValue

_EPS = 1e-12


def answer_probabilities(raw_answer: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Ordered (keys, probs) for any answer type; noul -> (['false','true'], [1-p, p])."""
    t = raw_answer["type"]
    if t == "noul":
        p = float(raw_answer["noul"])
        return ["false", "true"], [1.0 - p, p]
    probs = raw_answer["probabilities"]
    if t == "score":
        keys = sorted(probs, key=int)
    else:
        keys = list(probs)
    return keys, [float(probs[k]) for k in keys]


def scale_probs(probs: list[float] | np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    if temperature == 1.0:
        return p / p.sum()
    z = np.log(p) / temperature
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(raw_answer: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Return a NEW raw answer with probabilities rescaled and derived fields recomputed."""
    out = dict(raw_answer)
    if temperature == 1.0 or not math.isfinite(temperature) or temperature <= 0:
        return out
    keys, probs = answer_probabilities(raw_answer)
    p = scale_probs(probs, temperature)
    t = raw_answer["type"]
    if t == "noul":
        out["noul"] = round(float(p[1]), 4)
        out["confidence"] = round(max(float(p[1]), 1.0 - float(p[1])), 4)
        return out
    out["probabilities"] = {k: round(float(v), 4) for k, v in zip(keys, p)}
    if t == "choice":
        out["choice"] = keys[int(p.argmax())]
    else:
        out["score"] = round(float((np.arange(len(p)) * p).sum()), 4)
    return out


def _top(raw_answer: dict[str, Any]) -> tuple[str, float]:
    keys, probs = answer_probabilities(raw_answer)
    i = int(np.argmax(probs))
    return keys[i], float(probs[i])


def predicted_value(raw_answer: dict[str, Any], temperature: float = 1.0) -> ExpectedValue:
    """Value comparable with LabeledExample.expected: choice label, score argmax level, noul bool."""
    ans = apply_temperature(raw_answer, temperature)
    t = ans["type"]
    if t == "noul":
        return float(ans["noul"]) >= 0.5
    key, _ = _top(ans)
    return int(key) if t == "score" else key


def top_probability(raw_answer: dict[str, Any], temperature: float = 1.0) -> float:
    return _top(apply_temperature(raw_answer, temperature))[1]


def normalize_answer(
    raw_answer: dict[str, Any],
    *,
    detail: Detail = "compact",
    temperature: float = 1.0,
    threshold: float | None = None,
    unverified: bool = False,
) -> Answer:
    ans = apply_temperature(raw_answer, temperature)
    t = ans["type"]
    key, conf = _top(ans)
    value: str | float | bool
    level = p_true = None
    if t == "choice":
        value = key
    elif t == "score":
        value = round(float(ans["score"]), 4)
        level = int(key)
    else:
        p_true = round(float(ans["noul"]), 4)
        value = p_true >= 0.5

    status = None
    if unverified:
        status = "unverified"
    elif threshold is not None:
        status = "decided" if conf >= threshold else "needs_review"

    full = detail == "full"
    probabilities = None
    if full:
        keys, probs = answer_probabilities(ans)
        probabilities = {k: round(float(v), 4) for k, v in zip(keys, probs)}
    return Answer(
        type=t,
        value=value,
        confidence=round(conf, 4),
        level=level,
        p_true=p_true,
        status=status,
        probabilities=probabilities,
        legend=ans.get("legend") if full and t == "score" else None,
    )


def to_item_result(
    index: int,
    raw_result: dict[str, Any],
    *,
    detail: Detail = "compact",
    temperatures: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    default_threshold: float | None = None,
    unverified: bool = False,
) -> ItemResult:
    temperatures = temperatures or {}
    thresholds = thresholds or {}
    answers: dict[str, Answer] = {}
    needs_review: list[str] = []
    for qid, raw in raw_result.get("answers", {}).items():
        thr = thresholds.get(qid, default_threshold)
        a = normalize_answer(
            raw,
            detail=detail,
            temperature=temperatures.get(qid, 1.0),
            threshold=None if unverified else thr,
            unverified=unverified,
        )
        answers[qid] = a
        if a.status == "needs_review" or (unverified and thr is not None and a.confidence < thr):
            needs_review.append(qid)
    routing = raw_result.get("routing") or {}
    return ItemResult(
        index=index,
        answers=answers,
        model=routing.get("model", "unknown"),
        route_reason=routing.get("reason"),
        needs_review=needs_review,
    )
