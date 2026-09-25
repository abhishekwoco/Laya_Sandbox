"""Per-question temperature scaling fitted on labeled evaluation results.

Laya returns probabilities, not logits, so calibration works on log-probabilities:
    p' = softmax(log p / T)          (noul: the 2-vector [1 - p_true, p_true])
exactly as `engine.answers.apply_temperature` applies it at decision time. T > 1 softens an
over-confident question, T < 1 sharpens an under-confident one; the argmax never changes, so
accuracy is unaffected and only the confidence (and therefore thresholds and ECE) moves.

T is found by minimising the mean negative log-likelihood of the expected answers. The NLL is
convex in beta = 1/T, so a bounded scalar search over beta finds the global optimum.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from laya.common import ece_score
from scipy.optimize import minimize_scalar

from ..engine.answers import answer_probabilities, predicted_value, top_probability
from ..models import LabeledExample, SchemaInfo
from .evaluation import _canonical_prediction, coerce_expected, question_labels

MIN_FIT_SAMPLES = 10
_EPS = 1e-12


def fit_temperature(
    probs: list[list[float]], correct_idx: list[int], bounds: tuple[float, float] = (0.25, 10.0)
) -> float:
    """Temperature minimising NLL of softmax(log p / T) at the correct indices.

    Returns 1.0 (no change) when there are fewer than MIN_FIT_SAMPLES usable samples or the
    problem is degenerate: every top answer right (or every one wrong), which pushes T to a
    bound, or probability vectors that carry no information (all uniform)."""
    rows = [
        (np.log(np.clip(np.asarray(p, dtype=float), _EPS, 1.0)), int(y))
        for p, y in zip(probs, correct_idx)
        if len(p) >= 2 and 0 <= int(y) < len(p)
    ]
    if len(rows) < MIN_FIT_SAMPLES:
        return 1.0
    hits = np.array([int(np.argmax(lp)) == y for lp, y in rows])
    if hits.all() or not hits.any():
        return 1.0
    if all(float(np.ptp(lp)) < 1e-9 for lp, _ in rows):
        return 1.0

    # group by option count so each group is one matrix
    groups: dict[int, tuple[list[np.ndarray], list[int]]] = {}
    for lp, y in rows:
        g = groups.setdefault(len(lp), ([], []))
        g[0].append(lp)
        g[1].append(y)
    mats = [(np.vstack(lps), np.asarray(ys)) for lps, ys in groups.values()]
    n = len(rows)

    def nll(beta: float) -> float:
        total = 0.0
        for lp, ys in mats:
            z = lp * beta
            z = z - z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1))
            total += float((lse - z[np.arange(len(ys)), ys]).sum())
        return total / n

    lo, hi = bounds
    res = minimize_scalar(nll, bounds=(1.0 / hi, 1.0 / lo), method="bounded", options={"xatol": 1e-6})
    beta = float(res.x)
    if not math.isfinite(beta) or beta <= 0 or nll(beta) > nll(1.0) + 1e-12:
        return 1.0
    return round(1.0 / beta, 4)


def _target_index(keys: list[str], qtype: str, expected: Any) -> int | None:
    if qtype == "noul":
        return 1 if expected else 0
    want = str(expected).lower()
    for i, k in enumerate(keys):
        if str(k).lower() == want:
            return i
    return None


def fit_schema_temperatures(
    schema: SchemaInfo,
    examples: list[LabeledExample],
    raw_results: list[tuple[int, dict[str, Any]]],
    holdout: float = 0.3,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], int, int]:
    """Fit one temperature per question on a random fit split and measure ECE on the held-out split.

    Returns (temperatures, ece_before, ece_after, n_fit, n_holdout). Temperatures cover every
    question (1.0 where nothing could be fitted); ECE dicts only contain questions that have
    labeled answers in the holdout split. Raw results must be UNCALIBRATED (as jobs store them),
    so the fitted temperature is absolute, not relative to a previous calibration."""
    by_index = {int(i): r for i, r in raw_results}
    evaluated = [i for i in range(len(examples)) if i in by_index]
    perm = np.random.default_rng(seed).permutation(len(evaluated))
    n_hold = int(round(len(evaluated) * holdout))
    if len(evaluated) >= 2:
        n_hold = min(max(n_hold, 1), len(evaluated) - 1)
    else:
        n_hold = 0
    hold = {evaluated[j] for j in perm[:n_hold]}

    temperatures: dict[str, float] = {}
    ece_before: dict[str, float] = {}
    ece_after: dict[str, float] = {}
    for qid, q in schema.questions.items():
        answered = [
            (i, examples[i].expected[qid], by_index[i]["answers"][qid])
            for i in evaluated
            if qid in examples[i].expected and qid in (by_index[i].get("answers") or {})
        ]
        labels = question_labels(q, [raw for _, _, raw in answered])
        fit_p: list[list[float]] = []
        fit_y: list[int] = []
        held: list[tuple[Any, dict[str, Any]]] = []
        for i, expected_raw, raw in answered:
            try:
                exp = coerce_expected(q, labels, expected_raw)
            except ValueError:
                continue
            keys, probs = answer_probabilities(raw)
            y = _target_index(keys, q.type, exp)
            if y is None:
                continue
            if i in hold:
                held.append((exp, raw))
            else:
                fit_p.append(probs)
                fit_y.append(y)

        temp = fit_temperature(fit_p, fit_y)
        temperatures[qid] = temp
        if held:
            correct = np.array(
                [_canonical_prediction(q, labels, predicted_value(raw)) == exp for exp, raw in held], dtype=float
            )
            before = np.array([top_probability(raw) for _, raw in held])
            after = np.array([top_probability(raw, temp) for _, raw in held])
            ece_before[qid] = round(ece_score(before, correct), 4)
            ece_after[qid] = round(ece_score(after, correct), 4)

    return temperatures, ece_before, ece_after, len(evaluated) - n_hold, n_hold
