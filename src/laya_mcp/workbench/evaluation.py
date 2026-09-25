"""Evaluation metrics for a schema version over a labeled dataset. Pure functions, no I/O.

Definitions (per question; examples whose `expected` lacks the question are skipped):
- correct: `predicted_value(raw, T)` equals the coerced expected value
  (choice: label, case-insensitive; score: level index; noul: bool).
- confidence: `top_probability(raw, T)`, the probability of the chosen answer after the
  question's calibration temperature T. It is what thresholds and ECE are computed on, and
  what `laya_decide` compares against a threshold.
- accuracy: share of correct answers. overall_accuracy pools every scored answer of every question.
- mae (score only): mean |expected level - expected score|, expected score = sum(level * p(level)).
- ece: `laya.common.ece_score` (15 equal-width bins) on (confidence, correct).
- bands: accuracy of answers whose confidence falls in [0,.5), [.5,.6) ... [.9,1].
- recommended_threshold: smallest t on the grid 0.50, 0.51 ... 0.99 at which the answers with
  confidence >= t reach the target accuracy AND at least MIN_ANSWERS_AT_THRESHOLD answers clear t.
  coverage_at_threshold = share of answers that clear it. meets_target = such a t exists.
- passes_targets: every question meets its target AND has >= min_examples labeled answers AND the
  dataset has >= min_examples evaluated examples.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from laya.common import ece_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..engine.answers import answer_probabilities, apply_temperature, predicted_value, top_probability
from ..models import (
    ConfidenceBand,
    EvalReport,
    LabeledExample,
    LabelMetrics,
    Miss,
    Question,
    QuestionMetrics,
    SchemaInfo,
)
from ..models.library import ExpectedValue

BAND_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
THRESHOLD_GRID = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))  # 0.50 .. 0.99
MIN_ANSWERS_AT_THRESHOLD = 10
WORST_MISSES = 10
_TOL = 1e-9

_TRUE = {"true", "yes", "y", "1", "1.0"}
_FALSE = {"false", "no", "n", "0", "0.0"}


# --------------------------------------------------------------------------- labels / coercion
def question_labels(question: Question, raw_answers: list[dict[str, Any]] | None = None) -> list[str]:
    """Label order used for confusion matrices: choice labels, '0'..'k-1' for score, false/true for noul."""
    crit = question.criteria
    if question.type == "noul":
        return ["false", "true"]
    if question.type == "choice":
        if crit:
            return [str(k) for k in (crit.keys() if isinstance(crit, dict) else crit)]
    elif crit:
        return [str(i) for i in range(len(crit))]
    # criteria missing: fall back to whatever the model answered with
    seen: dict[str, None] = {}
    for raw in raw_answers or []:
        for k in answer_probabilities(raw)[0]:
            seen.setdefault(str(k), None)
    keys = list(seen)
    return sorted(keys, key=int) if question.type == "score" else keys


def coerce_expected(question: Question, labels: list[str], value: Any) -> ExpectedValue:
    """Canonical expected value (bool / level int / label str). Raises ValueError when unreadable."""
    t = question.type
    if t == "noul":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUE:
                return True
            if v in _FALSE:
                return False
        raise ValueError(f"{value!r} is not a yes/no answer (use true/false)")
    if t == "score":
        level: int | None = None
        if isinstance(value, bool):
            level = None
        elif isinstance(value, int):
            level = value
        elif isinstance(value, float) and value.is_integer():
            level = int(value)
        elif isinstance(value, str):
            try:
                f = float(value.strip())
            except ValueError:
                f = None
            if f is not None and f.is_integer():
                level = int(f)
        if level is None:
            raise ValueError(f"{value!r} is not a level index (use an integer 0..{len(labels) - 1})")
        if labels and not 0 <= level < len(labels):
            raise ValueError(f"level {level} is out of range 0..{len(labels) - 1}")
        return level
    # choice
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{value!r} is not one of the labels {labels}")
    key = str(value).strip().lower()
    for lab in labels:
        if lab.lower() == key:
            return lab
    raise ValueError(f"{value!r} is not one of the labels {labels}")


def label_of(value: ExpectedValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _canonical_prediction(question: Question, labels: list[str], pred: ExpectedValue) -> ExpectedValue:
    if question.type == "choice" and isinstance(pred, str):
        low = pred.lower()
        for lab in labels:
            if lab.lower() == low:
                return lab
    return pred


# --------------------------------------------------------------------------- metrics helpers
def confidence_bands(conf: np.ndarray, correct: np.ndarray) -> list[ConfidenceBand]:
    bands = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        upper = conf < hi - _TOL if hi < 1.0 else conf <= 1.0 + _TOL
        sel = (conf >= lo - _TOL) & upper
        n = int(sel.sum())
        bands.append(ConfidenceBand(lo=lo, hi=hi, n=n, accuracy=round(float(correct[sel].mean()), 4) if n else None))
    return bands


def recommend_threshold(
    conf: np.ndarray, correct: np.ndarray, target: float, min_answers: int = MIN_ANSWERS_AT_THRESHOLD
) -> tuple[float | None, float | None, float | None, tuple[float, float, float] | None]:
    """(threshold, accuracy_at, coverage_at, best) where best = (t, accuracy, coverage) of the most
    accurate grid point that still has `min_answers` answers (for explaining a miss)."""
    n = len(conf)
    best: tuple[float, float, float] | None = None
    for t in THRESHOLD_GRID:
        sel = conf >= t - _TOL
        k = int(sel.sum())
        if k < min_answers:
            break  # coverage only shrinks as t grows
        acc = float(correct[sel].mean())
        if best is None or acc > best[1] + _TOL:
            best = (t, acc, k / n)
        if acc >= target - _TOL:
            return t, acc, k / n, best
    return None, None, None, best


# --------------------------------------------------------------------------- report
def build_report(
    schema: SchemaInfo,
    examples: list[LabeledExample],
    raw_results: list[tuple[int, dict[str, Any]]],
    *,
    dataset: str,
    job_id: str | None,
    temperatures: dict[str, float],
    target_accuracy: float,
    min_examples: int,
) -> EvalReport:
    """Score raw (uncalibrated) laya results of an evaluate job against the dataset labels.
    `raw_results` item index == example index; examples without a result are skipped."""
    by_index = {int(i): r for i, r in raw_results}
    evaluated = [i for i in range(len(examples)) if i in by_index]
    notes: list[str] = []
    per_question: dict[str, QuestionMetrics] = {}
    misses: list[Miss] = []
    pooled_correct = pooled_total = 0

    missing = len(examples) - len(evaluated)
    if missing:
        notes.append(f"{missing} of {len(examples)} examples have no result (failed or unfinished items) and were skipped")

    unknown = sorted({k for ex in examples for k in ex.expected if k not in schema.questions})
    if unknown:
        notes.append(f"expected labels for question ids not in this schema were ignored: {', '.join(unknown)}")

    targets = schema.targets
    for qid, q in schema.questions.items():
        temp = float(temperatures.get(qid, 1.0))
        target = float(targets.per_question.get(qid, target_accuracy))
        pairs = [
            (i, examples[i].expected[qid], by_index[i]["answers"][qid])
            for i in evaluated
            if qid in examples[i].expected and qid in (by_index[i].get("answers") or {})
        ]
        labels = question_labels(q, [raw for _, _, raw in pairs])

        idx: list[int] = []
        exp_vals: list[ExpectedValue] = []
        pred_vals: list[ExpectedValue] = []
        conf_l: list[float] = []
        abs_err: list[float] = []
        bad: list[Any] = []
        for i, expected_raw, raw in pairs:
            try:
                exp = coerce_expected(q, labels, expected_raw)
            except ValueError:
                bad.append(expected_raw)
                continue
            pred = _canonical_prediction(q, labels, predicted_value(raw, temp))
            idx.append(i)
            exp_vals.append(exp)
            pred_vals.append(pred)
            conf_l.append(float(top_probability(raw, temp)))
            if q.type == "score":
                keys, probs = answer_probabilities(apply_temperature(raw, temp))
                expected_score = sum(int(k) * p for k, p in zip(keys, probs))
                abs_err.append(abs(int(exp) - expected_score))

        if bad:
            sample = ", ".join(repr(b) for b in bad[:3])
            notes.append(
                f"{qid}: {len(bad)} expected value(s) could not be read and were skipped (e.g. {sample}); "
                f"valid answers: {', '.join(labels) if q.type != 'noul' else 'true/false'}"
            )

        n = len(idx)
        if n == 0:
            per_question[qid] = QuestionMetrics(type=q.type, n=0, accuracy=0.0, ece=0.0, temperature=temp, labels=labels)
            notes.append(f"{qid}: no labeled examples in the dataset; add expected[{qid!r}] to your examples")
            continue

        conf = np.asarray(conf_l, dtype=float)
        correct = np.asarray([e == p and type(e) is type(p) for e, p in zip(exp_vals, pred_vals)], dtype=bool)
        y_true = [label_of(v) for v in exp_vals]
        y_pred = [label_of(v) for v in pred_vals]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        per_label = {
            lab: LabelMetrics(precision=round(float(p), 4), recall=round(float(r), 4), f1=round(float(f), 4), support=int(s))
            for lab, p, r, f, s in zip(labels, prec, rec, f1, sup)
        }
        thr, acc_at, cov_at, best = recommend_threshold(conf, correct, target)
        accuracy = float(correct.mean())

        per_question[qid] = QuestionMetrics(
            type=q.type,
            n=n,
            accuracy=round(accuracy, 4),
            mae=round(float(np.mean(abs_err)), 4) if abs_err else None,
            ece=round(ece_score(conf, correct.astype(float)), 4),
            temperature=temp,
            labels=labels,
            confusion=cm.astype(int).tolist(),
            per_label=per_label,
            bands=confidence_bands(conf, correct),
            recommended_threshold=thr,
            accuracy_at_threshold=round(acc_at, 4) if acc_at is not None else None,
            coverage_at_threshold=round(cov_at, 4) if cov_at is not None else None,
            meets_target=thr is not None,
        )

        if n < MIN_ANSWERS_AT_THRESHOLD:
            notes.append(
                f"{qid}: too few examples for a reliable threshold (n={n}; a threshold needs at least "
                f"{MIN_ANSWERS_AT_THRESHOLD} answers above it)"
            )
        elif thr is None:
            msg = f"{qid}: no confidence threshold reaches the {target:.0%} target accuracy (overall {accuracy:.1%}"
            if best is not None:
                msg += f"; best {best[1]:.1%} at >= {best[0]:.2f}, covering {best[2]:.0%}"
            notes.append(msg + ")")
        elif cov_at is not None and cov_at < 0.5:
            notes.append(
                f"{qid}: only {cov_at:.0%} of answers clear the {thr:.2f} threshold; the rest will be flagged needs_review"
            )
        if n < min_examples:
            notes.append(f"{qid}: {n} labeled answers; promotion needs at least {min_examples}")

        pooled_correct += int(correct.sum())
        pooled_total += n
        for i, e, p, c, ok in zip(idx, exp_vals, pred_vals, conf_l, correct):
            if not ok:
                misses.append(Miss(example_index=i, question_id=qid, expected=e, predicted=p, confidence=round(c, 4)))

    n_examples = len(evaluated)
    if n_examples < min_examples:
        notes.append(f"dataset has {n_examples} evaluated examples; promotion needs at least {min_examples} (100-200 recommended)")

    calibrated = any(float(temperatures.get(qid, 1.0)) != 1.0 for qid in schema.questions)
    if not calibrated:
        notes.append(
            "confidences are uncalibrated (shipped checkpoints tend to be over-confident); run laya_calibrate "
            "to fit per-question temperatures and refresh thresholds"
        )

    passes = (
        bool(per_question)
        and n_examples >= min_examples
        and all(m.meets_target and m.n >= min_examples for m in per_question.values())
    )
    misses.sort(key=lambda m: (-m.confidence, m.example_index, m.question_id))
    model_counts = Counter(
        str((by_index[i].get("routing") or {}).get("model", "unknown")) for i in evaluated
    )
    return EvalReport(
        schema_ref=schema.ref,
        dataset=dataset,
        job_id=job_id,
        created_at=datetime.now(timezone.utc),
        n_examples=n_examples,
        target_accuracy=target_accuracy,
        calibrated=calibrated,
        per_question=per_question,
        overall_accuracy=round(pooled_correct / pooled_total, 4) if pooled_total else 0.0,
        passes_targets=passes,
        worst_misses=misses[:WORST_MISSES],
        model_counts=dict(model_counts),
        notes=notes,
    )
