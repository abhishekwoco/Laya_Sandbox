"""Temperature fitting: recovery of a known temperature and ECE improvement on held-out data."""
from datetime import datetime, timezone

import numpy as np
import pytest

from laya_mcp.engine.answers import scale_probs
from laya_mcp.models import LabeledExample, Question, SchemaInfo, SchemaTargets
from laya_mcp.workbench.calibration import fit_schema_temperatures, fit_temperature
from laya_mcp.workbench.evaluation import build_report


def _softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def synthetic(n: int, k: int, true_temp: float, seed: int = 0):
    """Labels drawn from softmax(z); the model publishes softmax(z * true_temp), i.e. it is
    over-confident for true_temp > 1. softmax(log p / true_temp) restores the truth."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, k))
    truth = _softmax(z)
    y = np.array([rng.choice(k, p=p) for p in truth])
    published = _softmax(z * true_temp)
    return published, y


@pytest.mark.parametrize("true_temp,k", [(2.5, 4), (1.8, 2), (0.6, 3)])
def test_recovers_known_temperature(true_temp, k):
    probs, y = synthetic(6000, k, true_temp)
    t = fit_temperature(probs.tolist(), y.tolist())
    assert t == pytest.approx(true_temp, rel=0.08)


def test_identity_when_already_calibrated():
    probs, y = synthetic(6000, 3, 1.0, seed=3)
    assert fit_temperature(probs.tolist(), y.tolist()) == pytest.approx(1.0, abs=0.08)


def test_respects_bounds():
    probs, y = synthetic(3000, 3, 8.0, seed=1)
    assert fit_temperature(probs.tolist(), y.tolist(), bounds=(0.5, 3.0)) == pytest.approx(3.0, abs=1e-3)


def test_degenerate_inputs_return_one():
    assert fit_temperature([[0.9, 0.1]] * 9, [0] * 9) == 1.0                 # too few samples
    assert fit_temperature([[0.7, 0.3]] * 50, [0] * 50) == 1.0               # always right -> unbounded sharpening
    assert fit_temperature([[0.7, 0.3]] * 50, [1] * 50) == 1.0               # always wrong
    assert fit_temperature([[0.5, 0.5]] * 30, [0, 1] * 15) == 1.0            # no information
    assert fit_temperature([[0.9, 0.1]] * 30, [5] * 30) == 1.0               # invalid indices are dropped


def _schema() -> SchemaInfo:
    return SchemaInfo(
        team="dev", name="cal", version=1, status="evaluated",
        questions={
            "area": Question(type="choice", instructions="Which area?", criteria=["api", "ui", "db", "infra"]),
            "bug": Question(type="noul", instructions="Is it a bug?"),
        },
        targets=SchemaTargets(min_accuracy=0.9, min_examples=10), created_at=datetime.now(timezone.utc),
    )


def _dataset(n=600, temp=2.5):
    labels = ["api", "ui", "db", "infra"]
    choice_p, choice_y = synthetic(n, 4, temp, seed=7)
    noul_p, noul_y = synthetic(n, 2, temp, seed=8)
    examples, raw = [], []
    for i in range(n):
        examples.append(LabeledExample(state=f"s{i}", expected={"area": labels[choice_y[i]], "bug": bool(noul_y[i])}))
        answers = {
            "area": {"type": "choice", "choice": labels[int(choice_p[i].argmax())],
                     "probabilities": {lab: float(p) for lab, p in zip(labels, choice_p[i])}, "confidence": 0.0},
            "bug": {"type": "noul", "noul": float(noul_p[i][1]), "confidence": float(noul_p[i].max())},
        }
        raw.append((i, {"answers": answers, "routing": {"model": "english"}}))
    return examples, raw


def test_fit_schema_temperatures_improves_holdout_ece():
    examples, raw = _dataset()
    temps, before, after, n_fit, n_hold = fit_schema_temperatures(_schema(), examples, raw, holdout=0.3, seed=0)
    assert n_fit + n_hold == 600 and n_hold == 180
    assert set(temps) == {"area", "bug"}
    assert temps["area"] == pytest.approx(2.5, rel=0.2)
    assert temps["bug"] == pytest.approx(2.5, rel=0.25)
    for qid in ("area", "bug"):
        assert after[qid] < 0.75 * before[qid]   # binned ECE on 180 held-out answers keeps some sampling noise


def test_calibrated_report_is_better_calibrated():
    examples, raw = _dataset()
    schema = _schema()
    temps, *_ = fit_schema_temperatures(schema, examples, raw)
    kw = dict(dataset="dev/x", job_id="j", target_accuracy=0.9, min_examples=10)
    plain = build_report(schema, examples, raw, temperatures={}, **kw)
    cal = build_report(schema, examples, raw, temperatures=temps, **kw)
    assert cal.calibrated and not plain.calibrated
    for qid in ("area", "bug"):
        assert cal.per_question[qid].ece < plain.per_question[qid].ece
        assert cal.per_question[qid].accuracy == plain.per_question[qid].accuracy   # argmax unchanged


def test_skips_missing_and_unreadable_labels():
    examples, raw = _dataset(n=60)
    examples[0] = LabeledExample(state="s0", expected={"area": "nonsense"})
    raw = raw[:-5]                                    # five examples without results
    temps, before, after, n_fit, n_hold = fit_schema_temperatures(_schema(), examples, raw, holdout=0.3)
    assert n_fit + n_hold == 55
    assert all(np.isfinite(v) for v in temps.values())


def test_fit_matches_apply_temperature():
    """The fitted T, applied the way decisions apply it, minimises NLL."""
    probs, y = synthetic(4000, 3, 2.0, seed=5)
    t = fit_temperature(probs.tolist(), y.tolist())

    def nll(temp):
        return -np.mean([np.log(scale_probs(p, temp)[yy]) for p, yy in zip(probs, y)])

    assert nll(t) <= nll(t * 1.1) and nll(t) <= nll(t / 1.1)
