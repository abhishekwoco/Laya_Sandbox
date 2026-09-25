"""build_report against hand-computed fixtures."""
from datetime import datetime, timezone

import numpy as np
import pytest

from laya_mcp.models import LabeledExample, Question, SchemaInfo, SchemaTargets
from laya_mcp.workbench.evaluation import build_report, coerce_expected, recommend_threshold

QUESTIONS = {
    "team": Question(type="choice", instructions="Which team owns it?",
                     criteria={"a": "team a", "b": "team b", "c": "team c"}),
    "sev": Question(type="score", instructions="How severe?", criteria=["low", "mid", "high"]),
    "bug": Question(type="noul", instructions="Is this a bug?"),
}


def schema(questions=None, min_accuracy=0.9, min_examples=10, per_question=None) -> SchemaInfo:
    return SchemaInfo(
        team="dev", name="triage", version=1, status="draft", questions=questions or QUESTIONS,
        targets=SchemaTargets(min_accuracy=min_accuracy, min_examples=min_examples, per_question=per_question or {}),
        created_at=datetime.now(timezone.utc),
    )


def choice(pa, pb, pc):
    probs = {"a": pa, "b": pb, "c": pc}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": 0.0}


def score(p0, p1, p2):
    return {"type": "score", "score": p1 + 2 * p2, "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": p0, "1": p1, "2": p2}, "confidence": 0.0}


def noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


FILLER_TEAM = choice(0.34, 0.33, 0.33)
FILLER_SEV = score(0.34, 0.33, 0.33)

# team: labeled on 0..3  -> 3/4 correct; the miss (example 1) is the most confident answer
TEAM = {0: ("a", choice(0.7, 0.2, 0.1)), 1: ("B", choice(0.82, 0.13, 0.05)),
        2: ("c", choice(0.25, 0.25, 0.5)), 3: ("b", choice(0.05, 0.9, 0.05))}
# sev: labeled on 4..6; example 6 has an unreadable label
SEV = {4: (2, score(0.1, 0.2, 0.7)), 5: ("0", score(0.6, 0.3, 0.1)), 6: ("high", score(0.1, 0.1, 0.8))}
BUG_TRUE = [True, "yes", 1, "true", "TRUE"]


def bug_label(i):
    if i < 10:
        return BUG_TRUE[i % 5], noul(0.95)       # confident and right
    if i < 15:
        return BUG_TRUE[i % 5], noul(0.62)       # unsure and right
    return ["no", False, 0, "false", "No"][i % 5], noul(0.62)   # unsure and wrong


def fixture():
    examples, raw = [], []
    for i in range(21):
        expected = {}
        answers = {"team": FILLER_TEAM, "sev": FILLER_SEV}
        exp_bug, ans_bug = bug_label(min(i, 19))
        expected["bug"] = exp_bug
        answers["bug"] = ans_bug
        if i in TEAM:
            expected["team"], answers["team"] = TEAM[i]
        if i in SEV:
            expected["sev"], answers["sev"] = SEV[i]
        if i == 7:
            expected["extra"] = "x"
        examples.append(LabeledExample(state=f"item {i}", expected=expected))
        if i < 20:  # example 20 has no result (failed item)
            model = "english" if i < 10 else "multilingual"
            raw.append((i, {"answers": answers, "routing": {"model": model, "reason": "test"}}))
    return examples, raw


def report(**kw):
    examples, raw = fixture()
    args = dict(dataset="dev/issues", job_id="job1", temperatures={}, target_accuracy=0.9, min_examples=10)
    args.update(kw)
    return build_report(schema(), examples, raw, **args)


def test_choice_metrics():
    q = report().per_question["team"]
    assert q.type == "choice" and q.n == 4
    assert q.accuracy == 0.75
    assert q.labels == ["a", "b", "c"]
    assert q.confusion == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert q.per_label["a"].model_dump() == {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1}
    assert q.per_label["b"].model_dump() == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert q.per_label["c"].f1 == 1.0
    # 15 bins: .7 right, .82 wrong, .5 right, .9 right, each 1/4 of the answers
    assert q.ece == pytest.approx(0.25 * (0.3 + 0.82 + 0.5 + 0.1), abs=1e-4)
    assert q.mae is None
    assert q.recommended_threshold is None and not q.meets_target


def test_score_metrics_and_bad_labels():
    r = report()
    q = r.per_question["sev"]
    assert q.n == 2 and q.accuracy == 1.0
    assert q.mae == pytest.approx((abs(2 - 1.6) + abs(0 - 0.5)) / 2, abs=1e-4)
    assert q.labels == ["0", "1", "2"]
    assert q.confusion == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert any(n.startswith("sev: 1 expected value(s) could not be read") and "'high'" in n for n in r.notes)


def test_noul_metrics_threshold_and_bands():
    q = report().per_question["bug"]
    assert q.n == 20 and q.accuracy == 0.75
    assert q.labels == ["false", "true"]
    assert q.confusion == [[0, 5], [0, 15]]
    assert q.per_label["true"].precision == 0.75 and q.per_label["true"].recall == 1.0
    assert q.per_label["false"].model_dump() == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 5}
    # ECE: .95 bin all right (|.95-1| * .5) + .62 bin half right (|.62-.5| * .5)
    assert q.ece == pytest.approx(0.025 + 0.06, abs=1e-4)
    # at t=.62 all 20 answers count (75%); from .63 only the ten .95 answers (100%, n=10)
    assert q.recommended_threshold == 0.63
    assert q.accuracy_at_threshold == 1.0 and q.coverage_at_threshold == 0.5
    assert q.meets_target
    bands = {(b.lo, b.hi): (b.n, b.accuracy) for b in q.bands}
    assert bands[(0.6, 0.7)] == (10, 0.5)
    assert bands[(0.9, 1.0)] == (10, 1.0)
    assert bands[(0.5, 0.6)] == (0, None)
    assert sum(b.n for b in q.bands) == 20


def test_report_level_fields():
    r = report()
    assert r.schema_ref == "dev/triage@1" and r.dataset == "dev/issues" and r.job_id == "job1"
    assert r.n_examples == 20
    assert r.overall_accuracy == pytest.approx(round((3 + 2 + 15) / 26, 4))
    assert r.model_counts == {"english": 10, "multilingual": 10}
    assert not r.passes_targets          # team has 4 answers and no threshold
    assert not r.calibrated
    assert r.worst_misses[0].model_dump() == {
        "example_index": 1, "question_id": "team", "expected": "b", "predicted": "a", "confidence": 0.82,
    }
    assert [m.example_index for m in r.worst_misses[1:]] == [15, 16, 17, 18, 19]
    assert all(m.expected is False and m.predicted is True for m in r.worst_misses[1:])
    notes = "\n".join(r.notes)
    assert "1 of 21 examples have no result" in notes
    assert "team: too few examples for a reliable threshold" in notes
    assert "ignored: extra" in notes
    assert "uncalibrated" in notes


def test_passes_with_single_good_question():
    s = schema(questions={"bug": QUESTIONS["bug"]})
    examples, raw = fixture()
    r = build_report(s, examples, raw, dataset="d/x", job_id=None, temperatures={}, target_accuracy=0.9, min_examples=10)
    assert r.passes_targets and r.overall_accuracy == 0.75
    # same data, but the schema needs more examples than it has
    r = build_report(s, examples, raw, dataset="d/x", job_id=None, temperatures={}, target_accuracy=0.9, min_examples=50)
    assert not r.passes_targets
    assert any("promotion needs at least 50" in n for n in r.notes)


def test_per_question_target_override():
    s = schema(questions={"bug": QUESTIONS["bug"]}, per_question={"bug": 0.7})
    examples, raw = fixture()
    q = build_report(s, examples, raw, dataset="d/x", job_id=None, temperatures={}, target_accuracy=0.9,
                     min_examples=10).per_question["bug"]
    assert q.recommended_threshold == 0.5 and q.coverage_at_threshold == 1.0


def test_temperature_is_applied():
    s = schema(questions={"bug": QUESTIONS["bug"]})
    examples, raw = fixture()
    r = build_report(s, examples, raw, dataset="d/x", job_id=None, temperatures={"bug": 2.0}, target_accuracy=0.9,
                     min_examples=10)
    q = r.per_question["bug"]
    assert r.calibrated and q.temperature == 2.0
    hi = np.sqrt(0.95) / (np.sqrt(0.95) + np.sqrt(0.05))   # 0.8134
    lo = np.sqrt(0.62) / (np.sqrt(0.62) + np.sqrt(0.38))   # 0.5609
    assert q.recommended_threshold == pytest.approx(np.floor(lo * 100 + 1) / 100)   # first grid point above .5609
    assert q.accuracy == 0.75  # argmax unchanged
    bands = {(b.lo, b.hi): b.n for b in q.bands}
    assert bands[(0.5, 0.6)] == 10 and bands[(0.8, 0.9)] == 10
    assert q.ece == pytest.approx(0.5 * abs(hi - 1) + 0.5 * abs(lo - 0.5), abs=1e-3)


def test_question_without_labels():
    s = schema(questions={**QUESTIONS, "extra_q": Question(type="noul", instructions="Is it urgent?")})
    examples, raw = fixture()
    r = build_report(s, examples, raw, dataset="d/x", job_id=None, temperatures={}, target_accuracy=0.9, min_examples=10)
    assert r.per_question["extra_q"].n == 0 and not r.per_question["extra_q"].meets_target
    assert any(n.startswith("extra_q: no labeled examples") for n in r.notes)


def test_recommend_threshold_needs_ten_answers():
    conf = np.array([0.99] * 9 + [0.55] * 11)
    correct = np.array([True] * 9 + [False] * 11)
    thr, acc, cov, best = recommend_threshold(conf, correct, 0.9)
    assert thr is None and acc is None and cov is None
    assert best is not None and best[1] == pytest.approx(9 / 20)


@pytest.mark.parametrize(
    "qid,value,expected",
    [
        ("bug", "Yes", True), ("bug", 0, False), ("bug", "false", False), ("bug", 1.0, True),
        ("sev", "2", 2), ("sev", 1.0, 1), ("sev", " 0 ", 0),
        ("team", "A", "a"), ("team", " b ", "b"),
    ],
)
def test_coerce_expected(qid, value, expected):
    q = QUESTIONS[qid]
    labels = {"bug": ["false", "true"], "sev": ["0", "1", "2"], "team": ["a", "b", "c"]}[qid]
    out = coerce_expected(q, labels, value)
    assert out == expected and type(out) is type(expected)


@pytest.mark.parametrize("qid,value", [("bug", "maybe"), ("bug", 2), ("sev", 3), ("sev", True), ("sev", "1.5"),
                                       ("team", "d"), ("team", True)])
def test_coerce_expected_rejects(qid, value):
    labels = {"bug": ["false", "true"], "sev": ["0", "1", "2"], "team": ["a", "b", "c"]}[qid]
    with pytest.raises(ValueError):
        coerce_expected(QUESTIONS[qid], labels, value)
