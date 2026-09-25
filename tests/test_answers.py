import math

import pytest

from laya_mcp.engine.answers import (
    answer_probabilities,
    apply_temperature,
    normalize_answer,
    predicted_value,
    to_item_result,
)

CHOICE = {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}, "confidence": 0.3}
SCORE = {"type": "score", "score": 1.2, "legend": {"0": "low", "1": "mid", "2": "high"},
         "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}, "confidence": 0.1}
NOUL = {"type": "noul", "noul": 0.8, "confidence": 0.8}


def test_identity_temperature_keeps_values():
    assert apply_temperature(CHOICE, 1.0) == CHOICE
    a = normalize_answer(CHOICE)
    assert a.value == "a" and a.confidence == pytest.approx(0.7) and a.status is None


def test_temperature_softens_and_is_pure():
    soft = apply_temperature(CHOICE, 2.0)
    assert soft["probabilities"]["a"] < 0.7
    assert CHOICE["probabilities"]["a"] == 0.7
    assert sum(soft["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert soft["choice"] == "a"


def test_noul_temperature():
    soft = apply_temperature(NOUL, 2.0)
    expected = 1 / (1 + math.exp(-(math.log(0.8) - math.log(0.2)) / 2))
    assert soft["noul"] == pytest.approx(expected, abs=1e-3)


def test_score_value_level_and_prediction():
    a = normalize_answer(SCORE, detail="full")
    assert a.value == pytest.approx(1.2) and a.level in (1, 2)
    assert a.legend == SCORE["legend"] and a.probabilities is not None
    assert predicted_value(SCORE) in (1, 2)


def test_threshold_status():
    assert normalize_answer(CHOICE, threshold=0.6).status == "decided"
    assert normalize_answer(CHOICE, threshold=0.9).status == "needs_review"
    assert normalize_answer(CHOICE, threshold=0.9, unverified=True).status == "unverified"


def test_noul_answer_and_probabilities():
    a = normalize_answer(NOUL)
    assert a.value is True and a.p_true == 0.8 and a.confidence == 0.8
    assert answer_probabilities(NOUL) == (["false", "true"], [pytest.approx(0.2), 0.8])
    assert predicted_value(NOUL) is True


def test_item_result_needs_review():
    raw = {"answers": {"c": CHOICE, "n": NOUL}, "routing": {"model": "english", "reason": "latin"}}
    r = to_item_result(3, raw, thresholds={"c": 0.9}, default_threshold=0.5)
    assert r.index == 3 and r.model == "english"
    assert r.needs_review == ["c"]
    assert r.answers["n"].status == "decided"
