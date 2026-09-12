"""Tests for the evaluation parsing helpers.

We don't call the LLM here — we test the pure functions that interpret
its output, since those are the ones that break silently.
"""

from documind.evaluator import _extract_float


def test_extract_float_plain_number():
    assert _extract_float("0.85") == 0.85
    assert _extract_float("1.0") == 1.0
    assert _extract_float("0") == 0.0


def test_extract_float_from_json():
    assert _extract_float('{"score": 0.72}') == 0.72
    assert _extract_float('{"value": 0.4}') == 0.4
    assert _extract_float('{"faithfulness": 0.9}') == 0.9


def test_extract_float_with_prose():
    assert _extract_float("The score is 0.63 because...") == 0.63
    assert _extract_float("I'd rate this 0.5.") == 0.5


def test_extract_float_empty_or_garbage():
    assert _extract_float("") is None
    assert _extract_float("no number here") is None
