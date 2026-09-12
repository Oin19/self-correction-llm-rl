"""Tests for evaluation metrics (Pass@1, Fix@K, error distribution, recovery rate)."""

import pytest
from src.evaluation.metrics import (
    calculate_pass_at_1,
    calculate_fix_at_k,
    calculate_error_distribution,
    calculate_recovery_rate,
    summarize_eval_metrics,
)


def test_pass_at_1():
    results = [
        {"status": "AC"},
        {"status": "WA"},
        {"status": "AC"},
        {"status": "CE"},
    ]
    assert calculate_pass_at_1(results) == 0.5


def test_fix_at_k():
    sessions = [
        {"solved": True, "solved_turn": 1, "history": [{"status": "AC"}]},
        {"solved": True, "solved_turn": 3, "history": [{"status": "WA"}, {"status": "RE"}, {"status": "AC"}]},
        {"solved": False, "solved_turn": None, "history": [{"status": "WA"}, {"status": "WA"}, {"status": "WA"}]},
    ]

    assert calculate_fix_at_k(sessions, k=1) == pytest.approx(1 / 3)
    assert calculate_fix_at_k(sessions, k=3) == pytest.approx(2 / 3)
    assert calculate_fix_at_k(sessions, k=5) == pytest.approx(2 / 3)


def test_error_distribution():
    results = [
        {"status": "AC"},
        {"status": "WA"},
        {"status": "WA"},
        {"status": "TLE"},
    ]
    dist = calculate_error_distribution(results)
    assert dist["AC"] == 0.25
    assert dist["WA"] == 0.50
    assert dist["TLE"] == 0.25


def test_recovery_rate():
    sessions = [
        {"solved": True, "history": [{"status": "WA"}, {"status": "AC"}]},
        {"solved": False, "history": [{"status": "RE"}, {"status": "RE"}]},
    ]
    assert calculate_recovery_rate(sessions) == 0.5


def test_summarize_metrics():
    sessions = [
        {"solved": True, "solved_turn": 1, "status": "AC", "history": [{"status": "AC"}]},
        {"solved": True, "solved_turn": 2, "status": "AC", "history": [{"status": "WA"}, {"status": "AC"}]},
    ]
    summary = summarize_eval_metrics(sessions)
    assert summary["pass_at_1"] == 1.0
    assert summary["fix_at_1"] == 0.5
    assert summary["fix_at_5"] == 1.0
    assert summary["recovery_rate"] == 1.0
