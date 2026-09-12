from src.execution.status import STATUSES


def test_execution_statuses():
    assert "AC" in STATUSES
    assert "WA" in STATUSES
    assert "TLE" in STATUSES
    assert "RE" in STATUSES
