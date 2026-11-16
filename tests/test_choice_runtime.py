import importlib

import pytest


def test_budget_counts_allocations_once_and_reserves_running_limits():
    m = importlib.import_module("pid_choice_runtime")
    rows = "10|COMPLETED|600|20|cpu=12,gres/gpu=1\n10.batch|COMPLETED|600|20|cpu=12\n11|RUNNING|300|30|cpu=12,gres/gpu=1\n12|PENDING|0|10|\n"
    r = m.accounting(rows, ["10", "11", "12"])
    assert r["gpu_seconds"] == 900
    assert r["reserved_gpu_seconds"] == 3000
    assert r["active_gpus"] == 1


def test_budget_rejects_missing_jobs_overspend_and_third_gpu():
    m = importlib.import_module("pid_choice_runtime")
    with pytest.raises(ValueError, match="missing"):
        m.accounting("10|COMPLETED|1|1|gres/gpu=1", ["10", "11"])
    with pytest.raises(ValueError, match="budget"):
        m.accounting("10|RUNNING|1|361|gres/gpu=1", ["10"])
    with pytest.raises(ValueError, match="GPU"):
        m.accounting("10|RUNNING|1|1|gres/gpu=3", ["10"])


def test_failed_or_pending_jobs_cannot_disappear_from_accounting():
    m = importlib.import_module("pid_choice_runtime")
    r = m.accounting("10|FAILED|60|10|gres/gpu=1\n11|CANCELLED by 1|0|10|", ["10", "11"])
    assert r["gpu_seconds"] == r["reserved_gpu_seconds"] == 60
