import importlib

import numpy as np
import torch


def test_approximation_gate_accepts_small_tilt_rejects_large_tilt():
    m = importlib.import_module("pid_choice_audit")
    lp = torch.tensor([[.25, .75]], dtype=torch.float64).log()
    phi = torch.tensor([[[0.], [2.]]], dtype=torch.float64)
    p = torch.tensor([1])
    exact, approx = m.position_scores(lp, phi, p, torch.tensor([[.01]], dtype=torch.float64))
    assert m.error_gate((exact - approx).abs().numpy())["passed"]
    exact, approx = m.position_scores(lp, phi, p, torch.tensor([[5.]], dtype=torch.float64))
    assert not m.error_gate((exact - approx).abs().numpy())["passed"]


def test_rank_audit_uses_standardised_margin_and_rejects_disagreement():
    m = importlib.import_module("pid_choice_audit")
    exact = np.tile([0., 1., 2.], (70, 1))
    good = m.rank_gate(exact, exact * 1e-8, np.zeros(70, dtype=int))
    assert good["passed"] and good["eligible_pairs"] == 140
    assert not m.rank_gate(exact, -exact, np.zeros(70, dtype=int))["passed"]
    assert not m.rank_gate(np.zeros((70, 3)), np.zeros((70, 3)), np.zeros(70, dtype=int))["passed"]
