import importlib

import numpy as np
import pytest
import torch


def test_exact_block_score_matches_independent_strong_tilt_and_mask():
    m = importlib.import_module("aicup_pid.choice_exact")
    torch.manual_seed(17)
    lp = torch.randn(7, 5, dtype=torch.float64).log_softmax(1)
    lp[:, 4] = -torch.inf
    phi = torch.randn(7, 5, 3, dtype=torch.float64)
    played = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    z = torch.randn(9, 3, dtype=torch.float64) * 4
    want = []
    for zz in z:
        tilt = (phi * zz).sum(2)
        want.append((tilt[torch.arange(7), played] - torch.logsumexp(lp + tilt, 1) + torch.logsumexp(lp, 1)).sum())
    got = m.exact_scores(lp, phi, played, z, time_chunk=2, candidate_chunk=3)
    torch.testing.assert_close(got, torch.stack(want), atol=1e-10, rtol=1e-10)
    offset = torch.randn(7, 1, 3, dtype=torch.float64)
    torch.testing.assert_close(got, m.exact_scores(lp, phi + offset, played, z), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(m.exact_scores(lp, phi, played, z * 0), torch.zeros(9, dtype=torch.float64), atol=1e-10, rtol=0)


def test_exact_score_rejects_illegal_observation_and_nonfinite_parameters():
    m = importlib.import_module("aicup_pid.choice_exact")
    lp = torch.tensor([[0., -torch.inf]], dtype=torch.float64)
    phi = torch.tensor([[[0.], [1.]]], dtype=torch.float64)
    with pytest.raises(ValueError):
        m.exact_scores(lp, phi, torch.tensor([1]), torch.ones(1, 1))
    with pytest.raises(ValueError):
        m.exact_scores(lp, phi, torch.tensor([0]), torch.tensor([[torch.nan]]))


def test_exact_evidence_aggregates_games_and_requires_complete_keys():
    m = importlib.import_module("pid_choice_exact")
    values = {"pub2|a": np.array([[1., 2.], [3., 4.]]), "pub2|b": np.array([[5., 6.], [7., 8.]])}
    np.testing.assert_array_equal(m.aggregate_evidence(values, [["pub2|a"], ["pub2|a", "pub2|b"]], 1), [[3, 4], [10, 12]])
    with pytest.raises(ValueError, match="missing"):
        m.aggregate_evidence(values, [["priv2|a"]], 1)


def test_exact_cache_requires_complete_heldout_keys_and_matching_pool(tmp_path, monkeypatch):
    m = importlib.import_module("pid_choice_exact")
    monkeypatch.setattr(m, "read_protocol", lambda root: {"protocol_id": "p", "run_id": "r"})
    monkeypatch.setattr(m, "seed_split", lambda *args: {"heldout": {"owner": ["pub2|a", "pub2|b"]}})
    m.atomic_json(tmp_path / "exact_arithmetic_pub2_s2_3.json", {"protocol_id": "p", "verdict": "passed"})
    out = tmp_path / "exact_scores"
    out.mkdir()
    meta = {"protocol_id": "p", "pool": "pub2", "seeds": [2, 3], "owners": ["owner"], "configs": [list(c) for c in m.CONFIGS]}
    m._write(out / "pub2_s2_3_p0000.npz", {"keys_s2": np.array(["pub2|a"]), "scores_s2": np.zeros((1, 6, 1))}, meta)
    with pytest.raises(ValueError, match="incomplete"):
        m.load_evidence(tmp_path, {"run_id": "r"}, "pub2", 2, ["owner"])
    m._write(out / "pub2_s2_3_p0001.npz", {"keys_s2": np.array(["pub2|b"]), "scores_s2": np.ones((1, 6, 1))}, meta)
    assert len(m.load_evidence(tmp_path, {"run_id": "r"}, "pub2", 2, ["owner"])) == 2
    bad = meta | {"pool": "priv2"}
    m.atomic_json(out / "pub2_s2_3_p0001.json", {"metadata": bad, "sha256": m.digest_file(out / "pub2_s2_3_p0001.npz")})
    with pytest.raises(ValueError, match="provenance"):
        m.load_evidence(tmp_path, {"run_id": "r"}, "pub2", 2, ["owner"])
