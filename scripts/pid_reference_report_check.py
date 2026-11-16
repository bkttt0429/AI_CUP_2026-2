#!/usr/bin/env python
"""Independent raw-score arithmetic check and labeled post-hoc rank-band diagnostic."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pid_reference_policy import read_protocol, LEDGER
from pid_choice_manifest import atomic_json, source_record, verify_source
from pid_choice_runtime import check


def arithmetic(scores):
    result, top = [], []
    for owner, rows in enumerate(scores):
        def measure(row):
            order = np.argsort(-row, kind="stable")
            rank = int(np.flatnonzero(order == owner)[0])
            return np.array([np.exp(-rank) if rank < 5 else 0., float(rank == 0)])
        v = np.mean([np.mean([measure(row) for row in rows], axis=0), measure(rows[:2].sum(0)), measure(rows.sum(0))], axis=0)
        result.append(v[0]); top.append(v[1])
    return np.array(result), float(np.mean(top))


def ci(v):
    ix = np.random.default_rng(20261116).integers(len(v), size=(5000, len(v)))
    return {"mean": float(np.mean(v)), "ci95": np.quantile(v[ix].mean(1), [.025, .975]).tolist()}


def verify(root):
    root = Path(root); m = read_protocol(root)
    report = json.loads((root / "refpolicy/report.json").read_text())
    assert report["protocol"] == m["id"]
    for ck in report["checkpoints"]:
        verify_source(ck)
        saved = torch.load(ck["path"], map_location="cpu", weights_only=True)
        assert saved["protocol"] == m["id"]
        assert all(torch.isfinite(t).all() for t in saved["state"].values())
    owners = sorted({g["owner"] for g in m["games"] if g["split"] == "test"})
    assert owners == report["test_owners"]
    ranks = [{g["key"].split(":")[0] for g in m["games"] if g["owner"] == p} for p in owners]
    assert all(len(r) == 1 for r in ranks)
    ranks = np.array([next(iter(r)) for r in ranks])
    with np.load(root / "refpolicy/test_scores.npz", allow_pickle=False) as f:
        scores = {k: f[k] for k in f.files}
    per, diagnostic = {}, {}
    for label, s in scores.items():
        assert s.shape == (50, 3, 50) and np.isfinite(s).all()
        rs, top1 = arithmetic(s); per[label] = rs
        base_label = label.split("_permuted_keys")[0].split("_zero")[0]
        run = report["runs"][base_label]
        metrics = run["permuted_keys"] if label.endswith("permuted_keys") else run["zero"] if label.endswith("zero") else run["metrics"]
        assert abs(rs.mean() - metrics["rank_score"]) < 1e-12
        assert abs(top1 - metrics["top1"]) < 1e-12
        own = s[np.arange(50), :, np.arange(50)]
        assert abs(own.mean() - (metrics["base_nll"] - metrics["move_nll"])) < 5e-7
        if label == base_label:
            wrong = s[np.arange(50), :, (np.arange(50) + 1) % 50]
            expected = run["own_minus_wrong_move_ll"]
            actual = ci((own - wrong).mean(1))
            assert np.allclose(actual["ci95"], expected["ci95"], atol=5e-7, rtol=0)
            values = []
            for i in range(50):
                mask = (ranks == ranks[i]) & (np.arange(50) != i)
                if mask.any():
                    values.append(float(own[i].mean() - s[i][:, mask].mean()))
            diagnostic[label] = {"eligible_players": len(values), "own_minus_mean_same_rank_wrong_ll": ci(np.array(values))}
    for seed in (17, 29):
        a, u = f"attention_s{seed}", f"uniform_s{seed}"
        assert np.allclose(ci(per[a] - per[u])["ci95"], report["attention_minus_uniform"][str(seed)]["ci95"], atol=1e-12)
        for abl in ("permuted_keys", "zero"):
            assert np.allclose(ci(per[a] - per[a + "_" + abl])["ci95"], report["runs"][a]["minus_" + abl]["ci95"], atol=1e-12)
    out = {"protocol": m["id"], "arithmetic": "passed", "score_arrays": len(scores), "posthoc_same_rank": diagnostic,
           "scope": "Same-rank diagnostic chosen after primary results; descriptive, no gate change. Rank labels are coarse, not exact strength matching; openings uncontrolled.",
           "source": source_record(__file__), "runtime": check(root, LEDGER)}
    atomic_json(root / "refpolicy/report_check.json", out)
    atomic_json(ROOT / "results/final/reference_policy_check_20261116.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "runtime"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("production Slurm required")
    verify(ap.parse_args().root)
