"""Recalculate the local-choice decision directly from saved per-game scores."""
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_local_choice_data import DEST, LEDGER, ROOT, RUN, atomic_json, check, read_data, source_record, verify_source


def rank_terms(scores):
    n = len(scores)
    per_size = []
    hits = []
    for count in (1, 2, 3):
        matrix = scores if count == 1 else scores[:, :count].sum(1)[:, None]
        order = np.argsort(-matrix, axis=2, kind="stable")
        r = (order == np.arange(n)[:, None, None]).argmax(-1)
        per_size.append(np.where(r < 5, np.exp(-r.astype(float)), 0).mean(1))
        hits.append((r == 0).mean(1))
    return np.stack(per_size, axis=1), np.stack(hits, axis=1)


def verify():
    out = DEST / "simple_ko_v2"
    r = json.loads((out / "report.json").read_text())
    p = json.loads((out / "protocol.json").read_text())
    data = read_data()
    if p["id"] != r["protocol"] or p["data_id"] != data["id"] or r["data_id"] != data["id"]:
        raise ValueError("inconsistent protocol ids")
    for record in [*p["code"], p["p0"], *r["checkpoints"]]:
        verify_source(record)
    near = []
    rank = {}
    do_gate = []
    with np.load(out / "scores.npz", allow_pickle=False) as raw:
        labels = sorted({g["owner"] for g in data["games"] if g["split"] == "test"})
        if raw["owners"].tolist() != labels or len(labels) != 100:
            raise ValueError("test player order or size")
        n = len(labels)
        ix = np.arange(n)
        rivals = raw["rivals"]
        if not ((rivals >= 0) & (rivals < n) & (rivals != ix)).all():
            raise ValueError("invalid preselected rival")
        draws = np.random.default_rng(20261116).integers(n, size=(5000, n))
        for name, run in r["runs"].items():
            scores = raw[name]
            if scores.shape != (100, 3, 100) or not np.isfinite(scores).all():
                raise ValueError(f"invalid scores for {name}")
            values, hits = rank_terms(scores)
            rank[name] = values.mean(1)
            np.testing.assert_allclose(values.mean(), run["metrics"]["rank_score"], atol=1e-8)
            np.testing.assert_allclose(hits.mean(), run["metrics"]["top1"], atol=1e-8)
            own = scores[ix, :, ix]
            np.testing.assert_allclose(-(own + raw["base_ll"]).mean(), run["metrics"]["move_nll"], atol=1e-6)
            gap = (own - scores[ix, :, rivals]).mean(1)
            np.testing.assert_allclose(gap.mean(dtype=float), run["own_minus_near_same_rank_ll"]["mean"], atol=1e-8)
            if "_zero" in name:
                assert np.abs(scores).max() < 2e-6
                continue
            if "_permute" not in name:
                history = json.loads((out / f"{name}_dev.json").read_text())
                best = min(history, key=lambda d: d["objective"])["step"]
                assert best == run["checkpoint_step"]
                diag = run["metrics"]["own_retrieval_by_32_ply_band"]
                assert sum(v["positions"] for v in diag.values()) == n * 3 * 32
                assert all(.999 <= v["effective_tokens_per_reference_game"] <= 32.001 and 0 <= v["null"] <= 1 for v in diag.values())
                if name.startswith("L"):
                    near.append({"arm": name, "pair_wins": int((own > scores[ix, :, rivals]).sum()), "games": own.size})
        for seed in (17, 29):
            for comp, other in [("L-G", f"G_{seed}"), ("L-U", f"U_{seed}"), ("L-A", f"A_{seed}"), ("L-permuted", f"L_{seed}_permute")]:
                d = rank[f"L_{seed}"] - rank[other]
                lo, hi = np.percentile(d[draws].mean(1), [2.5, 97.5])
                record = r["comparisons"][str(seed)][comp]["equal_size_rank"]
                np.testing.assert_allclose([d.mean(), lo, hi], [record["mean"], *record["ci95"]], atol=1e-8)
                if comp != "L-A":
                    do_gate.append(lo > 0)
            assert r["runs"][f"G_{seed}"]["parameters"] == r["runs"][f"L_{seed}"]["parameters"] == r["runs"][f"A_{seed}"]["parameters"]
    personalized = all(r["runs"][f"L_{s}"]["own_minus_cyclic_ll"]["ci95"][0] > 0 for s in (17, 29))
    want = "local_retrieval_supported" if all(do_gate) else "personalization_only" if personalized else "mechanism_not_established"
    if want != r["verdict"]:
        raise ValueError("verdict does not match registered rule")
    result = {"verified": True, "protocol": p["id"], "data_id": data["id"], "verdict": want,
              "reference_rival_pair_wins": near, "scores": source_record(out / "scores.npz"), "job": os.environ["SLURM_JOB_ID"]}
    atomic_json(out / "checked.json", result)
    atomic_json(ROOT / "results/final/local_choice_20261118.json", r)
    atomic_json(ROOT / "results/final/local_choice_check_20261118.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("use production Slurm")
    verify()
    check(RUN, LEDGER)
