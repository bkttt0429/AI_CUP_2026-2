#!/usr/bin/env python
"""Line B4 (trunk variant): the trunk-style component (aicup_pid.trunkstyle) on top of fusion3o_ref, pre-specified.

The pools and the five fusion3o_ref components come from pid_ref_cv.run_pool (make_pool seed 0, references from the other
fold pair); the trunk rows of every pool game come from --store (scripts/pid_trunk_extract.py --keys ...). Fusion weights
are cross-fitted on the other pool. Pre-specified: fusion3o_ref_trunk vs fusion3o_ref; report only: trunk vs style,
trunk vs opening, fusion3_trunk vs fusion3, and the trunk column subsets (--feats) alone. Fold 0 is refused.

  python -I scripts/pid_trunk_cv.py --store data/trunk/train --pools 1,2 3,4 --workers 12 --out results/trunk/trunk_cv.json \\
      --fusion-out results/pid_fusion3o_ref_trunk.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from aicup_pid.trunkstyle import COLUMNS, fit_trunk, load_store, score_trunk  # noqa: E402
import pid_ref_cv  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
BASE = pid_ref_cv.FIVE  # opening, style, local, orient, ref
FUSIONS = (("fusion3", ("opening", "style", "local")), ("fusion3o_ref", BASE), ("fusion3_trunk", ("opening", "style", "local", "trunk")),
           ("fusion3o_ref_trunk", (*BASE, "trunk")))
PRE = [("fusion3o_ref_trunk", "fusion3o_ref")]
REPORT = [("trunk", "style"), ("trunk", "opening"), ("fusion3_trunk", "fusion3")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--feats", default="all", choices=list(COLUMNS))
    ap.add_argument("--shrink", type=float, default=0.1, help="covariance shrinkage of the trunk LDA")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()

    defaults = argparse.Namespace(n_players=pools.N_CANDIDATES, cand_games=pools.CAND_GAMES, seed=0, order="shuffle", max_questions=0, workers=a.workers,
                                  ref_min_black=30, ref_cap=40)
    cfg = {"depth": 12, "alpha": 2.0, "beta": 5.0, "shrink": 0.1, "local_alpha": 2.0, "local_beta": 5.0, "local_start": 12, "orient_alpha": 2.0,
           "orient_beta": 5.0, "orient_depth": 12, "orient_raw": 1, "ref_alpha": 2.0, "ref_beta": 5.0, "ref_min_black": 30, "ref_cap": 40, "ref_orient_raw": 2,
           "ref_seed": 0, "trunk_feats": a.feats, "trunk_shrink": a.shrink}
    specs = parse_pools(a.pools)
    if len(specs) != 2 or any(0 in f for f in specs):
        print("give exactly two fold pairs without fold 0", file=sys.stderr)
        return 2
    store = load_store(*a.store)
    print(f"trunk store: {len(store)} games", flush=True)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    runs = []
    for i, f in enumerate(specs):
        r = pid_ref_cv.run_pool(games, folds_df, f, specs[1 - i], defaults, cfg)
        t1 = time.time()
        model = fit_trunk(r["by_player"], store, cfg)
        if model.players != r["players"]:
            raise ValueError("candidate order differs")
        r["S"]["trunk"] = score_trunk(model, r["q_keys"], store)
        print(f"pool {f}: trunk component in {time.time() - t1:.0f} s", flush=True)
        runs.append(r)
    comps_all = (*BASE, "trunk")
    arms = (*comps_all, *[n for n, _ in FUSIONS])
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        weights, fused = {}, {}
        for name, comps in FUSIONS:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            weights[name], fused[name] = fus.w.tolist(), fus.scores([r["S"][c] for c in comps])
        scores = {}
        for c in comps_all:
            scores[c], per_q = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
            per_q_all[c].append(per_q)
        for name, _ in FUSIONS:
            scores[name], per_q = evaluate(fused[name], r["players"], r["questions"], r["truth"])
            per_q_all[name].append(per_q)
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        n_all.append(r["questions"]["n_games"].to_numpy())
        results.append({"pool": r["info"], "scores": scores, **{f"{n}_weights": weights[n] for n, _ in FUSIONS}})
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores) + f"  weights {np.round(weights['fusion3o_ref_trunk'], 3).tolist()}", flush=True)
    pid, n_games = np.concatenate(pids_all), np.concatenate(n_all)
    comps = {}
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
        comps[group] = {}
        for x, z in pairs:
            qx, qz = np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z])
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]
            n5 = n_games == 5
            d["diff_n5"] = float(np.mean((qx - qz)[n5])) if n5.any() else None
            d.update({"per_pool_diff": per_pool, "by_n": {str(n): float(np.mean((qx - qz)[n_games == n])) for n in sorted(set(n_games.tolist()))},
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})
            comps[group][f"{x} vs {z}"] = d
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
                  + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}, "comparisons": comps, "cfg": cfg,
           "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    if a.fusion_out:
        comps_ = (*BASE, "trunk")
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps_], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": list(comps_), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),
               "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", **pid_environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
