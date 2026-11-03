#!/usr/bin/env python
"""Transductive self-training on one-question-per-player pools (robustness line), pre-specified against fusion3 + assignment.

The pools and the cross-fit fusion weights are pid_assign_cv.py's (one question per player, seeds, control pool), with
the three components (opening, style, local). Arms per pool: assign3 = fusion3 + Hungarian Top-1 (the current best);
self3 = one round of aicup_pid.transduct.self_train on top of it (confident questions, margin >= --tau, lend their games to
the matched candidate; models refitted; everything rescored; assignment again). Pre-specified: self3 vs assign3 (player
bootstrap, both fold sets positive, overall >= 0.003 or n=5 >= 0.005). Report only: --tau-report, two rounds, and the
precision of the confident set (how many confident answers were wrong). Fold 0 is refused.

  python -I scripts/pid_selftrain_cv.py --pools 1,2 3,4 --seeds 0 1 2 3 4 --workers 12 --out results/selftrain/selftrain_cv.json
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
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import pools, transduct  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_assign_cv import build_group, score_arm  # noqa: E402
from pid_cv import parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
THREE = transduct.THREE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--per-size", type=int, default=100)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--local-alpha", type=float, default=2.0)
    ap.add_argument("--local-beta", type=float, default=5.0)
    ap.add_argument("--local-start", type=int, default=12)
    ap.add_argument("--tau", type=float, default=3.0, help="margin threshold of the main arm")
    ap.add_argument("--tau-report", type=float, default=6.0)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.control, a.control_questions = False, 0
    t0 = time.time()
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink, "local_alpha": a.local_alpha, "local_beta": a.local_beta, "local_start": a.local_start}
        groups = [build_group(games, folds_df, f, a, cfg) for f in specs]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    arms = ("assign3", "self3", "self3_report", "self3_two_rounds", "self3_noassign")
    results, per_q = [], {arm: [] for arm in arms}
    pids_all, n_all, group_of = [], [], []
    for i, runs in enumerate(groups):
        others = [r for j, g in enumerate(groups) if j != i for r in g]
        if others:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in THREE], np.concatenate([o["truth"] for o in others]))
            weights = fus.w.tolist()
        else:
            weights = [1.0, 1.0, 1.0]
        for r in runs:
            t1 = time.time()
            S_f = transduct.fuse(r["S"], weights)
            out_idx = {"assign3": assign_topk(S_f, pools.TOP_K)}
            infos = {}
            for arm, kw in (("self3", {"tau": a.tau, "rounds": 1, "assign": True}), ("self3_report", {"tau": a.tau_report, "rounds": 1, "assign": True}),
                            ("self3_two_rounds", {"tau": a.tau, "rounds": 2, "assign": True}), ("self3_noassign", {"tau": a.tau, "rounds": 1, "assign": False})):
                out_idx[arm], _, infos[arm] = transduct.self_train(r["by_player"], r["moves"], r["feats"], cfg, r["q_keys"], r["S"], weights, **kw)
            scores, rep = {}, {}
            for arm in arms:
                scores[arm], q = score_arm(out_idx[arm], r["players"], r["questions"], r["truth"])
                per_q[arm].append(q)
            # precision of the confident set of the main arm (first-round answers that lent their games)
            first = assign_topk(S_f, pools.TOP_K)[:, 0]
            conf = np.flatnonzero(transduct.margins_of(S_f, first) >= a.tau)
            wrong = int(sum(first[q] != r["truth"][q] for q in conf))
            rep = {"confident": int(len(conf)), "confident_wrong": wrong, "precision": float(1 - wrong / len(conf)) if len(conf) else None,
                   "rounds": infos["self3"]["rounds"], "report_confident": infos["self3_report"]["rounds"][0]["confident"]}
            pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
            n_all.append(r["questions"]["n_games"].to_numpy())
            group_of.append(np.full(len(r["questions"]), i))
            results.append({"pool": r["info"], "scores": scores, "fusion3_weights": weights, "self_training": rep, "seconds": time.time() - t1})
            print(f"pool {r['info']['folds']} seed {r['info']['seed']}: " + "  ".join(f"{arm} {scores[arm]['score']:.4f}" for arm in arms)
                  + f"  confident {rep['confident']}/{len(first)} (wrong {wrong})  {time.time() - t1:.0f} s", flush=True)
    pid, n_games, grp = np.concatenate(pids_all), np.concatenate(n_all), np.concatenate(group_of)
    comps = {"pre_specified": {}, "report_only": {}}
    base = np.concatenate(per_q["assign3"])
    for group, arm in (("pre_specified", "self3"), ("report_only", "self3_report"), ("report_only", "self3_two_rounds"), ("report_only", "self3_noassign")):
        qa = np.concatenate(per_q[arm])
        d = paired_player_bootstrap(qa, base, pid, n=a.n_boot)
        per_group = [float(np.mean((qa - base)[grp == g])) for g in range(len(groups))]
        n5 = n_games == 5
        d.update({"per_pool_diff": per_group, "diff_n5": float(np.mean((qa - base)[n5])) if n5.any() else None,
                  "by_n": {str(n): float(np.mean((qa - base)[n_games == n])) for n in sorted(set(n_games.tolist()))},
                  "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_group) and (d["diff"] >= MIN_DIFF or ((d.get("diff_n5") or 0) >= MIN_DIFF_N5)))})
        comps[group][f"{arm} vs assign3"] = d
        print(f"[{group}] {arm} vs assign3: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  fold sets " + " ".join(f"{v:+.4f}" for v in per_group)
              + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {arm: float(np.concatenate(per_q[arm]).mean()) for arm in arms}, "comparisons": comps, "cfg": cfg,
           "tau": a.tau, "tau_report": a.tau_report, "seeds": a.seeds, "n_players": a.n_players, "cand_games": a.cand_games, "per_size": a.per_size, "order": a.order,
           "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
