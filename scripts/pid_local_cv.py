#!/usr/bin/env python
"""Line B2: whole-game local-shape habits (aicup_pid.local) as a third fusion component, pre-specified vs the current fusion.

pid_cv.py's pools (make_pool, seed 0), parsed once. Per pool: opening + style as in production, plus LocalModel on the
candidates' games (tokens from ply --start on: displacement classes to White's and to Black's previous move, line, phase).
Fusion weights are cross-fitted on the other pools for both the two-component fusion and the three-component fusion3.
Pre-specified: fusion3 vs fusion (player bootstrap, both pools the same sign, overall >= 0.003 or n=5 >= 0.005); report only:
local vs style, local vs opening. Fold 0 is refused.

  python -I scripts/pid_local_cv.py --pools 1,2 3,4 --workers 12 --out results/local/local_cv.json
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
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, parse_all, score_questions  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.local import LocalModel  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
THREE = (*COMPONENTS, "local")
PRE = [("fusion3", "fusion")]
REPORT = [("local", "style"), ("local", "opening")]


def run_pool(games, folds_df, folds, a, cfg, log=print):
    t0 = time.time()
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed, pool_id="p" + "".join(map(str, folds)), order=a.order)
    if a.max_questions:
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
    log(f"pool {folds}: {info['players']} players, {len(cands)} candidate games, {len(qs)} questions; loading {len(keys)} games", flush=True)
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all(sgfs, a.workers)
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    q_keys = [list(k) for k in qs["game_keys"]]
    S = score_questions(opening, style, q_keys, moves, feats)
    loc = LocalModel(alpha=a.local_alpha, beta=a.local_beta, start=a.start).fit({p: [moves[k] for k in ks] for p, ks in by_player.items()})
    if loc.players != opening.players:
        raise ValueError("candidate order differs between the local and opening models")
    S["local"] = np.stack([loc.log_scores([moves[k] for k in ks]) for ks in q_keys])
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    log(f"pool {folds}: components in {time.time() - t0:.0f} s (local vocabulary {loc.vocabulary})", flush=True)
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "vocabulary": loc.vocabulary, "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--local-alpha", type=float, default=2.0)
    ap.add_argument("--local-beta", type=float, default=5.0)
    ap.add_argument("--start", type=int, default=12, help="first ply whose Black moves give local tokens (12 = after the opening model's window)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle", help="how a player's games are split: shuffle (default) or file = earliest games as the candidate's (drift stress test)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests only")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None, help="write the three-component fusion weights fitted on all pools (for predict_pid.py)")
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink}
        runs = [run_pool(games, folds_df, f, a, cfg) for f in specs]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    results, per_q_all, pids_all, n_all = [], {c: [] for c in (*THREE, "fusion", "fusion3")}, [], []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        weights = {}
        fused = {}
        for name, comps in (("fusion", COMPONENTS), ("fusion3", THREE)):
            if others:
                fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
                weights[name] = fus.w.tolist()
            else:
                fus = Fusion()
                fus.w = np.ones(len(comps))
                weights[name] = None
            fused[name] = fus.scores([r["S"][c] for c in comps])
        scores = {}
        for c in THREE:
            scores[c], per_q = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
            per_q_all[c].append(per_q)
        for name in ("fusion", "fusion3"):
            scores[name], per_q = evaluate(fused[name], r["players"], r["questions"], r["truth"])
            per_q_all[name].append(per_q)
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        n_all.append(r["questions"]["n_games"].to_numpy())
        results.append({"pool": r["info"], "scores": scores, "fusion_weights": weights["fusion"], "fusion3_weights": weights["fusion3"], "local_vocabulary": r["vocabulary"], "seconds": r["seconds"]})
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores)
              + f"  weights3 {np.round(weights['fusion3'], 3).tolist() if weights['fusion3'] else '(equal)'}", flush=True)
    pid, n_games = np.concatenate(pids_all), np.concatenate(n_all)
    comps = {}
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
        comps[group] = {}
        for x, z in pairs:
            qx, qz = np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z])
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]
            n5 = n_games == 5
            d.update({"per_pool_diff": per_pool, "diff_n5": float(np.mean((qx - qz)[n5])) if n5.any() else None,
                      "by_n": {str(n): float(np.mean((qx - qz)[n_games == n])) for n in sorted(set(n_games.tolist()))},
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d.get("diff_n5") or 0) >= MIN_DIFF_N5)))})
            comps[group][f"{x} vs {z}"] = d
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
                  + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}, "comparisons": comps, "cfg": cfg,
           "local_cfg": {"alpha": a.local_alpha, "beta": a.local_beta, "start": a.start}, "order": a.order, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    if a.fusion_out:
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in THREE], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": list(THREE), "weights": fus.w.tolist(), "cfg": {**cfg, "local_alpha": a.local_alpha, "local_beta": a.local_beta, "local_start": a.start},
               "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)), **pid_environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
