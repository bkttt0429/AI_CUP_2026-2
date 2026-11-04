#!/usr/bin/env python
"""Line B6: the raw board orientation (aicup_pid.orient) as a fourth fusion component, pre-specified vs fusion3.

pid_cv.py's pools (make_pool, seed 0), parsed once. Per pool: opening + style + local as in fusion3, plus OrientModel on
the candidates' games (tokens: the symmetry index canonical() picks for the first 12 moves, and the raw first Black move),
and the HD component of line B5 for the report-only five-component fusion. Fusion weights are cross-fitted on the other
pools. Pre-specified: fusion3o (opening, style, local, orient) vs fusion3; report only: orient vs style, orient vs opening,
fusion5 (fusion3o + hd) vs fusion3o. Fold 0 is refused. Per pool the per-player modal share of the raw first move is
printed (the official candidates: median 0.98).

  python -I scripts/pid_orient_cv.py --pools 1,2 3,4 --workers 12 --out results/orient/orient_cv.json --fusion-out results/pid_fusion3o.json
"""
import argparse
import sys
import time
from collections import Counter
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
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, fit_hd, fit_local, fit_opening_raw, fit_orient, parse_all, score_components  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.orient import orient_tokens  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
THREE = (*COMPONENTS, "local")
FOUR = (*THREE, "orient")
FIVE = (*FOUR, "hd")
FUSIONS = (("fusion", COMPONENTS), ("fusion3", THREE), ("fusion3o", FOUR), ("fusion5", FIVE))
PRE = [("fusion3o", "fusion3")]
REPORT = [("orient", "style"), ("orient", "opening"), ("fusion5", "fusion3o")]


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
    local, hd, orient = fit_local(by_player, moves, cfg), fit_hd(by_player, sgfs, cfg), fit_orient(by_player, moves, cfg)
    q_keys = [list(k) for k in qs["game_keys"]]
    S = score_components(FIVE, opening, style, q_keys, moves, feats, local, hd, sgfs, orient)
    if a.compare_raw:  # a second orientation model with another number of raw moves, as its own component
        orient_b = fit_orient(by_player, moves, {**cfg, "orient_raw": a.compare_raw})
        S["orient_b"] = score_components(("orient",), opening, style, q_keys, moves, feats, orient=orient_b)["orient"]
    if a.opening_raw:  # the opening back-off model on raw coordinates
        S["opening_raw"] = score_components(("opening_raw",), opening, style, q_keys, moves, feats, opening_raw=fit_opening_raw(by_player, moves, cfg))["opening_raw"]
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    modal = []
    for ks in by_player.values():
        firsts = Counter(t[1] for k in ks for t in orient_tokens(moves[k], a.orient_depth) if t[0] == "first")
        modal.append(firsts.most_common(1)[0][1] / max(sum(firsts.values()), 1) if firsts else 0.0)
    modal = np.array(modal)
    log(f"pool {folds}: components in {time.time() - t0:.0f} s; orientation vocabulary {orient.vocabulary}; per-player modal share of the raw first move: "
        f"median {np.median(modal):.2f}, >= 0.9 for {int((modal >= 0.9).sum())} / {len(modal)} players", flush=True)
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "orient_vocabulary": orient.vocabulary,
            "first_move_modal_share": {"median": float(np.median(modal)), "ge_0.9": int((modal >= 0.9).sum())}, "seconds": time.time() - t0}


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
    ap.add_argument("--start", type=int, default=12, help="first ply whose Black moves give local tokens")
    ap.add_argument("--hd-alpha", type=float, default=2.0)
    ap.add_argument("--hd-beta", type=float, default=5.0)
    ap.add_argument("--orient-alpha", type=float, default=2.0, help="shrinkage of a candidate's orientation token frequencies towards the pool's")
    ap.add_argument("--orient-beta", type=float, default=5.0)
    ap.add_argument("--orient-depth", type=int, default=12, help="moves canonical() looks at for the symmetry index")
    ap.add_argument("--orient-raw", type=int, default=1, help="Black's raw moves used as orientation tokens (1 = first move only)")
    ap.add_argument("--compare-raw", type=int, default=0, help="also fit an orientation model with this many raw moves; then the pre-specified comparison is fusion3o_b (with it) vs fusion3o")
    ap.add_argument("--opening-raw", action="store_true", help="also fit the opening model on raw coordinates: arms opening_raw, fusion3r (fusion3 + it), fusion3or (fusion3o + it); pre-specified fusion3or vs fusion3o")
    ap.add_argument("--fusion3or-out", default=None, help="with --opening-raw: write the fusion3or weights fitted on all pools")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests only")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None, help="write the fusion3o weights fitted on all pools (for predict_pid.py)")
    ap.add_argument("--fusion5-out", default=None, help="also write the five-component weights (report only)")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink, "local_alpha": a.local_alpha, "local_beta": a.local_beta, "local_start": a.start,
           "hd_alpha": a.hd_alpha, "hd_beta": a.hd_beta, "orient_alpha": a.orient_alpha, "orient_beta": a.orient_beta, "orient_depth": a.orient_depth, "orient_raw": a.orient_raw}
    fusions, pre, report, comps_all = list(FUSIONS), list(PRE), list(REPORT), list(FIVE)
    if a.compare_raw:
        comps_all.append("orient_b")
        fusions.append(("fusion3o_b", (*THREE, "orient_b")))
        pre, report = [("fusion3o_b", "fusion3o")], [("orient_b", "orient"), *REPORT]
    if a.opening_raw:
        comps_all.append("opening_raw")
        fusions += [("fusion3r", (*THREE, "opening_raw")), ("fusion3or", (*FOUR, "opening_raw"))]
        pre = [("fusion3or", "fusion3o"), *([] if not a.compare_raw else pre)]
        report = [("opening_raw", "opening"), ("fusion3r", "fusion3"), ("fusion3or", "fusion3r"), *report]
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        runs = [run_pool(games, folds_df, f, a, cfg) for f in specs]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    arms = (*comps_all, *[name for name, _ in fusions])
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        weights, fused = {}, {}
        for name, comps in fusions:
            if others:
                fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
                weights[name] = fus.w.tolist()
            else:
                fus = Fusion()
                fus.w = np.ones(len(comps))
                weights[name] = None
            fused[name] = fus.scores([r["S"][c] for c in comps])
        scores = {}
        for c in comps_all:
            scores[c], per_q = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
            per_q_all[c].append(per_q)
        for name, _ in fusions:
            scores[name], per_q = evaluate(fused[name], r["players"], r["questions"], r["truth"])
            per_q_all[name].append(per_q)
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        n_all.append(r["questions"]["n_games"].to_numpy())
        results.append({"pool": r["info"], "scores": scores, **{f"{name}_weights": weights[name] for name, _ in fusions}, "orient_vocabulary": r["orient_vocabulary"],
                        "first_move_modal_share": r["first_move_modal_share"], "seconds": r["seconds"]})
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores)
              + f"  weights3o {np.round(weights['fusion3o'], 3).tolist() if weights['fusion3o'] else '(equal)'}", flush=True)
    pid, n_games = np.concatenate(pids_all), np.concatenate(n_all)
    comps = {}
    for group, pairs in (("pre_specified", pre), ("report_only", report)):
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
           "compare_raw": a.compare_raw, "opening_raw": bool(a.opening_raw), "order": a.order, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    for path, comps_ in ((a.fusion_out, FOUR), (a.fusion5_out, FIVE), (a.fusion3or_out, (*FOUR, "opening_raw"))):
        if path:
            src = [("orient_b" if (c == "orient" and a.compare_raw) else c) for c in comps_]  # with --compare-raw the documents carry the compared model
            fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in src], np.concatenate([r["truth"] for r in runs]))
            doc_cfg = {**cfg, "orient_raw": a.compare_raw} if a.compare_raw else cfg
            doc = {"components": list(comps_), "weights": fus.w.tolist(), "cfg": doc_cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)), **pid_environment("cpu")}
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(to_json(doc, indent=1) + "\n")
            print(f"wrote {path}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
