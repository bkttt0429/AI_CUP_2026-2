#!/usr/bin/env python
"""Line B7: reference-player affinity (aicup_pid.ref) as an extra fusion component, pre-specified vs fusion3o.

Pools from one fold pair (make_pool, seed 0); reference players from the OTHER fold pair (>= --ref-min-black Black games,
up to --ref-cap games each), so no reference is a pool player. Per pool: opening, style, local, orient (N=1) as in
fusion3o, plus the ref component (cosine of mean z-scored log-likelihood profiles over the references). Fusion weights
cross-fitted on the other pool. Pre-specified: fusion3o_ref vs fusion3o; report only: ref vs style, ref vs opening,
fusion3_ref vs fusion3. Fold 0 is refused.

  python -I scripts/pid_ref_cv.py --pools 1,2 3,4 --workers 12 --out results/ref/ref_cv.json --fusion-out results/pid_fusion3o_ref.json
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
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, fit_local, fit_orient, fit_ref, parse_all, score_components, score_ref  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.ref import reference_players  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
THREE = (*COMPONENTS, "local")
FOUR = (*THREE, "orient")
FIVE = (*FOUR, "ref")
FUSIONS = (("fusion", COMPONENTS), ("fusion3", THREE), ("fusion3o", FOUR), ("fusion3_ref", (*THREE, "ref")), ("fusion3o_ref", FIVE))
PRE = [("fusion3o_ref", "fusion3o")]
REPORT = [("ref", "style"), ("ref", "opening"), ("fusion3_ref", "fusion3")]


def run_pool(games, folds_df, folds, ref_folds, a, cfg, log=print):
    t0 = time.time()
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed, pool_id="p" + "".join(map(str, folds)), order=a.order)
    if a.max_questions:
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)
    refs = reference_players(games, folds_df, ref_folds, min_black=a.ref_min_black, cap=a.ref_cap, seed=a.seed, exclude=set(cands["player_id"]))
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks} | {k for ks in refs.values() for k in ks})
    log(f"pool {folds}: {info['players']} players, {len(cands)} candidate games, {len(qs)} questions; references from folds {ref_folds}: {len(refs)} players, "
        f"{sum(len(v) for v in refs.values())} games; loading {len(keys)} games", flush=True)
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all({k: sgfs[k] for k in set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks}}, a.workers)
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    local, orient = fit_local(by_player, moves, cfg), fit_orient(by_player, moves, cfg)
    q_keys = [list(k) for k in qs["game_keys"]]
    t1 = time.time()
    ref_model = fit_ref(refs, moves, cfg)
    S_ref = score_ref(ref_model, by_player, q_keys, moves)
    log(f"pool {folds}: reference model {len(ref_model.players)} players x {ref_model.logp.shape[1]} tokens, profiles and cosines in {time.time() - t1:.0f} s", flush=True)
    S = score_components(FIVE, opening, style, q_keys, moves, feats, local=local, orient=orient, ref=S_ref)
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    log(f"pool {folds}: components in {time.time() - t0:.0f} s", flush=True)
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "n_refs": len(ref_model.players), "ref_tokens": int(ref_model.logp.shape[1]),
            "by_player": by_player, "q_keys": q_keys, "seconds": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="exactly two fold pairs: each pool's references come from the other")
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--local-alpha", type=float, default=2.0)
    ap.add_argument("--local-beta", type=float, default=5.0)
    ap.add_argument("--start", type=int, default=12)
    ap.add_argument("--orient-alpha", type=float, default=2.0)
    ap.add_argument("--orient-beta", type=float, default=5.0)
    ap.add_argument("--orient-depth", type=int, default=12)
    ap.add_argument("--orient-raw", type=int, default=1)
    ap.add_argument("--ref-min-black", type=int, default=30, help="a reference player needs this many Black games")
    ap.add_argument("--ref-cap", type=int, default=40, help="games per reference player")
    ap.add_argument("--ref-orient-raw", type=int, default=2, help="raw moves in the reference orientation tokens")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests only")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None, help="write the fusion3o_ref weights fitted on all pools (for predict_pid.py)")
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink, "local_alpha": a.local_alpha, "local_beta": a.local_beta, "local_start": a.start,
           "orient_alpha": a.orient_alpha, "orient_beta": a.orient_beta, "orient_depth": a.orient_depth, "orient_raw": a.orient_raw,
           "ref_alpha": a.alpha, "ref_beta": a.beta, "ref_min_black": a.ref_min_black, "ref_cap": a.ref_cap, "ref_orient_raw": a.ref_orient_raw, "ref_seed": a.seed}
    try:
        specs = parse_pools(a.pools)
        if len(specs) != 2:
            raise ValueError("give exactly two fold pairs")
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        runs = [run_pool(games, folds_df, f, specs[1 - i], a, cfg) for i, f in enumerate(specs)]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    arms = (*FIVE, *[name for name, _ in FUSIONS])
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        weights, fused = {}, {}
        for name, comps in FUSIONS:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            weights[name] = fus.w.tolist()
            fused[name] = fus.scores([r["S"][c] for c in comps])
        scores = {}
        for c in FIVE:
            scores[c], per_q = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
            per_q_all[c].append(per_q)
        for name, _ in FUSIONS:
            scores[name], per_q = evaluate(fused[name], r["players"], r["questions"], r["truth"])
            per_q_all[name].append(per_q)
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        n_all.append(r["questions"]["n_games"].to_numpy())
        results.append({"pool": r["info"], "scores": scores, **{f"{name}_weights": weights[name] for name, _ in FUSIONS}, "n_refs": r["n_refs"], "ref_tokens": r["ref_tokens"], "seconds": r["seconds"]})
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores) + f"  weights3o_ref {np.round(weights['fusion3o_ref'], 3).tolist()}", flush=True)
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
           "order": a.order, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    if a.fusion_out:
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in FIVE], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": list(FIVE), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),
               "ref_folds_official": [1, 2, 3, 4], **pid_environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
