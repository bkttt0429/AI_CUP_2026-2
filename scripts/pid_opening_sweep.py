#!/usr/bin/env python
"""Line B1: opening-model hyper-parameters, one change at a time, each pre-specified against the current fusion.

The pools are pid_cv.py's (make_pool: up to 400 players with >= 105 Black games, 100 candidate games each, the rest cut
into 5/10/15/20-game questions, seed 0), parsed once per pool. The style model and its scores are fitted once; for the
base configuration and every --grid entry the opening model is refitted (depth, alpha, beta, alpha_full) and fused with
the style scores using weights fitted on the other pools (cross-fit, as pid_cv.py). Pre-specified: every configuration vs
base on the fusion's per-question score -- player bootstrap, both pools the same sign, overall >= 0.003 or n=5 >= 0.005.
The opening model alone is reported too. Fold 0 is refused.

  python -I scripts/pid_opening_sweep.py --pools 1,2 3,4 --workers 12 --out results/opening/opening_sweep.json
"""
import argparse
import json
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
from aicup_pid.components import COMPONENTS, features_all, parse_all  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.opening import OpeningModel  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from aicup_pid.style import StyleModel  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
# the pre-specified grid: one parameter per entry, relative to the base (depth 12, alpha 2, beta 5, alpha_full 1)
GRID = [{"name": "depth8", "depth": 8}, {"name": "depth16", "depth": 16}, {"name": "depth20", "depth": 20},
        {"name": "alpha1", "alpha": 1.0}, {"name": "alpha4", "alpha": 4.0},
        {"name": "beta2", "beta": 2.0}, {"name": "beta10", "beta": 10.0},
        {"name": "full0.5", "alpha_full": 0.5}, {"name": "full2", "alpha_full": 2.0}, {"name": "full_off", "alpha_full": 1000.0}]


def prepare_pool(games, folds_df, folds, a, log=print):
    t0 = time.time()
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed, pool_id="p" + "".join(map(str, folds)))
    if a.max_questions:
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
    log(f"pool {folds}: {info['players']} players, {len(cands)} candidate games, {len(qs)} questions; loading {len(keys)} games", flush=True)
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all(sgfs, a.workers)
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    style = StyleModel(shrink=a.shrink).fit({p: np.stack([feats[k] for k in ks]) for p, ks in by_player.items()})
    q_keys = [list(k) for k in qs["game_keys"]]
    S_style = np.stack([style.log_scores(np.stack([feats[k] for k in ks])) for ks in q_keys])
    index = {p: i for i, p in enumerate(style.players)}
    truth = np.array([index[p] for p in qs["player_id"]])
    log(f"pool {folds}: parsed and style scored in {time.time() - t0:.0f} s", flush=True)
    return {"info": info, "players": style.players, "questions": qs, "q_keys": q_keys, "by_player": by_player, "moves": moves, "S_style": S_style, "truth": truth}


def opening_scores(pool, cfg):
    model = OpeningModel(depth=cfg["depth"], alpha=cfg["alpha"], beta=cfg["beta"], alpha_full=cfg["alpha_full"])
    model.fit({p: [pool["moves"][k] for k in ks] for p, ks in pool["by_player"].items()})
    if model.players != pool["players"]:
        raise ValueError("candidate order differs between the opening and style models")
    return np.stack([model.log_scores([pool["moves"][k] for k in ks]) for ks in pool["q_keys"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--alpha-full", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--grid", default=None, help="JSON list of {name, depth|alpha|beta|alpha_full} overrides (default: the pre-specified GRID)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests only")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    base = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "alpha_full": a.alpha_full, "shrink": a.shrink}
    grid = json.loads(a.grid) if a.grid else GRID
    configs = [{"name": "base", **base}] + [{**base, **g} for g in grid]
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        if len({c["name"] for c in configs}) != len(configs) or any(set(c) - {"name", *base} for c in configs):
            raise ValueError("grid entries need distinct names and only depth / alpha / beta / alpha_full")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        prepared = [prepare_pool(games, folds_df, f, a) for f in specs]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    pid = np.concatenate([p["questions"]["player_id"].to_numpy(dtype=str) for p in prepared])
    n_games = np.concatenate([p["questions"]["n_games"].to_numpy() for p in prepared])
    results, per_q = [], {}
    for c in configs:
        cfg = {k: c[k] for k in base}
        S_open = [opening_scores(p, cfg) for p in prepared]
        entry = {"name": c["name"], "cfg": cfg, "pools": []}
        q_f, q_o = [], []
        for i, p in enumerate(prepared):
            others = [j for j in range(len(prepared)) if j != i]
            if others:
                fus = Fusion().fit([np.vstack([S_open[j] for j in others]), np.vstack([prepared[j]["S_style"] for j in others])],
                                   np.concatenate([prepared[j]["truth"] for j in others]))
                weights = fus.w.tolist()
            else:
                fus = Fusion()
                fus.w = np.ones(2)
                weights = None
            S_f = fus.scores([S_open[i], p["S_style"]])
            s_o, pq_o = evaluate(S_open[i], p["players"], p["questions"], p["truth"])
            s_f, pq_f = evaluate(S_f, p["players"], p["questions"], p["truth"])
            q_f.append(pq_f)
            q_o.append(pq_o)
            entry["pools"].append({"folds": p["info"]["folds"], "scores": {"opening": s_o, "fusion": s_f}, "fusion_weights": weights})
        per_q[c["name"]] = {"fusion": np.concatenate(q_f), "opening": np.concatenate(q_o), "per_pool": q_f}
        entry["overall_score"] = {"opening": float(per_q[c["name"]]["opening"].mean()), "fusion": float(per_q[c["name"]]["fusion"].mean())}
        results.append(entry)
        print(f"{c['name']}: opening {entry['overall_score']['opening']:.4f}  fusion {entry['overall_score']['fusion']:.4f}  "
              + " ".join(f"pool{p['folds']} {p['scores']['fusion']['score']:.4f}" for p in entry["pools"]), flush=True)
    comps = {}
    for c in configs[1:]:
        x, z = per_q[c["name"]], per_q["base"]
        d = paired_player_bootstrap(x["fusion"], z["fusion"], pid, n=a.n_boot)
        per_pool = [float(np.mean(px - pz)) for px, pz in zip(x["per_pool"], z["per_pool"])]
        n5 = n_games == 5
        d["diff_n5"] = float(np.mean((x["fusion"] - z["fusion"])[n5])) if n5.any() else None
        d.update({"per_pool_diff": per_pool,
                  "opening_alone_diff": float(np.mean(x["opening"] - z["opening"])),
                  "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})
        comps[f"{c['name']} vs base"] = d
        print(f"[pre_specified] {c['name']} vs base: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
              + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"base": results[0], "configs": results, "comparisons": {"pre_specified": comps}, "pools": [p["info"] for p in prepared], "questions": int(len(pid)),
           "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
