#!/usr/bin/env python
"""Simulated-pool CV of the player-identification ladder's CPU components (docs/ARCHITECTURE.md sections 2-3).

Each --pools entry names the folds of one simulated pool (aicup_pid.pools.make_pool: up to 400 players with >= 105 Black
games, 100 candidate games each, the rest cut into 5/10/15/20-game questions). Per pool the opening model and the style
model are fitted on the candidates' games and score every question (one log score per candidate); each component alone
and their fusion give Top-5 lists. The fusion weights for a pool are fitted on the other pools' questions (no pool sees its
own); with a single pool the fusion uses equal weights. Pre-specified: fusion vs opening alone (player bootstrap of the
official per-question score). Writes --out (JSON) and, with --fusion-out, the fusion weights fitted on all pools for
predict_pid.py. Fold 0 is refused.

  python -I scripts/pid_cv.py --pools 1,2 3,4 --out results/pid_cv.json --fusion-out results/pid_fusion.json --workers 24
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import COMPONENTS, fit_models, score_questions  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402

PRE = [("fusion", "opening")]
REPORT = [("style", "opening"), ("fusion", "style")]


def parse_pools(specs):
    out = []
    for s in specs:
        folds = sorted({int(x) for x in s.split(",") if x.strip()})
        if not folds:
            raise ValueError(f"empty pool spec {s!r}")
        out.append(folds)
    return out


def run_pool(games, folds_df, folds, a, cfg, log=print):
    t0 = time.time()
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed,
                                      pool_id="p" + "".join(map(str, folds)))
    if a.max_questions:
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
    log(f"pool {folds}: {info['players']} players ({info['eligible']} eligible), {len(cands)} candidate games, {len(qs)} questions "
        f"{info['questions_by_n']}; loading {len(keys)} games", flush=True)
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    missing = [k for k in keys if k not in sgfs]
    if missing:
        raise ValueError(f"{len(missing)} games not found in the training files, e.g. {missing[:3]}")
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=a.workers)
    S = score_questions(opening, style, [list(k) for k in qs["game_keys"]], moves, feats)
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    log(f"pool {folds}: components in {time.time() - t0:.0f} s", flush=True)
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "seconds": time.time() - t0}


def evaluate(S, players, qs, truth):
    idx = topk(S, pools.TOP_K)
    tops = [[players[j] for j in row] for row in idx]
    per_q = np.array([pools.question_score(t, players[y]) for t, y in zip(tops, truth)])
    return pools.summarise(tops, [players[y] for y in truth], qs["n_games"].to_numpy()), per_q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="folds of each simulated pool, e.g. 1,2 3,4")
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests: only the first N questions of each pool")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None, help="write the fusion weights fitted on all pools (for predict_pid.py)")
    a = ap.parse_args(argv)
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
    results, per_q_all = [], {c: [] for c in (*COMPONENTS, "fusion")}
    pids_all = []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        if others:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in COMPONENTS], np.concatenate([o["truth"] for o in others]))
            weights = fus.w.tolist()
        else:
            fus = Fusion()
            fus.w = np.ones(len(COMPONENTS))
            weights = None
        S_f = fus.scores([r["S"][c] for c in COMPONENTS])
        scores, per_q = {}, {}
        for c in COMPONENTS:
            scores[c], per_q[c] = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
        scores["fusion"], per_q["fusion"] = evaluate(S_f, r["players"], r["questions"], r["truth"])
        for c in per_q:
            per_q_all[c].append(per_q[c])
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        results.append({"pool": r["info"], "scores": scores, "fusion_weights": weights, "seconds": r["seconds"]})
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f} (top1 {scores[c]['top1']:.3f})" for c in scores)
              + (f"  weights {np.round(weights, 3).tolist()}" if weights else "  (equal weights)"), flush=True)
    pid = np.concatenate(pids_all)
    comps = {}
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
        comps[group] = {}
        for x, z in pairs:
            d = paired_player_bootstrap(np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z]), pid, n=a.n_boot)
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]
            d.update({"per_pool_diff": per_pool, "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool))})
            comps[group][f"{x} vs {z}"] = d
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
                  + f"  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    overall = {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}
    out = {"pools": [p for p in results], "overall_score": overall, "comparisons": comps, "cfg": cfg, "n_players": a.n_players,
           "cand_games": a.cand_games, "seed": a.seed, "questions": int(len(pid)), **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    if a.fusion_out:
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in COMPONENTS], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": list(COMPONENTS), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs],
               "questions": int(len(pid)), **environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
