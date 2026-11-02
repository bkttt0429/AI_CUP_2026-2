#!/usr/bin/env python
"""Line B3: does a global one-to-one assignment beat the per-question Top-1 on one-question-per-player pools?

Each --pools entry (folds) gives one pool per --seeds value from aicup_pid.pools.make_pool_one_per_player (up to 400
players with >= 105 Black games, 100 candidate games each, exactly one question per player, sizes 20/15/10/5 up to
--per-size each: the official design). Per pool the opening and style models are fitted on the candidates, the questions
scored, the components fused with weights fitted on the OTHER fold sets' pools (cross-fit; equal weights with one fold
set), and two arms are scored: fusion = per-question Top-5, assign = aicup_pid.assign.assign_topk (Hungarian Top-1, the
row's order after). Pre-specified: assign vs fusion -- player bootstrap over all pools' questions (the question's true
player is the unit), every fold set's mean difference positive, and overall >= 0.003 or n=5 >= 0.005. --control adds a
report-only pool per fold set whose questions are drawn with repeated players (make_pool, --control-questions of them),
where the one-to-one constraint is false: the harm the assignment does when the design is not one per player, and the
distinct-Top-1 count that is meant to tell the two designs apart. Fold 0 is refused.

  python -I scripts/pid_assign_cv.py --pools 1,2 3,4 --seeds 0 1 2 3 4 --control --workers 12 --out results/assign/assign_cv.json
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
from aicup_pid.assign import assign_topk, assignment_report  # noqa: E402
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, parse_all, score_questions  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_cv import parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005


def score_arm(idx, players, qs, truth):
    tops = [[players[j] for j in row] for row in idx]
    per_q = np.array([pools.question_score(t, players[y]) for t, y in zip(tops, truth)])
    return pools.summarise(tops, [players[y] for y in truth], qs["n_games"].to_numpy()), per_q


def build_group(games, folds_df, folds, a, cfg, log=print):
    """All seeds' one-per-player pools of one fold set (plus the control pool), games parsed once."""
    t0 = time.time()
    made = [(pools.make_pool_one_per_player(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, per_size=a.per_size, seed=s,
                                             pool_id=f"p{''.join(map(str, folds))}s{s}", order=a.order), False) for s in a.seeds]
    if a.control:
        cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seeds[0],
                                          pool_id=f"c{''.join(map(str, folds))}", order=a.order)
        pick = np.sort(np.random.RandomState(a.seeds[0]).permutation(len(qs))[:a.control_questions])
        made.append(((cands, qs.iloc[pick].reset_index(drop=True), {**info, "questions": int(len(pick)), "control": True}), True))
    keys = sorted({k for (cands, qs, _), _ in made for k in cands["game_key"]} | {k for (_, qs, _), _ in made for ks in qs["game_keys"] for k in ks})
    log(f"fold set {folds}: {len(made)} pools, loading {len(keys)} games", flush=True)
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all(sgfs, a.workers)
    runs = []
    for (cands, qs, info), is_control in made:
        by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
        opening, style = fit_from_parsed(by_player, moves, feats, cfg)
        S = score_questions(opening, style, [list(k) for k in qs["game_keys"]], moves, feats)
        truth = np.array([opening.index[p] for p in qs["player_id"]])
        runs.append({"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "control": is_control, "folds": list(folds)})
    log(f"fold set {folds}: components of {len(runs)} pools in {time.time() - t0:.0f} s", flush=True)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="folds of each fold set, e.g. 1,2 3,4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--per-size", type=int, default=100)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--control", action="store_true", help="report-only pool per fold set with repeated players")
    ap.add_argument("--control-questions", type=int, default=None, help="questions in the control pool (default: --n-players)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle", help="how a player's games are split: shuffle (default) or file = earliest games as the candidate's (drift stress test)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.control_questions is None:
        a.control_questions = a.n_players
    t0 = time.time()
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink}
        groups = [build_group(games, folds_df, f, a, cfg) for f in specs]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    results, controls = [], []
    per_q_all = {"fusion": [], "assign": []}
    pids_all, n_all, group_of = [], [], []
    for i, runs in enumerate(groups):
        others = [r for j, g in enumerate(groups) if j != i for r in g if not r["control"]]
        if others:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in COMPONENTS], np.concatenate([o["truth"] for o in others]))
            weights = fus.w.tolist()
        else:
            fus = Fusion()
            fus.w = np.ones(len(COMPONENTS))
            weights = None
        for r in runs:
            S_f = fus.scores([r["S"][c] for c in COMPONENTS])
            idx_f, idx_a = topk(S_f, pools.TOP_K), assign_topk(S_f, pools.TOP_K)
            s_f, q_f = score_arm(idx_f, r["players"], r["questions"], r["truth"])
            s_a, q_a = score_arm(idx_a, r["players"], r["questions"], r["truth"])
            rep = assignment_report(S_f, idx_a)
            if r["control"]:
                controls.append({"folds": r["folds"], "questions": int(len(r["questions"])), "distinct_players": int(r["questions"]["player_id"].nunique()),
                                 "distinct_top1": rep["distinct_top1_before"], "n_top1_changed": rep["n_top1_changed"], "fusion": s_f["score"], "assign": s_a["score"],
                                 "diff": float(np.mean(q_a - q_f))})
                print(f"control {r['folds']}: {len(r['questions'])} questions from {controls[-1]['distinct_players']} players, distinct Top-1 {rep['distinct_top1_before']}; "
                      f"fusion {s_f['score']:.4f} assign {s_a['score']:.4f} diff {controls[-1]['diff']:+.4f}", flush=True)
                continue
            per_q_all["fusion"].append(q_f)
            per_q_all["assign"].append(q_a)
            pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
            n_all.append(r["questions"]["n_games"].to_numpy())
            group_of.append(np.full(len(q_f), i))
            results.append({"pool": r["info"], "scores": {"fusion": s_f, "assign": s_a}, "fusion_weights": weights, "assignment": rep})
            print(f"pool {r['info']['folds']} seed {r['info']['seed']}: fusion {s_f['score']:.4f} (top1 {s_f['top1']:.3f})  assign {s_a['score']:.4f} (top1 {s_a['top1']:.3f})  "
                  f"distinct Top-1 {rep['distinct_top1_before']}/{rep['n_questions']}, changed {rep['n_top1_changed']}" + (f"  weights {np.round(weights, 3).tolist()}" if weights else "  (equal weights)"), flush=True)
    pid, n_games, grp = np.concatenate(pids_all), np.concatenate(n_all), np.concatenate(group_of)
    qa, qf = np.concatenate(per_q_all["assign"]), np.concatenate(per_q_all["fusion"])
    d = paired_player_bootstrap(qa, qf, pid, n=a.n_boot)
    per_group = [float(np.mean(qa[grp == i] - qf[grp == i])) for i in range(len(groups))]
    d.update({"per_pool_diff": per_group, "diff_n5": float(np.mean((qa - qf)[n_games == 5])) if (n_games == 5).any() else None,
              "by_n": {str(n): float(np.mean((qa - qf)[n_games == n])) for n in sorted(set(n_games.tolist()))},
              "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_group) and (d["diff"] >= MIN_DIFF or ((d.get("diff_n5") or 0) >= MIN_DIFF_N5)))})
    print(f"[pre_specified] assign vs fusion: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  fold sets " + " ".join(f"{v:+.4f}" for v in per_group)
          + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {"fusion": float(qf.mean()), "assign": float(qa.mean())},
           "comparisons": {"pre_specified": {"assign vs fusion": d}}, "control": controls, "cfg": cfg, "seeds": a.seeds, "n_players": a.n_players,
           "cand_games": a.cand_games, "per_size": a.per_size, "order": a.order, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
