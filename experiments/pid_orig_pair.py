#!/usr/bin/env python3
"""METHOD_NOTES 2.241: a closer look at ONE disputed pair of the original pool (two questions, two candidates; the old rows and
the new networks assign them the opposite way). From the vectors of scripts/pid_v2_seq_finetune.py --mode orig:
  a. how alike the two candidates are: cosine of their prototypes against all candidate pairs of the pool
  b. can their own reference games be told apart: every reference game of A and B against the two prototypes (its own game
     left out of its own prototype), share assigned to the right one
  c. every game of the two questions: cosine with prototype A minus cosine with prototype B (mean over the networks), votes
  d. the two assignments' totals
Evaluation only; nothing here writes a submission.
  python -I scripts/pid_orig_pair.py --side priv --cands priv_p_0017 priv_p_0368 --questions priv_q_0115 priv_q_0255 --vec-dir /work/u2507115/pid_v2/orig_ft --out results/final/v2_orig_pair_priv.json
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup_pid import pools  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--side", required=True, choices=("pub", "priv"))
    ap.add_argument("--cands", nargs=2, required=True)
    ap.add_argument("--questions", nargs=2, required=True)
    ap.add_argument("--vec-dir", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    csv.field_size_limit(10 ** 9)
    tag = "public" if a.side == "pub" else "private"
    T = Path(a.tests_dir)
    with open(T / f"player_identification_candidates_{tag}.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    players = sorted({r["player_id"] for r in rows})
    pidx = {p: i for i, p in enumerate(players)}
    ckeys = [[] for _ in players]
    for i, r in enumerate(rows):
        ckeys[pidx[r["player_id"]]].append(f"c:{i}:{r['game_id']}")
    qs = {q["question_id"]: q for q in pools.read_questions(T / f"player_identification_test_{tag}.csv")}
    A, B = (pidx[c] for c in a.cands)
    unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731
    out = {"side": a.side, "candidates": a.cands, "questions": a.questions, "steps": {}}
    for step in a.steps:
        files = sorted(glob.glob(f"{a.vec_dir}/orig_{a.side}_net-1_*_t{step}.npz"))
        sims, loo, qd = [], {a.cands[0]: [], a.cands[1]: []}, {q: [] for q in a.questions}
        nn_rank = []
        for f in files:
            z = np.load(f)
            one = dict(zip(z["key"].astype(str), z["emb"].astype(np.float64)))
            G = [np.stack([one[k] for k in ks if k in one]) for ks in ckeys]
            P = unit(np.stack([g.sum(0) for g in G]))
            C = P @ P.T
            np.fill_diagonal(C, -np.inf)
            off = C[np.isfinite(C)]
            sims.append({"cos": float(C[A, B]), "z": float((C[A, B] - off.mean()) / off.std()), "pct": float((off < C[A, B]).mean())})
            nn_rank.append([int((C[A] > C[A, B]).sum()) + 1, int((C[B] > C[B, A]).sum()) + 1])
            for name, me, other in ((a.cands[0], A, B), (a.cands[1], B, A)):
                g = G[me]
                own = unit(g.sum(0)[None, :] - g)  # its own game left out
                d = (unit(g) * own).sum(1) - unit(g) @ P[other]
                loo[name].append(d)
            for q in a.questions:
                v = np.stack([one[f"q:{q}:{i}"] for i in range(len(qs[q]["games"])) if f"q:{q}:{i}" in one])
                qd[q].append(unit(v) @ P[A] - unit(v) @ P[B])
        res = {"networks": len(files),
               "prototype_cosine": float(np.mean([s["cos"] for s in sims])), "prototype_cosine_z_among_all_pairs": float(np.mean([s["z"] for s in sims])),
               "share_of_all_pairs_less_alike": float(np.mean([s["pct"] for s in sims])),
               "rank_of_each_other_among_neighbours_median": [float(np.median([r[0] for r in nn_rank])), float(np.median([r[1] for r in nn_rank]))]}
        print(f"{a.side} step {step}: {len(files)} networks; prototypes of {a.cands[0]} and {a.cands[1]}: cosine {res['prototype_cosine']:.3f}, z {res['prototype_cosine_z_among_all_pairs']:+.1f} among all pairs "
              f"({100 * res['share_of_all_pairs_less_alike']:.3f}% of the pairs are less alike); each is the other's neighbour number {res['rank_of_each_other_among_neighbours_median']}", flush=True)
        res["own_games"] = {}
        for name in a.cands:
            d = np.mean(loo[name], axis=0)  # per reference game, mean over the networks: own prototype (game left out) minus the other's
            res["own_games"][name] = {"games": int(len(d)), "closer_to_own": int((d > 0).sum()), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
            print(f"  reference games of {name}: {int((d > 0).sum())} of {len(d)} closer to their own prototype than to the other's (mean difference {d.mean():+.4f}, se {d.std(ddof=1) / np.sqrt(len(d)):.4f})", flush=True)
        res["question_games"] = {}
        tot = {}
        for q in a.questions:
            d = np.mean(qd[q], axis=0)  # per question game: cosine with A minus cosine with B
            tot[q] = float(d.sum())
            res["question_games"][q] = {"games": int(len(d)), "closer_to_first": int((d > 0).sum()), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                                        "per_game": [round(float(x), 4) for x in d]}
            print(f"  games of {q}: {int((d > 0).sum())} of {len(d)} closer to {a.cands[0]} than to {a.cands[1]} (mean difference {d.mean():+.4f}"
                  + (f", se {d.std(ddof=1) / np.sqrt(len(d)):.4f}" if len(d) > 1 else "") + f"); per game {[round(float(x), 3) for x in d]}", flush=True)
        q1, q2 = a.questions
        res["per_game_mean_gap"] = {f"{q1}->{a.cands[0]} and {q2}->{a.cands[1]} minus the opposite": float(res["question_games"][q1]["mean"] - res["question_games"][q2]["mean"])}
        print(f"  mean per-game preference for {a.cands[0]}: {q1} {res['question_games'][q1]['mean']:+.4f} vs {q2} {res['question_games'][q2]['mean']:+.4f} "
              f"-> the question that leans more to {a.cands[0]} is {q1 if res['question_games'][q1]['mean'] > res['question_games'][q2]['mean'] else q2}", flush=True)
        out["steps"][str(step)] = res
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
