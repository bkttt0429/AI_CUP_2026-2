#!/usr/bin/env python3
"""METHOD_NOTES 2.240: cross-check of the ORIGINAL pools' rows (frozen since 11-10: results/submission_old_embt_assign) with the
new sequence networks. One side at a time:
  vectors   orig_<side>_net-1_<tag>_t<step>.npz of scripts/pid_v2_seq_finetune.py --mode orig (the side's candidate games
            that have per-ply vectors + its question games)
  score     per network: cosine of the question's summed game vectors with every candidate's summed game vectors, row-standardised;
            mean over the networks (the "seq" component of aicup_pid.v2.components)
  answers   unassigned Top-5, and the one-to-one assignment of the 400 questions to the 400 candidates (Hungarian)
Reported: agreement of the new Top-1 with the old rows' Top-1 (on the Public side the old rows score 1.0000000 on the
leaderboard, so their Top-1 are the answers - used for EVALUATION only), the margins, every family alone, and each
disagreement in detail. Nothing here writes a submission.
  python -I scripts/pid_orig_check.py --side pub --vec-dir /work/u2507115/pid_v2/orig_ft --out results/final/v2_orig_check_pub.json
"""
import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup_pid import pools  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402


def assign(S):
    r, c = linear_sum_assignment(-S)
    out = np.empty(len(S), int)
    out[r] = c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--side", required=True, choices=("pub", "priv"))
    ap.add_argument("--vec-dir", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50])
    ap.add_argument("--old", default=str(ROOT / "results" / "submission_old_embt_assign" / "submission_player.csv"))
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
    qs = pools.read_questions(T / f"player_identification_test_{tag}.csv")
    qids = [q["question_id"] for q in qs]
    nq = np.array([len(q["games"]) for q in qs])
    with open(a.old, newline="", encoding="utf-8") as fh:
        old = {r["question_id"]: [r[f"top_{k}"] for k in range(1, 6)] for r in csv.DictReader(fh)}
    old_top = np.array([[pidx[p] for p in old[q]] for q in qids])
    out = {"side": a.side, "questions": len(qids), "candidates": len(players), "old_rows_distinct_top1": len(set(old_top[:, 0].tolist())), "steps": {}}
    print(f"{a.side}: {len(qids)} questions, {len(players)} candidates; the old rows name {out['old_rows_distinct_top1']} distinct Top-1 candidates", flush=True)
    answers = {}
    for step in a.steps:
        files = sorted(glob.glob(f"{a.vec_dir}/orig_{a.side}_net-1_*_t{step}.npz"))
        fam = {}
        for f in files:
            m = re.search(r"_net-1_(o\d+F)s\d+e_t\d+\.npz$", f)
            fam.setdefault(m.group(1), []).append(f)
        assert files, f"no vectors of step {step}"
        unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
        mats, used_c, used_q = {}, None, None
        for f in files:
            z = np.load(f)
            one = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
            cm = [[k for k in ks if k in one] for ks in ckeys]
            qm = [[f"q:{q['question_id']}:{i}" for i in range(len(q["games"])) if f"q:{q['question_id']}:{i}" in one] for q in qs]
            assert all(cm) and all(qm), "a candidate or a question has no vector"
            used_c, used_q = np.array([len(x) for x in cm]), np.array([len(x) for x in qm])
            C = unit(np.stack([np.sum([one[k] for k in ks], axis=0) for ks in cm]).astype(np.float64))
            Q = unit(np.stack([np.sum([one[k] for k in ks], axis=0) for ks in qm]).astype(np.float64))
            mats[f] = standardise(Q @ C.T)
        S = np.mean(list(mats.values()), axis=0)
        top = np.argsort(-S, axis=1)[:, :5]
        asg = assign(S)
        by_fam = {k: assign(np.mean([mats[f] for f in v], axis=0)) for k, v in sorted(fam.items())}
        srt = np.sort(S, axis=1)
        own = S[np.arange(len(S)), asg]
        best_other = np.where(top[:, 0] == asg, srt[:, -2], srt[:, -1])
        res = {"networks": len(files), "families": {k: len(v) for k, v in sorted(fam.items())},
               "candidate_games_used": [int(used_c.min()), int(used_c.max())], "question_games_used_vs_given": [int(used_q.sum()), int(nq.sum())],
               "top1_same_as_old_unassigned": int((top[:, 0] == old_top[:, 0]).sum()), "top1_same_as_old_assigned": int((asg == old_top[:, 0]).sum()),
               "assignment_changes_top1": int((asg != top[:, 0]).sum()),
               "margin_assigned_minus_best_other_percentiles": {str(p): float(np.percentile(own - best_other, p)) for p in (0, 1, 5, 10, 25, 50)},
               "families_same_as_old": {k: int((v == old_top[:, 0]).sum()) for k, v in by_fam.items()}, "disagreements": []}
        for i in np.flatnonzero(asg != old_top[:, 0]):
            o = int(old_top[i, 0])
            res["disagreements"].append({"question": qids[i], "games": int(nq[i]), "new": players[asg[i]], "old": players[o],
                                         "margin_new_minus_old": float(S[i, asg[i]] - S[i, o]), "rank_of_old_under_new": int((S[i] > S[i, o]).sum()) + 1,
                                         "rank_of_new_in_old_row": (old[qids[i]].index(players[asg[i]]) + 1) if players[asg[i]] in old[qids[i]] else None,
                                         "unassigned_top1": players[top[i, 0]], "families": {k: players[v[i]] for k, v in by_fam.items()}})
        out["steps"][str(step)] = res
        answers[step] = asg
        print(f"{a.side} step {step}: {len(files)} networks {res['families']}; games per candidate {res['candidate_games_used']}; question games {res['question_games_used_vs_given']}", flush=True)
        print(f"  new Top-1 equal to the old rows' Top-1: unassigned {res['top1_same_as_old_unassigned']} / {len(qids)}, assigned {res['top1_same_as_old_assigned']} / {len(qids)}"
              f" (the assignment changes {res['assignment_changes_top1']} Top-1); each family alone: {res['families_same_as_old']}", flush=True)
        print(f"  margin of the assigned answer over the best other candidate, percentiles 0/1/5/10/25/50: " + " ".join(f"{v:.2f}" for v in res["margin_assigned_minus_best_other_percentiles"].values()), flush=True)
        for d in res["disagreements"]:
            print(f"  DIFFERENT {d['question']} ({d['games']} games): new {d['new']} vs old {d['old']}; margin {d['margin_new_minus_old']:+.2f}; old is rank {d['rank_of_old_under_new']} under the new scores; "
                  f"new is rank {d['rank_of_new_in_old_row']} in the old row; unassigned Top-1 {d['unassigned_top1']}; families {d['families']}", flush=True)
    if len(a.steps) > 1:
        s0, s1 = a.steps[0], a.steps[-1]
        out["answers_equal_between_steps"] = int((answers[s0] == answers[s1]).sum())
        print(f"  assigned answers equal at {s0} and {s1} steps: {out['answers_equal_between_steps']} / {len(qids)}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
