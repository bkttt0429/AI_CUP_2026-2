#!/usr/bin/env python
"""B1 step 4 (real-data gate) on the void 10-30 Public pool: the rank model's 10-class output as an identification signal.

Inputs from the rank session (results/b1_public, ct9 with all 24 members; official players are in no training set, so
nothing leaks): submission_rank_ct9_proba.csv rows pub_c<cand>_k<0..4> (5 groups x 20 of each candidate's 100 games) and
pub_q_<id> (the questions), plus cands_map.csv. Pre-set component (fixed before looking): candidate vector = mean of the 5
groups' log probabilities; question vector = log probabilities of its n games; every dimension z-scored with the 400
candidates' mean / sd; score = cosine. Truth = Top-1 of results/probe/upload_fix.csv (Public 1.0000000). Report-only
alternatives: -|E[rank] difference| and -symmetric KL. Writes the [Q, C] matrix (questions in file order, candidates in
sorted id order, the order predict_pid uses) for the later fusion test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def official(S, players, truth_idx, n):
    order = np.argsort(-S, axis=1)
    rank = np.array([int(np.where(order[q] == truth_idx[q])[0][0]) + 1 for q in range(len(S))])
    sc = np.where(rank <= 5, np.exp(-(rank - 1)), 0.0)
    return {"score": float(sc.mean()), "top1": float((rank == 1).mean()), "top5": float((rank <= 5).mean()), "median_rank": float(np.median(rank)),
            "by_n": {str(k): float(sc[n == k].mean()) for k in sorted(set(n.tolist()))}}, rank


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--b1-dir", default=str(Path.home() / "BK_AICUP_2026-1" / "results" / "b1_public"))
    ap.add_argument("--questions", default=str(ROOT / "data" / "tests" / "player_identification_test_public.csv"))
    ap.add_argument("--truth", default=str(ROOT / "results" / "probe" / "upload_fix.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.b1_dir)
    P = pd.read_csv(d / "submission_rank_ct9_proba.csv")
    cmap = pd.read_csv(d / "inputs" / "cands_map.csv", dtype=str)
    L = np.log(np.clip(P[[f"p_{c}" for c in CLS]].to_numpy(), 1e-30, None))
    lp = dict(zip(P.question_id, L))
    qid_col = next(c for c in cmap.columns if "question" in c)
    cand_rows = cmap.groupby("player_id")[qid_col].apply(lambda s: sorted(set(s))).to_dict()
    players = sorted(cand_rows)
    C = np.stack([np.mean([lp[k] for k in cand_rows[p]], axis=0) for p in players])
    qs = pd.read_csv(a.questions, dtype=str, keep_default_na=False)
    Q = np.stack([lp[q] for q in qs.question_id])
    n = qs.num_games.astype(int).to_numpy()
    truth = pd.read_csv(a.truth, dtype=str).set_index("question_id").top_1
    tix = np.array([players.index(truth[q]) for q in qs.question_id])
    mu, sd = C.mean(axis=0), C.std(axis=0) + 1e-9
    Zc, Zq = (C - mu) / sd, (Q - mu) / sd
    Zc /= np.linalg.norm(Zc, axis=1, keepdims=True)
    Zq /= np.linalg.norm(Zq, axis=1, keepdims=True)
    S = Zq @ Zc.T
    pc, pq = np.exp(C - np.log(np.exp(C).sum(axis=1, keepdims=True))), np.exp(Q - np.log(np.exp(Q).sum(axis=1, keepdims=True)))
    E = lambda p: p @ np.arange(10)
    alt = {"neg_abs_E_diff": -np.abs(E(pq)[:, None] - E(pc)[None, :]),
           "neg_sym_kl": -(np.einsum("qk,qk->q", pq, np.log(pq + 1e-30))[:, None] - pq @ np.log(pc + 1e-30).T
                           + np.einsum("ck,ck->c", pc, np.log(pc + 1e-30))[None, :] - np.log(pq + 1e-30) @ pc.T)}
    res, rank = official(S, players, tix, n)
    out = {"pre_set_zcos": res, "random_baseline": float(sum(np.exp(-k) for k in range(5)) / len(players)), "report_only": {}}
    print(f"B1 alone on old Public (pre-set z-cosine): score {res['score']:.4f}, Top-1 {res['top1']:.3f}, Top-5 {res['top5']:.3f}, median rank of the truth {res['median_rank']:.0f} of {len(players)}; "
          f"by n {res['by_n']}; random {out['random_baseline']:.4f}", flush=True)
    for k, M in alt.items():
        r2, _ = official(M, players, tix, n)
        out["report_only"][k] = r2
        print(f"  report only {k}: score {r2['score']:.4f}, Top-1 {r2['top1']:.3f}, median rank {r2['median_rank']:.0f}", flush=True)
    # where the truth sits for the questions the identification models get wrong without assignment
    for name in ("submission_f3oreft", "embed/oldcheck_off"):
        f = ROOT / "results" / name / "submission_player.csv"
        if f.exists():
            s = pd.read_csv(f, dtype=str).set_index("question_id")
            for qi, q in enumerate(qs.question_id):
                if s.loc[q, "top_1"] != truth[q]:
                    pick = players.index(s.loc[q, "top_1"])
                    print(f"  {name} wrong on {q} (n={n[qi]}): B1 rank of the truth {rank[qi]}, B1 rank of the wrong pick {int((S[qi] > S[qi, pick]).sum()) + 1}")
    np.save(Path(a.out).with_suffix(".npy"), S)
    out["players"], out["questions"] = players, qs.question_id.tolist()
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.npy')}")


if __name__ == "__main__":
    sys.exit(main())
