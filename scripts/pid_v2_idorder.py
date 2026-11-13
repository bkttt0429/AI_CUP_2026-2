#!/usr/bin/env python
"""Is there any regularity between question numbers and candidate numbers in the released files? (METHOD_NOTES 2.54)
Label-free: the Top-1 of a submission file stands in for the owner (5-game questions of the added pools are right about 94 %
of the time; the original pools are known to be right). Per pool and question size: Spearman correlation between the question
number and the Top-1 candidate number, the same against the question's row in the file, the spread of Top-1 candidate numbers
over six equal ranges of the candidate ids, and whether candidate ids follow the candidates' order in the candidate file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sub = pd.read_csv(sys.argv[1], dtype=str).set_index("question_id")
T = ROOT / "data" / "tests_1111"
num = lambda s: int(s.rsplit("_", 1)[1])  # noqa: E731
for pre, qf, cf in (("pub", "test_public", "candidates_public"), ("priv", "test_private", "candidates_private"), ("pub2", "test2_public", "candidates2_public"), ("priv2", "test2_private", "candidates2_private")):
    q = pd.read_csv(T / f"player_identification_{qf}.csv", dtype=str, usecols=["question_id", "num_games"])
    c = pd.read_csv(T / f"player_identification_{cf}.csv", dtype=str, usecols=["player_id", "game_id"])
    first = c.drop_duplicates("player_id").player_id.map(num).to_numpy()
    gid = c.game_id.map(num).to_numpy()
    print(f"{pre}: candidates {len(first)}; ids in file order increasing: {bool((np.diff(first) > 0).all())}; game ids increasing: {bool((np.diff(gid) > 0).all())}; "
          f"question ids in file order increasing: {bool((np.diff(q.question_id.map(num)) > 0).all())}; sizes along the file (first 12): {q.num_games.tolist()[:12]}")
    qn, row, n = q.question_id.map(num).to_numpy(), np.arange(len(q)), q.num_games.astype(int).to_numpy()
    t1 = sub.loc[q.question_id, "top_1"].map(num).to_numpy()
    C = len(first)
    for m in sorted(set(n.tolist())):
        k = n == m
        r1, p1 = spearmanr(qn[k], t1[k])
        hist = np.bincount((t1[k] - 1) * 6 // C, minlength=6)
        print(f"   size {m:2d}: {int(k.sum())} questions; Spearman(question number, Top-1 candidate number) {r1:+.3f} (p {p1:.2f}); Top-1 over six id ranges {hist.tolist()}; "
              f"distinct Top-1 {len(set(t1[k].tolist()))}; question numbers {qn[k].min()}-{qn[k].max()}")
    r, p = spearmanr(qn, t1)
    print(f"   all sizes: Spearman {r:+.3f} (p {p:.2f}); Top-1 over six id ranges {np.bincount((t1 - 1) * 6 // C, minlength=6).tolist()}; never named as Top-1: {C - len(set(t1.tolist()))} of {C} candidates")
