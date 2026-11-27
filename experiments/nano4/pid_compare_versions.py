"""Pairwise differences between the four official submissions, per pool and per question size."""
import pandas as pd
R2 = "/home/u2507115/BK_AICUP_2026-2"
V = {"fusion": "submission", "fusion3": "submission_f3", "assign": "submission_assign", "f3_assign": "submission_f3_assign"}
subs = {k: pd.read_csv(f"{R2}/results/{v}/submission_player.csv", dtype=str) for k, v in V.items()}
n = {}
for pool, f in (("pub_", "player_identification_test_public.csv"), ("priv_", "player_identification_test_private.csv")):
    q = pd.read_csv(f"{R2}/data/tests/{f}", dtype=str, usecols=["question_id", "num_games"])
    n.update(dict(zip(q.question_id, q.num_games.astype(int))))
base = subs["fusion"]
qid = base.question_id.tolist()
pool = pd.Series([x[: x.index("_") + 1] for x in qid], index=qid)
ng = pd.Series([n[x] for x in qid], index=qid)
pairs = [("fusion", "fusion3"), ("fusion", "assign"), ("fusion3", "f3_assign"), ("assign", "f3_assign"), ("fusion3", "assign"), ("fusion", "f3_assign")]
for a, b in pairs:
    A, B = subs[a].set_index("question_id"), subs[b].set_index("question_id")
    d1 = (A.top_1 != B.top_1)
    d5 = (A[[f"top_{i}" for i in range(1, 6)]].values != B[[f"top_{i}" for i in range(1, 6)]].values).any(axis=1)
    d5 = pd.Series(d5, index=A.index)
    # truth of A's top_1 appearing in B's top-5 and vice versa (how far the changed Top-1 moved)
    inB = pd.Series([A.top_1[i] in B.loc[i, [f"top_{k}" for k in range(1, 6)]].tolist() for i in A.index], index=A.index)
    for p in ("pub_", "priv_"):
        m = pool == p
        by = {s: int(d1[m & (ng == s)].sum()) for s in (5, 10, 15, 20)}
        print(f"{a} vs {b} {p}: top1 differ {int(d1[m].sum())} (by n {by}); any of top5 differ {int(d5[m].sum())}; A's top1 still in B's top5 {int((d1 & inB)[m].sum())}/{int(d1[m].sum())}")
