import pandas as pd
R2 = "/home/u2507115/BK_AICUP_2026-2"
V = {"f3o_assign": "submission_f3o_assign", "f5r4": "submission_f5r4", "f5r4_assign": "submission_f5r4_assign", "f3or4_assign": "submission_f3or4_assign", "f3o": "submission_f3o"}
subs = {k: pd.read_csv(f"{R2}/results/{v}/submission_player.csv", dtype=str).set_index("question_id") for k, v in V.items()}
n = {}
for f in ("player_identification_test_public.csv", "player_identification_test_private.csv"):
    q = pd.read_csv(f"{R2}/data/tests/{f}", dtype=str, usecols=["question_id", "num_games"]); n.update(dict(zip(q.question_id, q.num_games.astype(int))))
for a, b in (("f3o_assign", "f5r4_assign"), ("f3o", "f5r4"), ("f5r4", "f5r4_assign"), ("f5r4_assign", "f3or4_assign")):
    A, B = subs[a], subs[b]; d = A.top_1 != B.top_1
    for p in ("pub_", "priv_"):
        m = A.index.str.startswith(p)
        print(f"{a} vs {b} {p}: top1 differ {int(d[m].sum())} (by n {{ {', '.join(f'{s}: {sum(1 for q in A.index[m & d] if n[q] == s)}' for s in (5, 10, 15, 20))} }})")
