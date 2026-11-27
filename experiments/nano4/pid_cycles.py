"""With a bijective truth and a permutation submission, the 3 wrong Public questions of a file scoring 0.9937580 (one
truth at rank 2, one at rank 3, one outside the five) form a 3-cycle: the truth of each is the Top-1 of the next.
Enumerate every 3-cycle A->B->C->A whose edge labels (rank of Top1(next) in the list, or 'out') are {2, 3, out}."""
import itertools
import pandas as pd

R2 = "/home/u2507115/BK_AICUP_2026-2"
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
for name in ("submission_f3oref_assign", "submission_f5r4_assign"):
    df = pd.read_csv(f"{R2}/results/{name}/submission_player.csv", dtype=str)
    df = df[df.question_id.str.startswith("pub_")].set_index("question_id")
    lists = {q: list(r) for q, r in zip(df.index, df[cols].values)}
    owner = {r[0]: q for q, r in lists.items()}
    assert len(owner) == 400, "Top-1 is not a permutation"

    def label(x, y):
        c = lists[y][0]
        return lists[x].index(c) + 1 if c in lists[x] else "out"

    cycles = set()
    for a in lists:
        for r in (2, 3):  # one edge out of a has label 2 or 3; walk it
            b = owner[lists[a][r - 1]]
            if b == a:
                continue
            for c in lists:
                if c in (a, b):
                    continue
                labs = sorted([label(a, b), label(b, c), label(c, a)], key=str)
                if labs == sorted([2, 3, "out"], key=str):
                    cycles.add(tuple(sorted([(a, b), (b, c), (c, a)])))
    print(f"{name}: {len(cycles)} candidate 3-cycles with edge labels {{2, 3, out}}")
    # how many distinct questions appear, and how many cycles use only 'tight' edges
    qs = sorted({q for cyc in cycles for e in cyc for q in e})
    print(f"  questions involved: {len(qs)}")
