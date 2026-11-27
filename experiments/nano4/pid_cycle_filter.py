"""Each candidate 3-cycle of a base file implies the full Public truth (all other questions: the base Top-1). Score every
uploaded file with a known Public score under each hypothesis and keep the hypotheses that reproduce them to 7 decimals.
Hard checks: the files whose identity is certain (upload_1, upload_2). Soft checks: the other reported scores."""
import math
import pandas as pd

R2 = "/home/u2507115/BK_AICUP_2026-2"
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
def load(path):
    df = pd.read_csv(f"{R2}/results/{path}", dtype=str)
    df = df[df.question_id.str.startswith("pub_")].set_index("question_id")
    return {q: list(r) for q, r in zip(df.index, df[cols].values)}
HARD = {"probe/upload_1.csv": 0.9562058, "probe/upload_2.csv": 0.9817331}
SOFT = {"submission_f3_assign/submission_player.csv": 0.9833134, "submission_f3o_assign/submission_player.csv": 0.9873479,
        "submission/submission_player.csv": 0.9487291, "submission_assign/submission_player.csv": 0.9573466,
        "submission_f3/submission_player.csv": 0.9704912, "submission_f3o/submission_player.csv": 0.9658747,
        "submission_f5r4/submission_player.csv": 0.9737858}
files = {k: load(k) for k in (*HARD, *SOFT)}
def score(lists, truth):
    return sum(math.exp(-lists[q].index(t)) if t in lists[q] else 0.0 for q, t in truth.items()) / 400
for base_name in ("submission_f3oref_assign", "submission_f5r4_assign"):
    base = load(f"{base_name}/submission_player.csv")
    owner = {r[0]: q for q, r in base.items()}
    def label(x, y):
        c = base[y][0]
        return base[x].index(c) + 1 if c in base[x] else "out"
    cycles = set()
    for a in base:
        for r in (2, 3):
            b = owner[base[a][r - 1]]
            if b == a:
                continue
            for c in base:
                if c in (a, b):
                    continue
                if sorted([label(a, b), label(b, c), label(c, a)], key=str) == sorted([2, 3, "out"], key=str):
                    cycles.add(((a, b), (b, c), (c, a)))
    default = {q: r[0] for q, r in base.items()}
    survivors = []
    for cyc in cycles:
        truth = dict(default)
        for x, y in cyc:
            truth[x] = base[y][0]
        assert abs(score(base, truth) - 0.9937580) < 1.5e-7
        hard = {k: round(score(files[k], truth), 7) for k in HARD}
        if all(abs(hard[k] - v) < 1.5e-7 for k, v in HARD.items()):
            soft = {k: round(score(files[k], truth), 7) for k in SOFT}
            survivors.append((cyc, hard, soft, sum(abs(soft[k] - v) < 1.5e-7 for k, v in SOFT.items())))
    print(f"== base {base_name}: {len(cycles)} cycles, {len(survivors)} reproduce upload_1 and upload_2 exactly")
    for cyc, hard, soft, nsoft in sorted(survivors, key=lambda s: -s[3])[:10]:
        print(f"  cycle {[f'{x}: truth {base[y][0]} (rank {label(x, y)})' for x, y in cyc]}  soft checks matched {nsoft}/{len(SOFT)}")
        for k, v in SOFT.items():
            print(f"      {k.split('/')[0]:24s} predicted {soft[k]:.7f} reported {v:.7f} {'OK' if abs(soft[k] - v) < 1.5e-7 else 'x'}")
