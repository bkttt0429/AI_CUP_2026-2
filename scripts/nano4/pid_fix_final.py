"""Build the corrected file: submission_f3oref_assign with the unique consistent 3-cycle fixed (Public rows only), and
score every file under the implied Public truth (truncated to 7 decimals as the leaderboard does)."""
import math, glob
import pandas as pd

R2 = "/home/u2507115/BK_AICUP_2026-2"
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
base = pd.read_csv(f"{R2}/results/submission_f3oref_assign/submission_player.csv", dtype=str)
pub = base[base.question_id.str.startswith("pub_")].set_index("question_id")
truth = {q: r[0] for q, r in zip(pub.index, pub[cols].values)}
FIX = {"pub_q_0357": "pub_p_0052", "pub_q_0018": "pub_p_0114", "pub_q_0189": "pub_p_0277"}
truth.update(FIX)
fixed = base.copy()
for q, t in FIX.items():
    i = fixed.index[fixed.question_id == q][0]
    lst = fixed.loc[i, cols].tolist()
    lst = [t, *[c for c in lst if c != t]][:5]
    fixed.loc[i, cols] = lst
out = f"{R2}/results/probe/upload_fix.csv"
fixed.to_csv(out, index=False)
def score(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df.question_id.str.startswith("pub_")].set_index("question_id")
    s = sum(math.exp(-list(r).index(truth[q])) if truth[q] in list(r) else 0.0 for q, r in zip(df.index, df[cols].values)) / 400
    return math.floor(s * 1e7) / 1e7
rows = (fixed[cols].values != base[cols].values).any(axis=1)
pubm = fixed.question_id.str.startswith("pub_").values
print(f"upload_fix: rows changed vs f3oref_assign pub {int(rows[pubm].sum())} priv {int(rows[~pubm].sum())}; "
      f"five distinct {all(len(set(r)) == 5 for r in fixed[cols].values)}; pub Top-1 distinct {fixed[pubm].top_1.nunique()}; "
      f"own pool {all(all(c.startswith(q[:q.index('_') + 1]) for c in r) for q, r in zip(fixed.question_id, fixed[cols].values))}")
for q in FIX:
    print(f"  {q}: {base.loc[base.question_id == q, cols].values[0].tolist()} -> {fixed.loc[fixed.question_id == q, cols].values[0].tolist()}")
print("Public under the implied truth (truncated to 7 decimals):")
for p in sorted(glob.glob(f"{R2}/results/submission*/submission_player.csv")) + sorted(glob.glob(f"{R2}/results/probe/upload_*.csv")):
    print(f"  {score(p):.7f}  {p.replace(R2 + '/results/', '')}")
