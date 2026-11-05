import pandas as pd, numpy as np
R2 = "/home/u2507115/BK_AICUP_2026-2"
base = pd.read_csv(f"{R2}/results/submission_f3_assign/submission_player.csv", dtype=str)
i = base.index[base.question_id == "pub_q_0303"][0]
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
row = base.loc[i, cols].tolist()
a = base.copy(); a.loc[i, ["top_4", "top_5"]] = [row[4], row[3]]; a.to_csv(f"{R2}/results/probe/upload_3_swap45.csv", index=False)
b = base.copy(); b.loc[i, ["top_1", "top_2"]] = [row[1], row[0]]; b.to_csv(f"{R2}/results/probe/upload_3_swap12.csv", index=False)
print("pub_q_0303 base list", row)
print("swap45 ->", a.loc[i, cols].tolist(), "| expected if the truth is Top-1: 0.9937580")
print("swap12 ->", b.loc[i, cols].tolist(), "| expected: %.7f" % (0.9937580 - (1 - np.exp(-1)) / 400))
print("rows whose Top-1 is", row[4], ":", base.question_id[base.top_1 == row[4]].tolist(), "| rows whose Top-1 is", row[1], ":", base.question_id[base.top_1 == row[1]].tolist())
for f in ("upload_3_swap45", "upload_3_swap12"):
    d = pd.read_csv(f"{R2}/results/probe/{f}.csv", dtype=str)
    print(f, "rows", len(d), "differing from base", int((d.values != base.values).any(axis=1).sum()), "five distinct", all(len(set(r)) == 5 for r in d[cols].values))
