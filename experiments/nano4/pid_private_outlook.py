"""Private outlook: (1) per-pool distribution of the simulated score and wrong count of fusion3o_ref+assign (final CV e),
(2) structure of the now-known Public truth (does the question order map to the candidate order in a way that would
transfer to Private?), (3) Public vs Private difficulty indicators of the f3oref files."""
import json, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R2 = "/home/u2507115/BK_AICUP_2026-2"
cv = json.load(open(f"{R2}/results/final/final_cv_e.json"))
sc = [p["summary"]["fusion3o_ref+assign"]["score"] for p in cv["pools"]]
wr = [p["summary"]["fusion3o_ref+assign"]["wrong"] for p in cv["pools"]]
print("simulated one-per-player pools (400 questions each), fusion3o_ref+assign:")
print("  scores", [round(x, 4) for x in sorted(sc)], "| wrong Top-1", sorted(wr), "| mean", round(float(np.mean(sc)), 4))
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
fix = pd.read_csv(f"{R2}/results/probe/upload_fix.csv", dtype=str)
pub = fix[fix.question_id.str.startswith("pub_")]
qi = pub.question_id.str.extract(r"(\d+)$")[0].astype(int).values
pi = pub.top_1.str.extract(r"(\d+)$")[0].astype(int).values
rho, p = spearmanr(qi, pi)
print(f"Public truth: spearman(question index, player index) = {rho:+.3f} (p={p:.1g}); equal index {int((qi == pi).sum())}")
best = max(((a, b, int(((a * qi + b) % 400 == (pi - 1) % 400).sum())) for a in range(1, 400) for b in range(400)), key=lambda t: t[2])
print(f"  best affine map player = a*question + b mod 400: a={best[0]} b={best[1]} matches {best[2]}/400")
q = pd.read_csv(f"{R2}/data/tests/player_identification_test_public.csv", dtype=str, usecols=["question_id", "num_games"]).set_index("question_id")
n = q.loc[pub.question_id, "num_games"].astype(int).values
for k in (5, 10, 15, 20):
    m = n == k
    print(f"  n={k}: player index range {pi[m].min()}-{pi[m].max()}, mean {pi[m].mean():.0f}")
for name in ("submission_f3oref", "submission_f3oref_assign"):
    r = json.load(open(f"{R2}/results/{name}/submission_player.report.json"))
    for pool in r["pools"]:
        a = pool["assign"]
        print(f"  {name} {pool['questions'].split('/')[-1]}: distinct Top-1 {a['top1_collisions'].get('distinct_top1')}, assignment changed {a['n_top1_changed']}, median margin {pool['median_top1_margin']:.2f}")
