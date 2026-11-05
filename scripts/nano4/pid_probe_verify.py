"""Which file got which Public score? Check the probe files against the base, then test every reported score as the
base of upload_2 / upload_1 under the documented per-question metric e^-(r-1)/400, and check the transitions between
the assigned versions against the loss decompositions."""
import hashlib, itertools, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid  # noqa: F401
from aicup_pid import probe

R2 = "/home/u2507115/BK_AICUP_2026-2"
cols = ["top_1", "top_2", "top_3", "top_4", "top_5"]
F = {"f3_assign": "submission_f3_assign/submission_player.csv", "f3o_assign": "submission_f3o_assign/submission_player.csv",
     "f3oref_assign": "submission_f3oref_assign/submission_player.csv", "f5r4_assign": "submission_f5r4_assign/submission_player.csv",
     "upload_1": "probe/upload_1.csv", "upload_2": "probe/upload_2.csv"}
D = {k: pd.read_csv(f"{R2}/results/{v}", dtype=str).set_index("question_id") for k, v in F.items()}
print("== file hashes")
for k, v in F.items():
    print(f"  {k:14s} {hashlib.sha256(open(f'{R2}/results/{v}', 'rb').read()).hexdigest()[:16]}")
print("== rows differing (pub / priv), any rank / Top-1")
for a, b in itertools.combinations(F, 2):
    A, B = D[a], D[b]
    any_ = (A[cols].values != B[cols].values).any(axis=1); t1 = (A.top_1 != B.top_1).values
    pub = A.index.str.startswith("pub_")
    print(f"  {a:14s} vs {b:14s}: any {int(any_[pub].sum()):3d}/{int(any_[~pub].sum()):3d}   top1 {int(t1[pub].sum()):3d}/{int(t1[~pub].sum()):3d}")

S = {"0.9487291": 0.9487291, "0.9573466": 0.9573466, "0.9704912": 0.9704912, "0.9873479": 0.9873479, "0.9658747": 0.9658747,
     "0.9833134": 0.9833134, "0.9737858": 0.9737858, "0.9937580": 0.9937580}
c = probe.deltas(probe.CYCLE)
print("== upload_2 (only pub_q_0303 cycled) = 0.9817331: implied change for each candidate base (x400)")
for k, v in S.items():
    d = (0.9817331 - v) * 400
    hit = [t for t, x in zip(probe.POSITIONS, c) if abs(x - d) < 1e-4]
    print(f"  base {k}: delta {d:+9.5f}  -> " + (f"q0303 truth at rank {hit[0]} (exact)" if hit else "impossible for one cycled question"))
print("== upload_1 (20 cycled) = 0.9562058 decoded against base 0.9833134")
d1 = (0.9562058 - 0.9833134) * 400
b = probe.bounds_from_score(0.9833134)
m = probe.decode_histogram(d1, 20, bounds=b)
print(f"  delta {d1:+.5f}; bounds {b}; matching histograms (rank 1..5, out): {m}")
for h in m:
    print(f"  predicted {probe.histogram_sum(h):+.6f} vs observed {d1:+.6f} (diff {probe.histogram_sum(h) - d1:+.1e})")
print("== loss decompositions (truth at rank 2, 3, 4, 5, outside)")
for k in ("0.9833134", "0.9873479", "0.9937580"):
    print(f"  {k}: {probe.deficit_decompositions(S[k])}")
gain = {2: 1 - np.exp(-1), 3: 1 - np.exp(-2), 4: 1 - np.exp(-3), 5: 1 - np.exp(-4), 0: 1.0}
for lo, hi in (("0.9833134", "0.9873479"), ("0.9833134", "0.9937580"), ("0.9873479", "0.9937580")):
    need = (S[hi] - S[lo]) * 400
    for x in probe.deficit_decompositions(S[lo]):
        for y in probe.deficit_decompositions(S[hi]):
            if all(yi <= xi for xi, yi in zip(x, y)):
                g = sum((xi - yi) * gain[r] for xi, yi, r in zip(x, y, (2, 3, 4, 5, 0)))
                if abs(g - need) < 1e-4:
                    print(f"  {lo} {x} -> {hi} {y}: fixes only, gain {g:.5f} = observed {need:.5f}")
