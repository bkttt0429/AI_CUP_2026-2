"""How much the trunk rows move when a game is rotated / mirrored: rows of --sym (pid_trunk_extract --sym-seed) vs the
plain rows of the same keys. Prints, per column group, the size of the change relative to the spread across games, the
correlation, and whether the plain row of the same game is the nearest plain row to its rotated row (Mahalanobis-free,
column-standardised Euclidean)."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401
from aicup.scoring import load_games  # noqa: E402
from aicup_pid.symmetry import game_sym  # noqa: E402
from aicup_pid.trunkstyle import COLUMNS, load_store  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plain", required=True)
ap.add_argument("--sym", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
plain, sym = load_store(a.plain), load_store(a.sym)
keys = sorted(set(sym) & set(plain))
sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
ks = np.array([game_sym(k, sgfs[k], a.seed) for k in keys])
P, R = np.stack([plain[k] for k in keys]), np.stack([sym[k] for k in keys])
print(f"{len(keys)} games, {int((ks == 0).sum())} with the identity (incl. HD>=2), symmetry counts {np.bincount(ks, minlength=8).tolist()}")
ident = ks == 0
print(f"identity games: max |row difference| {np.abs(P[ident] - R[ident]).max():.2e} (0 = deterministic replay)")
m = ~ident
for name, cols in COLUMNS.items():
    p, r = P[m][:, cols], R[m][:, cols]
    sd = P[:, cols].std(axis=0) + 1e-9
    z_p, z_r = (p - P[:, cols].mean(axis=0)) / sd, (r - P[:, cols].mean(axis=0)) / sd
    rel = np.sqrt(((z_r - z_p) ** 2).sum(axis=1)) / np.sqrt((z_p ** 2).sum(axis=1))
    corr = np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(z_p, z_r)])
    Zall = (P[:, cols] - P[:, cols].mean(axis=0)) / sd
    sub = np.arange(min(1000, len(z_r)))
    d = ((z_r[sub, None, :] - Zall[None, :, :]) ** 2).sum(axis=2)
    own = np.flatnonzero(m)[sub]
    nn = (d.argmin(axis=1) == own).mean()
    print(f"{name:7s} ({len(cols)} cols): change / own deviation median {np.median(rel):.3f} (p90 {np.quantile(rel, 0.9):.3f}); "
          f"per-game corr {corr:.4f}; own plain row is the nearest of {len(P)}: {nn:.3f}")
