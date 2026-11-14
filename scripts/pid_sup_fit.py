#!/usr/bin/env python
"""The supervised projection of the trunk vectors for probe B (METHOD_NOTES 2.88): fitted on per-game rows of unprojected
statistics (scripts/pid_trunk_extract.py --full) of the training players of --fit-folds only (the players of the folds whose
simulated pools score the probe never enter; fold 0 is never read).
  pid_sup_fit.py --rows DIR [DIR ...] --fit-folds 3 4 --k 64 --out proj.npz
Output: mu [768], sd [768], W [768, k] with feature = ((x - mu) / sd) @ W for x = [pooled vector, played-point vector].
  --blocks g x e   the projection over [pooled, played-point, policy-EXPECTED played-point] means instead (METHOD_NOTES 2.102;
                   rows of scripts/pid_pol_extract.py, which hold `pole` next to `full`); the output then carries `blocks`,
                   and scripts/pid_sup_ply_extract.py computes the expected vector when it sees "e" there."""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
import pid_sup  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--fit-folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", nargs="+", default=None, choices=("g", "x", "e"), help="blocks of per-game means to project jointly (default: g x through pid_sup.fit)")
    a = ap.parse_args(argv)
    t0 = time.time()
    assert 0 not in a.fit_folds, "fold 0 is never read"
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    owner_of = dict(zip(games.game_key[games.fold.isin(a.fit_folds)], games.player_id[games.fold.isin(a.fit_folds)]))
    rows, own = {}, {}
    for d in a.rows:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if "full" not in z.files or (a.blocks and "e" in a.blocks and "pole" not in z.files):
                continue
            C_ = z["full"].shape[1] // 4
            if a.blocks:  # the chosen mean blocks side by side
                src = {"g": z["full"][:, 0:C_], "x": z["full"][:, 2 * C_:3 * C_]}
                if "e" in a.blocks:
                    src["e"] = z["pole"][:, 0:C_]
                mat = np.hstack([src[b] for b in a.blocks])
            else:
                mat = z["full"]
            for k, r in zip(z["keys"].astype(str), mat):
                if k in owner_of and k not in rows:
                    rows[k] = r.astype(np.float32)
    keys = sorted(rows)
    owners = np.array([owner_of[k] for k in keys])
    u, c = np.unique(owners, return_counts=True)
    keep = np.isin(owners, u[c >= a.min_games])
    X, owners = np.stack([rows[k] for k in keys])[keep], owners[keep]
    mu, sd, W, lam = pid_sup.fit_means(X, owners, a.k) if a.blocks else pid_sup.fit(X, owners, a.k)
    print(f"{len(X)} games of {len(set(owners.tolist()))} players of folds {a.fit_folds}; between / within ratio of directions 1, 8, 16, 32, {a.k}: "
          + " ".join(f"{lam[i - 1]:.2f}" for i in (1, 8, 16, 32, a.k)) + f" ({time.time() - t0:.0f} s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, mu=mu, sd=sd, W=W, ratio=lam.astype(np.float32), fit_folds=np.array(a.fit_folds), games=np.array(len(X)), **({"blocks": np.array(a.blocks)} if a.blocks else {}))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
