#!/usr/bin/env python
"""Per-game-side rows WITH phase means for the training games that have stored per-ply vectors (METHOD_NOTES 2.33): the 256-dim
row (mean, sd of the owner's plies) and the three phase means (plies t < 60, 60 <= t < 150, t >= 150; 128 dims each), computed
with the rank session's own function (logs/nano4/m84_rows_extract.py side_rows) from results/trunk_vec*, so that they match
results/m84_rowsp (which covers the other 595k games). Read-only on their files; run from the AI_CUP_2026-1 checkout.
Output: <out>/stored_f<fold>.npz with key, T [n, 256] float32 (their layout), PT [n, 384] float16."""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AICUP1 = Path.cwd()
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "scripts"), str(AICUP1 / "logs/nano4")]
DIRS = (AICUP1 / "results/trunk_vec", AICUP1 / "results/trunk_vec_x40", AICUP1 / "results/trunk_vec_x80")


def one(csv_path):
    from m84_rows_extract import side_rows
    csv_path = Path(csv_path)
    tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:
        X = np.hstack([z["g"].astype(np.float32), z["l"].astype(np.float32)])
        out, ph, cnt = side_rows(X, z["game_row"], z["target"], z["t"], len(tab))
    return tab["game_key"].to_numpy().astype(str), out[0], ph[0], cnt[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/work/u2507115/pid_v2/phase_rows")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        files = [x for d in DIRS for x in sorted((d / f"fold{f}").glob("games_s*_p*.csv"))]
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(one, files, chunksize=4)
        key, T, PT, n = (np.concatenate([r[i] for r in res]) for i in range(4))
        keep = n > 0
        np.savez(out / f"stored_f{f}.npz", key=key[keep], T=T[keep].astype(np.float32), PT=PT[keep].astype(np.float16))
        print(f"fold {f}: {len(files)} parts, {int(keep.sum())} games ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
