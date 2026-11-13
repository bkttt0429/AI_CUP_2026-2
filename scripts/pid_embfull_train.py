#!/usr/bin/env python
"""The embfull component's networks (METHOD_NOTES 2.75): aicup_pid.embed.PlayerEmbedding on per-game rows of the UNPROJECTED
trunk statistics (scripts/pid_trunk_extract.py --full: `full` = whole-game mean / std of the board-pooled and played-point
vectors, 1,536 values; `fullp` = their means over three phases of the game, 2,304 values).

Training players: every player of folds 1-4 with rows (fold 0 is never read); --leave-fold v leaves fold v out (networks for
the simulated pools of that fold). Several seeds are written to one file: a list of PlayerEmbedding.state_dict().
  --phase          rows = full + fullp (3,840 values) instead of full alone (games whose part has no fullp are left out)
  --rows DIR ...   directories of parts
Run from the AI_CUP_2026-1 checkout (one GPU).
"""
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
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402


def load_rows(dirs, phase, keep=None):
    """{key: row float32} from the parts of the given directories; `keep`: only these keys."""
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            if "full" not in z.files or (phase and "fullp" not in z.files):
                continue
            keys = z["keys"].astype(str)
            X = np.hstack([z["full"], z["fullp"]]).astype(np.float32) if phase else z["full"].astype(np.float32)
            for k, r in zip(keys, X):
                if keep is None or k in keep:
                    out[k] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--phase", action="store_true")
    ap.add_argument("--leave-fold", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--d-out", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--support", type=int, default=10)
    ap.add_argument("--query-sizes", nargs="+", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--min-games", type=int, default=12)
    ap.add_argument("--max-games", type=int, default=0, help="at most this many games of a player (0: all), a fixed random choice")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    ok = (games.fold != 0) & ((games.fold != a.leave_fold) if a.leave_fold is not None else True)
    owner_of = dict(zip(games.game_key[ok], games.player_id[ok]))
    rows = load_rows(a.rows, a.phase, keep=owner_of)
    by = {}
    for k in sorted(rows):
        by.setdefault(owner_of[k], []).append(k)
    rng = np.random.default_rng(0)
    if a.max_games:
        by = {p: (sorted(rng.permutation(ks)[:a.max_games].tolist()) if len(ks) > a.max_games else ks) for p, ks in by.items()}
    rows_by = {p: np.stack([rows[k] for k in ks]) for p, ks in sorted(by.items()) if len(ks) >= a.min_games}
    d = next(iter(rows_by.values())).shape[1]
    print(f"rows of {sum(len(x) for x in rows_by.values())} games, {len(rows_by)} players with >= {a.min_games} games (folds 1-4{'' if a.leave_fold is None else f' without fold {a.leave_fold}'}), {d} values per game ({time.time() - t0:.0f} s)", flush=True)
    del rows
    states = []
    for s in a.seeds:
        m = PlayerEmbedding(hidden=a.hidden, d_out=a.d_out, dropout=a.dropout, steps=a.steps, support=a.support, query_sizes=tuple(a.query_sizes), lr=a.lr, p_aug=0.0, min_games=a.min_games, seed=s)
        m.fit(rows_by)
        states.append(m.state_dict())
        print(f"  seed {s}: episode top-1 at the end {m.history[-1]['acc']:.3f}, loss {m.history[-1]['loss']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states, "phase": bool(a.phase), "rows": a.rows, "leave_fold": a.leave_fold, "players": len(rows_by), "d_in": d}, a.out)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
