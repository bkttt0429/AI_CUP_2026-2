#!/usr/bin/env python
"""Player embedding for the v2 pools (METHOD_NOTES 2.29): the B8 network (aicup_pid.embed) trained on training players' rows of
BOTH colours, in the v2 geometry: support of 10 games, query sets of 1 / 2 / 3 / 5 games, any colour mix. Rows: the rank
session's per-game-side trunk rows (aicup_pid.v2.load_side_rows; same features as data/trunk, both colours, about 855k games).
Writes emb2_p12.pt (trained on folds 3, 4: for simulated pools of folds 1, 2), emb2_p34.pt and emb2_final.pt (folds 1-4; no
official game enters any training; fold 0 never used)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.v2 import load_phase_rows, load_side_rows, training_games  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "embed"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--support", type=int, default=10)
    ap.add_argument("--tag", default="emb2")
    ap.add_argument("--phase-dir", default=None, help="experiments/pid_v2_phase_rows.py output: train on the 640-dim rows with phase means (use --tag emb3)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="seed 0 keeps the plain file names; seed k > 0 writes <tag>_<name>_s<k>.pt")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = load_phase_rows(AICUP1, a.phase_dir) if a.phase_dir else load_side_rows(AICUP1)
    games = training_games(aicup_pid.TRAIN_DIR)
    folds = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    g = games.merge(folds[["player_id", "fold"]], on="player_id")
    g = g[g.game_key.isin(rows.keys())]
    print(f"{len(rows)} rows of {len(next(iter(rows.values())))} dims, {len(g)} games of {g.player_id.nunique()} players with a fold; colours {g.color.value_counts().to_dict()} ({time.time() - t0:.0f} s)", flush=True)
    for name, fs in (("p12", (3, 4)), ("p34", (1, 2)), ("final", (1, 2, 3, 4))):
        sub = g[g.fold.isin(fs)]
        by = {p: np.stack([rows[k] for k in ks]) for p, ks in sub.groupby("player_id")["game_key"]}
        for seed in a.seeds:
            m = PlayerEmbedding(steps=a.steps, support=a.support, query_sizes=(1, 2, 3, 5), p_aug=0.0, min_games=12, seed=seed)
            t1 = time.time()
            m.fit(by, None, log=None)
            dest = Path(a.out_dir) / (f"{a.tag}_{name}.pt" if seed == 0 else f"{a.tag}_{name}_s{seed}.pt")
            torch.save(m.state_dict(), dest)
            print(f"{name} seed {seed}: folds {fs}, {m.players_trained} players, final loss {m.history[-1]['loss']:.3f}, {time.time() - t1:.0f} s -> {dest}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
