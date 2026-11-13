#!/usr/bin/env python
"""Game keys of every training game of the players of folds 1-4 (never fold 0), for scripts/pid_trunk_extract.py --keys --full
(METHOD_NOTES 2.75). Keys that already have a row in the given directories are left out.   pid_fullrows_keys.py <out.txt> [rows dir ...]"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2 / "src"))
import aicup_pid  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402

games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
keys = set(games.game_key[games.fold != 0])
assert not (set(games.game_key[games.fold == 0]) & keys), "a game listed under a fold-0 player and another player"
have = set()
for d in sys.argv[2:]:
    for f in glob.glob(f"{d}/*.npz"):
        if not f.endswith(".tmp.npz"):
            have.update(np.load(f)["keys"].astype(str).tolist())
todo = sorted(keys - have)
tmp = Path(sys.argv[1] + ".tmp")
tmp.write_text("\n".join(todo) + "\n")
tmp.replace(sys.argv[1])  # the other nodes wait for this file: it appears complete or not at all
print(f"{len(keys)} training games of folds 1-4, {len(keys & have)} already have rows, {len(todo)} to extract")
