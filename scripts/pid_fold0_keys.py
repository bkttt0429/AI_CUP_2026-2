#!/usr/bin/env python
"""Game keys of every training game of the players of FOLD 0 (METHOD_NOTES 2.201), for scripts/pid_sup_ply_extract.py --keys.

Fold 0 was never read by this repository until the user's consent of 2026-11-18 00:29 ("同意用 fold 0"); it is used for the
training of the sequence networks only (--all-folds), never for any validation tool. The counterpart for folds 1-4 is
scripts/pid_fullrows_keys.py.   pid_fold0_keys.py <out.txt>"""
import os
import sys
from pathlib import Path

import pandas as pd

R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2 / "src"))
import aicup_pid  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402

games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
keys = set(games.game_key[games.fold == 0])
assert not (set(games.game_key[games.fold != 0]) & keys), "a game listed under a fold-0 player and another player"
tmp = Path(f"{sys.argv[1]}.{os.getpid()}.tmp")
tmp.write_text("\n".join(sorted(keys)) + "\n")
tmp.replace(sys.argv[1])  # several nodes may write it at the same time: the same content, complete or not at all
print(f"{len(keys)} training games of the {games.player_id[games.fold == 0].nunique()} players of fold 0 "
      f"(folds 1-4: {int((games.fold != 0).sum())} games)")
