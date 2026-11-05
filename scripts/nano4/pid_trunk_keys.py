import sys
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid
import pandas as pd
from aicup_pid import pools
from aicup_pid.paths import FOLDS_CSV
games = pools.black_games(aicup_pid.TRAIN_DIR)
folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
keys = set()
for f in ([1, 2], [3, 4]):
    c, q, _ = pools.make_pool(games, folds_df, f, seed=0, pool_id="x")
    k = set(c.game_key) | {x for ks in q.game_keys for x in ks}
    print(f, "make_pool seed 0:", len(k)); keys |= k
    for s in range(5):
        c, q, _ = pools.make_pool_one_per_player(games, folds_df, f, seed=s, pool_id="x")
        k = set(c.game_key) | {x for ks in q.game_keys for x in ks}
        keys |= k
    print(f, "union so far:", len(keys))
print("total training games to extract:", len(keys))
import pathlib; pathlib.Path("/home/u2507115/BK_AICUP_2026-2/data/trunk").mkdir(parents=True, exist_ok=True); pathlib.Path("/home/u2507115/BK_AICUP_2026-2/data/trunk/train_keys.txt").write_text("\n".join(sorted(keys)) + "\n"); print("wrote data/trunk/train_keys.txt")
