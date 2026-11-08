"""Data audit across the two competitions (no player identification): which whole games (symmetry-canonical hash,
aicup_pid.dedup.game_hash) are shared between the rank-prediction test files, the void 10-30 player-identification
files and the 1M training games. Also reports colour mix and question sizes of the rank test."""
import collections
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
from aicup_pid.dedup import game_hash  # noqa: E402

R1 = Path.home() / "BK_AICUP_2026-1" / "data" / "tests"
PT = ROOT / "data" / "tests"


def questions(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        for i in range(1, int(d["num_games"]) + 1):
            out.append((d["question_id"], d[f"sgf_{i}"], d[f"color_{i}"]))
    return df, out


sets = {}
for pool in ("public", "private"):
    df, games = questions(R1 / f"rank_prediction_test_{pool}.csv")
    sets[f"rank_{pool}"] = {game_hash(parse_moves(s)): (q, c) for q, s, c in games}
    print(f"rank {pool}: {len(df)} questions, {len(games)} games, n mix {dict(sorted(collections.Counter(df.num_games.astype(int)).items()))}, "
          f"colours {dict(collections.Counter(c for *_, c in games))}")
    _, qg = questions(PT / f"player_identification_test_{pool}.csv")
    sets[f"pid_q_{pool}"] = {game_hash(parse_moves(s)): (q, c) for q, s, c in qg}
    cand = pd.read_csv(PT / f"player_identification_candidates_{pool}.csv", dtype=str, keep_default_na=False)
    sets[f"pid_c_{pool}"] = {game_hash(parse_moves(s)): (p, "B") for p, s in zip(cand.player_id, cand.sgf_content)}
print({k: len(v) for k, v in sets.items()})
names = list(sets)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        shared = set(sets[a]) & set(sets[b])
        if shared:
            same_colour = sum(sets[a][h][1] == sets[b][h][1] for h in shared)
            print(f"SHARED {a} x {b}: {len(shared)} games ({same_colour} with the same colour role)")
# training (1M games, both colours)
tr = set()
for f in sorted((Path(aicup_pid.TRAIN_DIR)).glob("train_*.csv")):
    for chunk in pd.read_csv(f, usecols=["sgf_content"], dtype=str, chunksize=50000):
        tr.update(game_hash(parse_moves(s)) for s in chunk.sgf_content)
print(f"training: {len(tr)} distinct games")
for a in names:
    print(f"{a} x training: {len(set(sets[a]) & tr)} shared")
