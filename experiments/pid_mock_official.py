#!/usr/bin/env python
"""Official-format files from a simulated pool, one question per candidate (the 11/04 rehearsal; line B3's evaluation set).

Each --pools entry (folds) becomes one pool: the first is written with the pub_ prefix, the second with priv_
(aicup_pid.pools.make_pool_one_per_player: up to 400 players with >= 105 Black games, 100 candidate games each, exactly one
question per player, sizes 20/15/10/5 handed out largest first up to --per-size each). Files in --out-dir:
player_identification_candidates_<public|private>.csv (game_id, player_id, sgf_content, color), player_identification_test_<public|private>.csv
(question_id, num_games, sgf_1..20, color_1..20), truth_<public|private>.csv (question_id, player_id, train_player_id, rank, n_games)
and mock_info.json. Fold 0 is refused.

  python -I scripts/pid_mock_official.py --pools 1,2 3,4 --out-dir data/mock_official --seed 0
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.submission import MAX_GAMES  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from pid_cv import parse_pools  # noqa: E402

NAMES = {"pub_": "public", "priv_": "private"}


def write_pool(prefix, folds, games, folds_df, a, out_dir, log=print):
    t0 = time.time()
    cands, qs, info = pools.make_pool_one_per_player(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, per_size=a.per_size,
                                                     seed=a.seed, pool_id=prefix.rstrip("_"))
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
    sgfs = load_games(keys, aicup_pid.TRAIN_DIR)
    name = NAMES[prefix]
    cand_rows = [{"game_id": f"{prefix}g_{i + 1:06d}", "player_id": f"{prefix}p_{r.player_id}", "sgf_content": sgfs[r.game_key][0], "color": "B"}
                 for i, r in enumerate(cands.itertuples())]
    q_rows, truth_rows = [], []
    for r in qs.itertuples():
        row = {"question_id": r.question_id, "num_games": r.n_games}
        for j in range(MAX_GAMES):
            row[f"sgf_{j + 1}"] = sgfs[r.game_keys[j]][0] if j < r.n_games else ""
            row[f"color_{j + 1}"] = "B" if j < r.n_games else ""
        q_rows.append(row)
        truth_rows.append({"question_id": r.question_id, "player_id": f"{prefix}p_{r.player_id}", "train_player_id": r.player_id, "rank": r.rank, "n_games": r.n_games})
    pd.DataFrame(cand_rows, columns=["game_id", "player_id", "sgf_content", "color"]).to_csv(out_dir / f"player_identification_candidates_{name}.csv", index=False)
    pd.DataFrame(q_rows).to_csv(out_dir / f"player_identification_test_{name}.csv", index=False)
    pd.DataFrame(truth_rows, columns=["question_id", "player_id", "train_player_id", "rank", "n_games"]).to_csv(out_dir / f"truth_{name}.csv", index=False)
    info = {**info, "prefix": prefix, "seconds": time.time() - t0}
    log(f"{name}: {info['players']} candidates ({info['eligible']} eligible), {len(cands)} candidate games, {len(qs)} questions {info['questions_by_n']} in {info['seconds']:.0f} s", flush=True)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="folds of the public pool, then of the private pool")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--per-size", type=int, default=100)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    a = ap.parse_args(argv)
    try:
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a mock pool")
        if len(specs) > len(pools.PREFIXES):
            raise ValueError(f"at most {len(pools.PREFIXES)} pools (public, private)")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        infos = [write_pool(prefix, folds, games, folds_df, a, out_dir) for prefix, folds in zip(pools.PREFIXES, specs)]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    doc = {"pools": infos, "seed": a.seed, "n_players": a.n_players, "cand_games": a.cand_games, "per_size": a.per_size, **pid_environment("cpu")}
    (out_dir / "mock_info.json").write_text(to_json(doc, indent=1) + "\n")
    print(f"wrote {out_dir}/mock_info.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
