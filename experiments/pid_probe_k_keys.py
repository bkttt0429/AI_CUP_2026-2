#!/usr/bin/env python
"""Probe K (METHOD_NOTES 2.119): the probe games whose player is in the given folds (for fitting a projection; never fold 0).
  python -I <this> --keys /work/u2507115/pid_v2/probeb/keys.txt --folds 3 4 --out /work/u2507115/pid_v2/probek/keys_fit.txt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    assert 0 not in a.folds, "fold 0 is not used"
    keys = set(Path(a.keys).read_text().split())
    g = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    fit = sorted(g[g.fold.isin(a.folds) & g.game_key.isin(keys)].game_key)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(fit) + "\n")
    print(f"{len(fit)} of {len(keys)} probe games belong to players of folds {a.folds}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
