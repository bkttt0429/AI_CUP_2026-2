#!/usr/bin/env python
"""METHOD_NOTES 2.179: how wrong is the window input's board? (measurement, nothing is selected)

scripts/pid_win.py builds the board of every ply from the moves alone: the last stone played on a point stays there. Captured
stones are therefore still shown, and setup (handicap) stones are missing. This script replays a sample of competition games
exactly (the KataGo board of scripts/trunk_extract.py) and counts, for every non-pass ply, the cells of the N x N window
around the played point where the two boards differ - before deciding whether an exact board is worth building
(notes 2.170, point 4).
  python -I scripts/pid_probe_captures.py --keys /work/u2507115/pid_v2/layerrows/keys.txt --games 3000 --workers 10 --out results/final/v2_probe_captures.json
Run from the AI_CUP_2026-1 checkout (CPU only). The key list has no game of fold 0.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pid_trunk_extract as PT  # noqa: E402  (puts both checkouts on sys.path)
import aicup_pid  # noqa: E402
from aicup.scoring import load_games  # noqa: E402

te = PT.te
SIZES = (5, 9, 13)
BINS = ((0, 60), (60, 120), (120, 180), (180, 400))


def one(task):
    key, sgf, _color = task
    stones, moves = te.setup_and_moves(sgf)
    gs = te.start_state(stones, te.RULES)
    b = gs.board
    locs = np.array([b.loc(x, y) for y in range(19) for x in range(19)])
    approx = np.zeros(361, np.int8)  # 0 empty, 1 black, 2 white: the board of pid_win.py (moves only, nothing removed)
    rows = []  # per non-pass ply: t, differing cells per window size, differing cells on the whole board
    captured = 0
    for t, (c, x, y) in enumerate(moves):
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE
        if x is not None:
            exact = np.asarray(b.board)[locs].astype(np.int8)
            exact = np.where(exact == te.Board.BLACK, 1, np.where(exact == te.Board.WHITE, 2, 0)).astype(np.int8)
            d = (exact != approx).reshape(19, 19)
            r = [t]
            for n in SIZES:
                h = n // 2
                r.append(int(d[max(0, y - h):y + h + 1, max(0, x - h):x + h + 1].sum()))
            r.append(int(d.sum()))
            rows.append(r)
        loc = te.Board.PASS_LOC if x is None else b.loc(x, y)
        before = int((np.asarray(b.board)[locs] != te.Board.EMPTY).sum())
        gs.play(pla, loc)
        b = gs.board
        after = int((np.asarray(b.board)[locs] != te.Board.EMPTY).sum())
        if x is not None:
            captured += before + 1 - after
            approx[y * 19 + x] = 1 if c == "B" else 2
    return key, len(stones), captured, np.asarray(rows, np.int32).reshape(-1, len(SIZES) + 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    keys = sorted(set(PT.read_keys(a.keys)))
    rng = np.random.default_rng(a.seed)
    keys = [keys[i] for i in sorted(rng.choice(len(keys), min(a.games, len(keys)), replace=False))]
    sgf = load_games(keys, aicup_pid.TRAIN_DIR)
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = list(pool.imap_unordered(one, [(k, *sgf[k]) for k in keys], chunksize=8))
    rows = np.concatenate([r[3] for r in res])
    n_setup = sum(1 for r in res if r[1] > 0)
    caps = np.array([r[2] for r in res])
    out = {"games": len(res), "plies": int(len(rows)), "games_with_setup_stones": n_setup, "games_with_a_capture": int((caps > 0).sum()),
           "captured_stones_per_game": float(caps.mean()), "by_window": {}, "by_ply": {}}
    print(f"{len(res)} games, {len(rows)} non-pass plies ({time.time() - t0:.0f} s); games with setup stones {n_setup}; "
          f"games with a capture {(caps > 0).mean():.3f}, captured stones per game mean {caps.mean():.1f} / median {np.median(caps):.0f}")
    for i, n in enumerate(SIZES, 1):
        d = rows[:, i]
        out["by_window"][str(n)] = {"plies_with_a_wrong_cell": float((d > 0).mean()), "wrong_cells_per_window": float(d.mean()),
                                    "wrong_cells_given_any": float(d[d > 0].mean()) if (d > 0).any() else 0.0, "plies_with_3_or_more": float((d >= 3).mean())}
        print(f"window {n:2d} x {n:2d}: plies with at least one wrong cell {(d > 0).mean():.3f}, with three or more {(d >= 3).mean():.3f}; "
              f"wrong cells per window {d.mean():.3f} of {n * n} ({d.mean() / (n * n):.4f}); given any: {d[d > 0].mean() if (d > 0).any() else 0:.2f}")
    for lo, hi in BINS:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if not m.any():
            continue
        d = rows[m, 2]
        out["by_ply"][f"{lo}-{hi}"] = {"plies": int(m.sum()), "plies_with_a_wrong_cell_9": float((d > 0).mean()), "wrong_cells_per_window_9": float(d.mean()),
                                       "wrong_cells_on_board": float(rows[m, -1].mean())}
        print(f"plies {lo:3d}-{hi:3d}: {m.sum():7d} plies; 9 x 9 window with a wrong cell {(d > 0).mean():.3f}, wrong cells per window {d.mean():.3f}; wrong cells on the whole board {rows[m, -1].mean():.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
