#!/usr/bin/env python
"""Probe P (METHOD_NOTES 2.102): features of a move RELATIVE TO WHAT THE NETWORK EXPECTED at that position.

Every feature so far describes the position (board-pooled trunk vector) and the point that was played (trunk vector at that
point). Most of the variation of the played-point vector comes from the position, not from the player: in a fight everybody
plays contact moves. The same forward pass also gives the network's policy, so the played-point vector that an average player
would produce there is known: e = sum over points of policy(point) x trunk(point). The difference x - e is the player's choice
with the position's contribution removed (a background model, as in authorship attribution).

One pass over the games (scripts/pid_trunk_extract.py policy_vectors; the subject's plies only) writes
  <out>/rows/pol_s<shard>_p<k>.npz   per game: keys, n_target, full [4C] (mean / std of pooled and played-point vectors, as
                                     pid_trunk_extract.py --full), pole [2C] (mean / std of e over the non-pass plies),
                                     pold [C] (std of x - e), pols [36] (surprisal / rank / entropy statistics, whole game
                                     and three phases)
  <out>/ply_r/sup_s<shard>_p<k>.npz  per ply, the layout of scripts/pid_sup_ply_extract.py (keys, n, t, z float16):
                                     z = the EXISTING supervised projection (--proj: mu, sd, W over [pooled, played]) applied
                                     to [pooled, played - expected], 64 dims
  <out>/ply_fc/...                   z = [the existing supervised features (64), the expected vector through the played-point
                                     half of the same projection (64)]: a network can form the difference itself
  <out>/ply_s/...                    z = [surprisal, log(1 + rank), entropy, distance to the policy's top point]
Resumable per part (a part counts as done when its rows file exists; the per-ply files are written first).
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --proj proj.npz --out DIR --workers 10
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
import pid_trunk_extract as PT  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows  # noqa: E402
from aicup.scoring import load_games  # noqa: E402

te = PT.te
PHASES = 4
N_STAT = 9


def game_rows(X, E, S, pos, ts, lens):
    """pole [G, 2C], pold [G, C], pols [G, PHASES x N_STAT] (see the module docstring)."""
    C = X.shape[1]
    pole, pold, pols = np.zeros((len(lens), 2 * C), np.float32), np.zeros((len(lens), C), np.float32), np.zeros((len(lens), PHASES * N_STAT), np.float32)
    start = 0
    for i, (n, t) in enumerate(zip(lens, ts)):
        if n:
            sl = slice(start, start + n)
            on = pos[sl] < te.PASS_POS
            if on.any():
                e = E[sl][on]
                d = X[sl][on] - e
                pole[i, :C], pole[i, C:], pold[i] = e.mean(axis=0), e.std(axis=0), d.std(axis=0)
            sc = S[sl]
            for j, pm in enumerate((np.ones(n, bool), t < 60, (t >= 60) & (t < 150), t >= 150)):
                if pm.any():
                    q = sc[pm]
                    r = q[:, 1]
                    pols[i, N_STAT * j:N_STAT * (j + 1)] = [q[:, 0].mean(), q[:, 0].std(), (r == 0).mean(), (r < 3).mean(), (r < 10).mean(), np.log1p(r).mean(),
                                                            q[:, 2].mean(), q[:, 3].mean(), q[:, 4].mean()]
        start += n
    return pole, pold, pols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True, help="training game keys (text file)")
    ap.add_argument("--proj", default=None, help="the existing supervised projection (scripts/pid_sup_fit.py); not needed with --rows-only")
    ap.add_argument("--ckpt", default=None, help="another KataGo checkpoint than humanv0 (probe K, METHOD_NOTES 2.119: a strong network as the feature extractor)")
    ap.add_argument("--rows-only", action="store_true", help="write only the per-game rows (for fitting a projection), no per-ply stores")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-parts", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    if not a.rows_only:
        P = np.load(a.proj)
        mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)
        C = len(mu) // 2
    ckpt = a.ckpt or te.CKPT
    out = Path(a.out)
    for d in (("rows",) if a.rows_only else ("rows", "ply_r", "ply_fc", "ply_s")):
        (out / d).mkdir(parents=True, exist_ok=True)
    keys = sorted(set(PT.read_keys(a.keys)))
    mine = keys[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / "rows" / f"pol_s{a.shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    config = torch.load(str(ckpt), map_location="cpu", weights_only=a.ckpt is None)["config"]  # a foreign checkpoint holds more than tensors (as the rank session's experiments/ai_state_extract.py loads it)
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))

    def save(path, **arrays):
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    try:
        model, _ = load_humansl(str(ckpt), "cuda")
        model.eval()
        rows = neutral_rows()
        done_games = bad_all = 0
        for k, part in todo:
            sgf = load_games(part, aicup_pid.TRAIN_DIR)
            done = list(pool.imap(PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))
            some = [d for d in done if len(d[3])]
            bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])
            pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
            lens, ts = [len(d[3]) for d in done], [d[5] for d in done]
            G, X, E, S, bad = PT.policy_vectors(model, bins, globs, pos, mover, rows, a.batch)
            bad_all += bad
            if not a.rows_only:
                z_old = ((np.hstack([G, X]) - mu) / sd) @ W  # the existing supervised features
                c = (E / sd[C:]) @ W[C:]  # the expected vector through the played-point half (no mean: z_old - c is the projection of [pooled, played - expected])
            for arr in ((S,) if a.rows_only else (z_old, c, S)):
                if not np.isfinite(arr).all():
                    raise ValueError(f"part {k}: non-finite features")
            full = PT.aggregate_full(G, X, pos, lens)
            pole, pold, pols = game_rows(X, E, S, pos, ts, lens)
            head = dict(keys=np.array([d[0] for d in done]), n=np.array(lens, np.int32), t=np.concatenate(ts).astype(np.int16))
            name = f"sup_s{a.shard:03d}_p{k:04d}.npz"
            if not a.rows_only:
                save(out / "ply_r" / name, z=(z_old - c).astype(np.float16), **head)
                save(out / "ply_fc" / name, z=np.hstack([z_old, c]).astype(np.float16), **head)
                save(out / "ply_s" / name, z=np.stack([S[:, 0], np.log1p(S[:, 1]), S[:, 2], S[:, 4]], axis=1).astype(np.float16), **head)
            save(out / "rows" / f"pol_s{a.shard:03d}_p{k:04d}.npz", keys=head["keys"], n_target=np.array(lens, np.int16), full=full.astype(np.float16),
                 pole=pole.astype(np.float16), pold=pold.astype(np.float16), pols=pols)
            done_games += len(part)
            if (k % 10) == 0:
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s, played moves outside my legality mask {bad_all}", flush=True)
            if a.max_parts and done_games >= a.max_parts * a.part_size:
                break
    finally:
        pool.close()
        pool.join()
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min; played moves outside my legality mask {bad_all}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
