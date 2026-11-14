#!/usr/bin/env python
"""Per-ply supervised features of training games for probe B (METHOD_NOTES 2.88): the humanv0 trunk's board-pooled and
played-point vectors at the SUBJECT's plies (replay of scripts/pid_trunk_extract.py --fast), pushed through the projection of
scripts/pid_sup_fit.py instead of the random one. Parts <out>/sup_s<shard>_p<k>.npz: keys, n (plies per game), t (ply numbers),
z ([plies, K] float16, unstandardised). Resumable per part; shards as pid_trunk_extract.py.
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
import pid_trunk_extract as PT  # noqa: E402  (replay_target, full_vectors, read_keys; its imports put AI_CUP_2026-1 on the path)
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows  # noqa: E402
from aicup.scoring import load_games  # noqa: E402

te = PT.te


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--proj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-parts", type=int, default=0)
    ap.add_argument("--stats", default=None, help="npz (mean, sd) of pid_sup.write_stats: the parts are written as int8 `zq` = standardised value x 20 (half the size)")
    ap.add_argument("--skip-keys", default=None, help="a key list whose games are left out (already extracted elsewhere)")
    a = ap.parse_args(argv)
    t0 = time.time()
    P = np.load(a.proj)
    mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(PT.read_keys(a.keys)) - (set(PT.read_keys(a.skip_keys)) if a.skip_keys else set()))
    st = np.load(a.stats) if a.stats else None
    mine = keys[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"sup_s{a.shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do; projection to {W.shape[1]} dims", flush=True)
    if not todo:
        return 0
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        done_games = 0
        for k, part in todo:
            sgf = load_games(part, aicup_pid.TRAIN_DIR)
            done = list(pool.imap(PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))
            some = [d for d in done if len(d[3])]
            bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])
            pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
            fg, fl = PT.full_vectors(model, bins, globs, pos, mover, rows, a.batch)
            z = ((np.hstack([fg, fl]) - mu) / sd) @ W
            if not np.isfinite(z).all():
                raise ValueError(f"part {k}: non-finite features")
            path = out / f"sup_s{a.shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            feat = {"z": z.astype(np.float16)} if st is None else {"zq": np.clip(np.rint((z - st["mean"]) / st["sd"] * 20.0), -127, 127).astype(np.int8)}
            np.savez(tmp, keys=np.array([d[0] for d in done]), n=np.array([len(d[3]) for d in done], np.int32), t=np.concatenate([d[5] for d in done]).astype(np.int16), **feat)
            os.replace(tmp, path)
            done_games += len(part)
            if (k % 10) == 0:
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s", flush=True)
            if a.max_parts and done_games >= a.max_parts * a.part_size:
                break
    finally:
        pool.close()
        pool.join()
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
