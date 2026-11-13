#!/usr/bin/env python
"""Per-ply trunk vectors of the external games chosen by scripts/ext_fox_select.py (METHOD_NOTES 2.46), in exactly the form
of the rank session's M90 parts (AI_CUP_2026-1 logs/nano4/m90_ply_extract.py: same network, replay, projection, the same
quantisation statistics results/m90/glstats.npz), so that the sequence identity networks train on them next to the
competition's games. That session's modules are imported read-only; nothing of theirs is written.
Parts <out>/ply_x_s<shard>of<n>_p<k>.npz: key, LEN, SUBJ, POS, MOV, GLQ (the first 256 plies). Resumable per part.
Run from the AI_CUP_2026-1 checkout:
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I <this> --select <select.csv> --out <dir> --workers 10
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_extract as te  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402

STATS, T_MAX, Q = ROOT / "results/m90/glstats.npz", 256, 20.0


def safe_replay(task):
    try:
        return te.replay(task)
    except Exception as e:  # a record the replay cannot follow: left out
        return (task[0], None, repr(e)[:80])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--select", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="smoke run: only this many games")
    ap.add_argument("--start", type=int, default=0, help="rows [start, stop) of the selection (applied before the shards), for a selection extracted in stages")
    ap.add_argument("--stop", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    csv.field_size_limit(10 ** 9)
    with open(a.select, newline="") as fh:
        tasks = [(r["game_key"], r["sgf_content"], r["color"]) for r in csv.DictReader(fh)]
    tasks = tasks[a.start:(a.stop or len(tasks))][a.shard::a.nshards]
    tasks = tasks[:a.limit] if a.limit else tasks
    st = np.load(STATS)
    mu, sd = st["mu"], st["sd"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} games in this shard ({time.time() - t0:.0f} s)", flush=True)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        nrows, proj, plies, skipped, t1 = neutral_rows(), None, 0, 0, time.time()
        for k, start in enumerate(range(0, len(tasks), a.part)):
            path = out / f"ply_x_s{a.shard}of{a.nshards}_p{k:05d}.npz"
            if path.exists():
                continue
            res = list(pool.map(safe_replay, tasks[start:start + a.part], chunksize=2))
            done = [d for d in res if d[1] is not None and len(d[3]) > 0]
            skipped += len(res) - len(done)
            if not done:
                continue
            if proj is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)
                proj = te.projections(trunk.shape[1], trunk.device)
            lens = np.array([len(d[3]) for d in done])
            bins, globs = np.concatenate([d[1] for d in done]), np.concatenate([d[2] for d in done])
            pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
            g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch)
            keep = np.concatenate([np.arange(n) for n in lens]) < T_MAX
            q = np.clip(np.rint((np.hstack([g, l])[keep].astype(np.float32) - mu) / sd * Q), -127, 127).astype(np.int8)
            subj = np.array([int(d[4][d[5]][0]) if d[5].any() else 0 for d in done], np.int8)
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, key=np.array([d[0] for d in done]), LEN=np.minimum(lens, T_MAX).astype(np.int16), SUBJ=subj, POS=pos[keep].astype(np.int16), MOV=mover[keep].astype(np.int8), GLQ=q)
            os.replace(tmp, path)
            plies += int(lens.sum())
            if k % 20 == 0:
                print(f"part {k + 1}/{(len(tasks) + a.part - 1) // a.part}: {plies} plies, {plies / (time.time() - t1):.0f} plies/s, skipped {skipped}", flush=True)
        print(f"done: {len(tasks)} games, {plies} plies, skipped {skipped}, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
