#!/usr/bin/env python
"""Per-ply supervised features of training games for probe B (METHOD_NOTES 2.88): the humanv0 trunk's board-pooled and
played-point vectors at the SUBJECT's plies (replay of scripts/pid_trunk_extract.py --fast), pushed through the projection of
scripts/pid_sup_fit.py instead of the random one. Parts <out>/sup_s<shard>_p<k>.npz: keys, n (plies per game), t (ply numbers),
z ([plies, K] float16, unstandardised). Resumable per part; shards as pid_trunk_extract.py.
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --proj proj.npz --out DIR --workers 10
--select: the external games of the selections of scripts/ext_fox_select.py instead (METHOD_NOTES 2.101; the subject is the
selection's colour; a record the replay cannot follow is stored without plies; a shard is the games whose key hashes to it, so
a process only holds its own share of the records).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import zlib
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
MAX_T = 0  # --max-plies (set before the worker pool is forked)


def replay_external(task):
    """PT.replay_target for an external record: one the replay cannot follow counts as a game without plies; only the plies
    with a number below MAX_T are kept (the sequence networks read the first 256 plies)."""
    try:
        d = PT.replay_target(task)
    except Exception:  # noqa: BLE001
        return task[0], None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)
    if MAX_T and len(d[3]):
        ok = d[5] < MAX_T
        if not ok.any():
            return d[0], None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)
        d = (d[0], d[1][ok], d[2][ok], d[3][ok], d[4][ok], d[5][ok])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", default=None, help="training game keys (text file)")
    ap.add_argument("--pools", default=None, help="instead of --keys: the official pools of this tests folder - every game of the added pools, and the games of the original pools that have per-ply vectors in --original-ply")
    ap.add_argument("--original-ply", default="/work/u2507115/pid_v2/ply_original")
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
    ap.add_argument("--select", nargs="+", default=None, help="instead of --keys: external games, the select.csv files of scripts/ext_fox_select.py (game_key, sgf_content, color)")
    ap.add_argument("--max-plies", type=int, default=0, help="with --select: keep only the subject's plies whose ply number is below this (0: all)")
    ap.add_argument("--scalars", action="store_true", help="append four policy scalars to every ply's features: surprisal, log(1 + rank), entropy, distance to the policy's top point (METHOD_NOTES 2.102)")
    ap.add_argument("--ckpt", default=None, help="another KataGo checkpoint than humanv0 (probe K, METHOD_NOTES 2.119)")
    ap.add_argument("--out2", default=None, help="with --scalars and float parts: also write the same parts WITHOUT the scalars here (two stores from one pass)")
    a = ap.parse_args(argv)
    global MAX_T
    MAX_T = a.max_plies if a.select else 0
    t0 = time.time()
    P = np.load(a.proj)
    mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)
    blocks = [str(b) for b in P["blocks"]] if "blocks" in P.files else ["g", "x"]  # "e": the policy-expected played-point vector
    use_pol = "e" in blocks or a.scalars
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks_all = None
    if a.pools:
        import glob
        tasks_all = PT.official2_tasks(a.pools)
        want = set()
        for f in glob.glob(f"{a.original_ply}/ply_*.npz"):
            if not f.endswith(".tmp.npz"):
                want.update(np.load(f)["key"].astype(str).tolist())
        old = PT.official_tasks(a.pools)
        tasks_all.update({k: v for k, v in old.items() if k in want})
        keys = sorted(tasks_all)
    elif a.select:
        import csv
        csv.field_size_limit(10 ** 9)
        tasks_all, n_rows = {}, 0
        for f in a.select:
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    n_rows += 1
                    if zlib.crc32(r["game_key"].encode()) % a.nshards == a.shard:
                        tasks_all[r["game_key"]] = (r["sgf_content"], r["color"])
        keys = sorted(tasks_all)
        print(f"{n_rows} records in {len(a.select)} selections, {len(keys)} in this shard ({time.time() - t0:.0f} s)", flush=True)
    else:
        keys = sorted(set(PT.read_keys(a.keys)) - (set(PT.read_keys(a.skip_keys)) if a.skip_keys else set()))
    st = np.load(a.stats) if a.stats else None
    mine = keys if a.select else keys[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"sup_s{a.shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do; projection of {'+'.join(blocks)} to {W.shape[1]} dims" + (" + 4 policy scalars" if a.scalars else ""), flush=True)
    if not todo:
        return 0
    config = torch.load(str(a.ckpt or te.CKPT), map_location="cpu", weights_only=a.ckpt is None)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(a.ckpt or te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        done_games = empty = 0
        for k, part in todo:
            sgf = {q: tasks_all[q] for q in part} if tasks_all is not None else load_games(part, aicup_pid.TRAIN_DIR)
            done = list(pool.imap(replay_external if a.select else PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))
            some = [d for d in done if len(d[3])]
            empty += len(done) - len(some)
            if some:
                bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])
                pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
                if use_pol:
                    fg, fl, fe, sc_, _bad = PT.policy_vectors(model, bins, globs, pos, mover, rows, a.batch)
                    src = {"g": fg, "x": fl, "e": fe}
                    z = ((np.hstack([src[b] for b in blocks]) - mu) / sd) @ W
                    if a.scalars:
                        z = np.hstack([z, np.stack([sc_[:, 0], np.log1p(sc_[:, 1]), sc_[:, 2], sc_[:, 4]], axis=1)])
                else:
                    fg, fl = PT.full_vectors(model, bins, globs, pos, mover, rows, a.batch)
                    z = ((np.hstack([fg, fl]) - mu) / sd) @ W
            else:
                z = np.zeros((0, W.shape[1] + (4 if a.scalars else 0)), np.float32)
            if not np.isfinite(z).all():
                raise ValueError(f"part {k}: non-finite features")
            path = out / f"sup_s{a.shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            feat = {"z": z.astype(np.float16)} if st is None else {"zq": np.clip(np.rint((z - st["mean"]) / st["sd"] * 20.0), -127, 127).astype(np.int8)}
            head = dict(keys=np.array([d[0] for d in done]), n=np.array([len(d[3]) for d in done], np.int32), t=np.concatenate([d[5] for d in done]).astype(np.int16))
            if a.out2:  # first, so that a part counts as done only when both stores have it
                assert a.scalars and st is None, "--out2 goes with --scalars and float parts"
                p2 = Path(a.out2) / path.name
                p2.parent.mkdir(parents=True, exist_ok=True)
                t2 = p2.with_name(p2.name + ".tmp.npz")
                np.savez(t2, z=z[:, :W.shape[1]].astype(np.float16), **head)
                os.replace(t2, p2)
            np.savez(tmp, **head, **feat)
            os.replace(tmp, path)
            done_games += len(part)
            if (k % 10) == 0:
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s" + (f", {empty} without plies" if empty else ""), flush=True)
            if a.max_parts and done_games >= a.max_parts * a.part_size:
                break
    finally:
        pool.close()
        pool.join()
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
