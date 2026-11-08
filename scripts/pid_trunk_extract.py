#!/usr/bin/env python
"""Line B4 (trunk variant): per-game trunk-style features of the pristine humanv0 trunk (METHOD_NOTES 2.21).

Every game is replayed exactly as AI_CUP_2026-1's scripts/trunk_extract.py replays it (same Features, rules, handicap fix,
random projections to 64 + 64 dims with the same seed); the trunk is run on the TARGET player's positions only (Black in
this task), and each game is stored as one 256-dimensional row: mean and standard deviation of the board-pooled vector
and of the played-point vector over the target's plies (passes have no played point). Rows are written in parts
(<out>/<tag>_s<global shard>_p<k>.npz: keys, feats float32 [G, 256], n_target) and the run resumes per part.

Inputs: --official (both official pools: candidate games keyed c:<row>:<game_id> as predict_pid.py keys them, question
games q:<question_id>:<i>) or --keys FILE (training game keys, one per line or a CSV with a game_key column). Sharding:
shards.sh appends --shard i --nshards N per GPU; --node k --nodes M spreads one key list over M such jobs.

  sbatch -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J trunkx scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 \\
      python -I ../BK_AICUP_2026-2/scripts/pid_trunk_extract.py --official --out ../BK_AICUP_2026-2/data/trunk/official --node 0 --nodes 2 --workers 10
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402  (puts AI_CUP_2026-1/src on sys.path)
from aicup_pid import pools  # noqa: E402
from aicup_pid.paths import AICUP1  # noqa: E402,F401  (its scripts directory is on sys.path)
import trunk_extract as te  # noqa: E402  (AI_CUP_2026-1's replay, projections and vectors)
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.submission import check_game  # noqa: E402

DIM = 4 * te.PROJ_DIM
T = ROOT / "data" / "tests"


def official_tasks(tests_dir=T):
    """{key: (sgf, colour)} of every usable game of both official pools in tests_dir (default data/tests)."""
    out = {}
    for c_name, q_name in (("player_identification_candidates_public.csv", "player_identification_test_public.csv"),
                           ("player_identification_candidates_private.csv", "player_identification_test_private.csv")):
        cands = pools.read_candidates(Path(tests_dir) / c_name)
        for i, r in enumerate(cands.itertuples()):
            out[f"c:{i}:{r.game_id}"] = (r.sgf, "B")
        for q in pools.read_questions(Path(tests_dir) / q_name):
            for i, (sgf, colour) in enumerate(q["games"]):
                if check_game(sgf, colour or "B")[0]:
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")
    return out


def read_keys(path):
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p, dtype=str)["game_key"].tolist()
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def aggregate(g, l, pos, lens):
    """[G, 256]: per game mean / std of the pooled and of the played-point vectors over its target plies."""
    out = np.zeros((len(lens), DIM), np.float32)
    start = 0
    for i, n in enumerate(lens):
        if n:
            gi = g[start:start + n].astype(np.float32)
            li = l[start:start + n][pos[start:start + n] < te.PASS_POS].astype(np.float32)
            out[i, 0:64], out[i, 64:128] = gi.mean(axis=0), gi.std(axis=0)
            if len(li):
                out[i, 128:192], out[i, 192:256] = li.mean(axis=0), li.std(axis=0)
        start += n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--official", action="store_true")
    ap.add_argument("--tests-dir", default=str(T), help="folder of the official CSVs for --official (a re-released set goes in its own folder)")
    src.add_argument("--keys", help="training game keys (text file or CSV with game_key)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="part")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-games", type=int, default=0, help="smoke tests only")
    a = ap.parse_args(argv)
    t0 = time.time()
    shard, nshards = a.node * a.nshards + a.shard, a.nodes * a.nshards
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.official:
        tasks_all = official_tasks(a.tests_dir)
        keys = sorted(tasks_all)
    else:
        tasks_all, keys = None, sorted(set(read_keys(a.keys)))
    if a.max_games:
        keys = keys[: a.max_games]
    mine = keys[shard::nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {shard}/{nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        proj_g = proj_l = None
        games_done = plies_done = 0
        for k, part in todo:
            sgf = {q: tasks_all[q] for q in part} if tasks_all is not None else load_games(part, aicup_pid.TRAIN_DIR)
            done = list(pool.imap(te.replay, [(q, *sgf[q]) for q in part], chunksize=1))
            sel = [d[5] for d in done]  # target plies of each game
            bins = np.concatenate([d[1][m] for d, m in zip(done, sel)])
            globs = np.concatenate([d[2][m] for d, m in zip(done, sel)])
            pos = np.concatenate([d[3][m] for d, m in zip(done, sel)])
            mover = np.concatenate([d[4][m] for d, m in zip(done, sel)])
            lens = [int(m.sum()) for m in sel]
            if proj_g is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(bins[:1]).cuda(), torch.as_tensor(globs[:1]).cuda(),
                                                    torch.as_tensor(mover[:1]).cuda().long(), rows)
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)
            g, l = te.vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, a.batch)
            feats = aggregate(g, l, pos, lens)
            if not np.isfinite(feats).all():
                raise ValueError(f"part {k}: non-finite features")
            path = out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, keys=np.array([d[0] for d in done]), feats=feats, n_target=np.array(lens, np.int16))
            os.replace(tmp, path)
            games_done += len(part)
            plies_done += sum(len(d[3]) for d in done)
            el = time.time() - t0
            print(f"part {k + 1}/{len(parts)}: {games_done} games, {games_done / el:.1f} games/s, {plies_done / el:.0f} plies/s replayed", flush=True)
    finally:
        pool.close()
        pool.join()
    print(f"shard {shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
