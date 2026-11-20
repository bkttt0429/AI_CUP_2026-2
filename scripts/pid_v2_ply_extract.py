#!/usr/bin/env python
"""Per-ply trunk vectors of the v2 release's added pools, in the layout of the rank session's sequence identity model
(AI_CUP_2026-1 logs/nano4/m88_seqid.py read_part): per game the first 256 plies of BOTH players, GL [256, 128] fp16 (board-pooled
64 + played-point 64), POS (tensor position, 361 pass, 362 padding), MOV (0 Black / 1 White), LEN, SUBJ (the player's colour).
Same replay, network and projections as scripts/pid_trunk_extract.py / AI_CUP_2026-1 scripts/trunk_extract.py. Sharded by
shards.sh (--shard i --nshards N); one .npz per part, resumable. Keys as pid_trunk_extract.py --official2."""
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
import aicup_pid  # noqa: E402,F401
import trunk_extract as te  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from pid_trunk_extract import official2_tasks  # noqa: E402

T_MAX = 256


def original_tasks(tests_dir, per_candidate):
    """{key: (sgf, 'B')} of the ORIGINAL pools' candidate games (pub_ / priv_; Black only), the first `per_candidate` usable games
    of every candidate in file order; keys c:<row in its file>:<game_id> as pid_trunk_extract.py --official. No question game."""
    import pandas as pd
    from collections import Counter
    out = {}
    for tag in ("public", "private"):
        c = pd.read_csv(Path(tests_dir) / f"player_identification_candidates_{tag}.csv", dtype=str, keep_default_na=False)
        seen = Counter()
        for i, r in enumerate(c.itertuples()):
            if seen[r.player_id] < per_candidate and check_game(r.sgf_content, "B")[0]:
                out[f"c:{i}:{r.game_id}"] = (r.sgf_content, "B")
                seen[r.player_id] += 1
    return out


def original_question_tasks(tests_dir):
    """{key: (sgf, colour)} of the ORIGINAL pools' QUESTION games (pub_q_ / priv_q_; METHOD_NOTES 2.240): keys q:<question_id>:<i>
    as the added pools' question games (pid_trunk_extract.official2_tasks)."""
    import pid_trunk_extract as pte
    out = {}
    for tag in ("public", "private"):
        for q in pte.pools.read_questions(Path(tests_dir) / f"player_identification_test_{tag}.csv"):
            for i, (sgf, colour) in enumerate(q["games"]):
                if check_game(sgf, colour or "B")[0]:
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--original", type=int, default=0, help="instead of the added pools: this many candidate games per candidate of the ORIGINAL pools (METHOD_NOTES 2.54)")
    ap.add_argument("--original-questions", action="store_true", help="instead of the added pools: the QUESTION games of the ORIGINAL pools (METHOD_NOTES 2.240)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not (a.original and a.original_questions)
    tasks = original_question_tasks(a.tests_dir) if a.original_questions else original_tasks(a.tests_dir, a.original) if a.original else official2_tasks(a.tests_dir)
    mine = sorted(tasks)[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"ply_s{a.shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(tasks)} games, {len(todo)} parts to do", flush=True)
    if not todo:
        return 0
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        proj_g = proj_l = None
        for k, part in todo:
            done = list(pool.imap(te.replay, [(q, *tasks[q]) for q in part], chunksize=1))
            n = len(done)
            GL = np.zeros((n, T_MAX, 128), np.float16)
            POS = np.full((n, T_MAX), 362, np.int16)
            MOV = np.zeros((n, T_MAX), np.int8)
            LEN = np.zeros(n, np.int16)
            SUBJ = np.array([0 if tasks[d[0]][1] == "B" else 1 for d in done], np.int8)
            lens = [min(len(d[3]), T_MAX) for d in done]
            bins = np.concatenate([d[1][:m] for d, m in zip(done, lens)])
            globs = np.concatenate([d[2][:m] for d, m in zip(done, lens)])
            pos = np.concatenate([d[3][:m] for d, m in zip(done, lens)])
            mover = np.concatenate([d[4][:m] for d, m in zip(done, lens)])
            if proj_g is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(bins[:1]).cuda(), torch.as_tensor(globs[:1]).cuda(), torch.as_tensor(mover[:1]).cuda().long(), rows)
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)
            g, l = te.vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, a.batch)
            start = 0
            for i, m in enumerate(lens):
                GL[i, :m, :64], GL[i, :m, 64:] = g[start:start + m], l[start:start + m]
                POS[i, :m], MOV[i, :m], LEN[i] = pos[start:start + m], mover[start:start + m], m
                start += m
            path = out / f"ply_s{a.shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, key=np.array([d[0] for d in done]), GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ)
            os.replace(tmp, path)
            print(f"part {k + 1}/{len(parts)}: {n} games, {int(sum(lens))} plies ({time.time() - t0:.0f} s)", flush=True)
    finally:
        pool.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
