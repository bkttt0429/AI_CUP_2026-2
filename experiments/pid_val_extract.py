#!/usr/bin/env python
"""Probe P7 (METHOD_NOTES 2.115): per-ply features that say HOW GOOD the position and the move are, next to the second
supervised feature set.

The same forward pass that gives the trunk and the policy also gives the network's value, score-lead and uncertainty heads.
None of this is in any input so far, and the difference between two consecutive own turns (after my move and the reply) says
whether the game got better or worse for the player: a measure of move quality that needs no further network evaluation.

Per ply of the subject (the layout of scripts/pid_sup_ply_extract.py: keys, n, t, z float16):
  z[:, 0:64]    the joint projection over pooled / played-point / policy-expected vectors (--proj, blocks g x e)
  z[:, 64:68]   policy scalars: surprisal, log(1 + rank), entropy, distance to the policy's top point
  z[:, 68:76]   value scalars: expected outcome (win - loss), score lead, score standard deviation, short-term value error,
                short-term score error, short-horizon expected outcome; then the CHANGE of expected outcome and of lead until the
                subject's next own turn (0 at the subject's last ply)
The first 68 dimensions are exactly the "js" set, so the probe compares 76 against 68 dimensions.
This file is self-contained on purpose: the extraction scripts of the running jobs are not touched.
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --proj proj3.npz --out DIR --workers 10
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
import pid_trunk_extract as PT  # noqa: E402  (replay_target, read_keys; nothing of it is changed)
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import _GraphExtraOutputs, neutral_rows  # noqa: E402
from aicup.scoring import load_games  # noqa: E402

te = PT.te


def forward_all(model, bins, globs, pos, mover, rows, batch):
    """One forward pass per position: pooled g, played-point x, policy-expected e ([n, C] each), policy scalars S [n, 5]
    (as pid_trunk_extract.policy_vectors) and value scalars V [n, 6]: expected outcome (win - loss) for the side to move,
    score lead, score standard deviation, short-term value error, short-term score error, short-horizon expected outcome."""
    n = len(pos)
    C = None
    S, V = np.zeros((n, 5), np.float32), np.zeros((n, 6), np.float32)
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device="cuda")
    meta = model.get_has_metadata_encoder()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b, g = torch.as_tensor(bins[i:j]).cuda().float(), torch.as_tensor(globs[i:j]).cuda().float()
            c = torch.as_tensor(mover[i:j]).cuda().long()
            extra = _GraphExtraOutputs(["trunkfinal"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(b, g, input_meta=rows_t[c] if meta else None, extra_outputs=extra)
                post = model.postprocess_output(outputs)[0]
            trunk, logits = extra.returned["trunkfinal"].float(), post[0][:, 0, :].float()
            if C is None:
                C = trunk.shape[1]
                G, X, E = np.empty((n, C), np.float32), np.empty((n, C), np.float32), np.empty((n, C), np.float32)
            legal_b = ((b[:, 0] - b[:, 1] - b[:, 2] - b[:, 6]) > 0.5).flatten(1)
            legal = torch.cat([legal_b, torch.ones(j - i, 1, dtype=torch.bool, device=b.device)], dim=1)
            lp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
            p = torch.as_tensor(pos[i:j].astype(np.int64)).cuda()
            on = p < te.PASS_POS
            lpp = lp.gather(1, p[:, None])[:, 0]
            lpp = torch.where(torch.isfinite(lpp), lpp, torch.full_like(lpp, -20.0)).clamp(min=-20.0)
            rank = (lp > lpp[:, None]).sum(1).float()
            pr = lp.exp()
            ent = -(pr * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum(1)
            pb = torch.softmax(logits[:, :361].masked_fill(~legal_b, float("-inf")), dim=1)
            pb = torch.where(torch.isfinite(pb), pb, torch.zeros_like(pb))
            e = torch.einsum("bp,bcp->bc", pb, trunk.flatten(2)) * on[:, None].float()
            local = torch.zeros(j - i, C, device=trunk.device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            top = pb.argmax(1)
            dist = torch.maximum((p // 19 - top // 19).abs(), (p % 19 - top % 19).abs()).float() * on.float()
            G[i:j], X[i:j], E[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy(), e.cpu().numpy()
            S[i:j] = torch.stack([-lpp, rank, ent, pr.max(1).values, dist], dim=1).cpu().numpy()
            vp = torch.softmax(post[1].float(), dim=1)  # win, loss, no result (side to move)
            tds = torch.softmax(post[2][:, 2].float(), dim=1)  # the short-horizon value
            V[i:j] = torch.stack([vp[:, 0] - vp[:, 1], post[10].float(), post[9].float(), post[12].float(), post[13].float(), tds[:, 0] - tds[:, 1]], dim=1).cpu().numpy()
    return G, X, E, S, V


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--proj", required=True, help="the joint projection (pid_sup_fit.py --blocks g x e)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-parts", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    P = np.load(a.proj)
    mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)
    assert [str(b) for b in P["blocks"]] == ["g", "x", "e"], "the joint projection is expected"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(PT.read_keys(a.keys)))
    mine = keys[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"sup_s{a.shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do; 64 + 4 + 8 dims", flush=True)
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
            lens = [len(d[3]) for d in done]
            G, X, E, S, V = forward_all(model, bins, globs, pos, mover, rows, a.batch)
            z = ((np.hstack([G, X, E]) - mu) / sd) @ W
            pol = np.stack([S[:, 0], np.log1p(S[:, 1]), S[:, 2], S[:, 4]], axis=1)
            dw, dl = np.zeros(len(pos), np.float32), np.zeros(len(pos), np.float32)
            start = 0
            for n in lens:  # change until the subject's next own turn, inside each game
                if n > 1:
                    dw[start:start + n - 1] = V[start + 1:start + n, 0] - V[start:start + n - 1, 0]
                    dl[start:start + n - 1] = V[start + 1:start + n, 1] - V[start:start + n - 1, 1]
                start += n
            feat = np.hstack([z, pol, V, dw[:, None], dl[:, None]]).astype(np.float32)
            if not np.isfinite(feat).all():
                raise ValueError(f"part {k}: non-finite features")
            path = out / f"sup_s{a.shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, keys=np.array([d[0] for d in done]), n=np.array(lens, np.int32), t=np.concatenate([d[5] for d in done]).astype(np.int16), z=np.clip(feat, -6e4, 6e4).astype(np.float16))
            os.replace(tmp, path)
            done_games += len(part)
            if (k % 10) == 0:
                v = feat[:, 68:]
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s; value scalars mean " + " ".join(f"{x:.2f}" for x in v.mean(0)) + " sd " + " ".join(f"{x:.2f}" for x in v.std(0)), flush=True)
            if a.max_parts and done_games >= a.max_parts * a.part_size:
                break
    finally:
        pool.close()
        pool.join()
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
