#!/usr/bin/env python
"""METHOD_NOTES 2.172 (layer probe): per-game rows of the humanv0 trunk AT SEVERAL DEPTHS.

Every stored feature of the project comes from the trunk's last output ("trunkfinal": the residual stream after the last block,
normalised and activated), which the network was trained to predict the next move from. This script stores, for a sample of
games (the probe sample of scripts/pid_probe_fullrows.py --stage keys), the per-game MEAN of the board-pooled vector and of the
played-point vector of the residual stream at chosen depths:
  emb      after the input embedding            b<k>   after block k (the stream = embedding + the residuals of blocks 1..k)
  final    trunkfinal, as every existing feature (the reference; equals pid_trunk_extract.py --full's means)
Only the target player's plies, positions before the move, the neutral metadata rows - as pid_trunk_extract.py --fast.
A part holds keys, n_target and rows [G, L, 2 x 384] float16 (pooled mean | played-point mean; a pass has no point).
The stream is rebuilt from the blocks' reported residuals; the first batch checks that normalising and activating the rebuilt
last stream gives trunkfinal (it stops otherwise).
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --out DIR --workers 10
Run from the AI_CUP_2026-1 checkout. Fold 0 is never read (the key list has none).
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pid_trunk_extract as PT  # noqa: E402  (puts both checkouts on sys.path; replay_target, read_keys)
import aicup_pid  # noqa: E402
from aicup import l1b  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows  # noqa: E402
from aicup.scoring import load_games  # noqa: E402

te = PT.te
DEFAULT = ("emb", "b3", "b6", "b9", "b12", "b15", "b18", "final")


def states(model, b, g, c, rows, want, check=False):
    """{name: [B, C, 19, 19] float32} of one batch (see the module text)."""
    names = [blk.name for blk in model.blocks]
    extra = l1b._GraphExtraOutputs(["trunk_after_embedding", "trunkfinal"] + [n + ".out" for n in names])
    bf, gf = b.float(), g.float()
    meta = None
    if model.get_has_metadata_encoder():
        meta = torch.as_tensor(rows, dtype=torch.float32, device=b.device)[c]
    model(bf, gf, input_meta=meta, extra_outputs=extra)
    r = extra.returned
    missing = [n for n in ["trunk_after_embedding", "trunkfinal"] + [n + ".out" for n in names] if n not in r]
    if missing:
        raise KeyError(f"the model did not report {missing[:4]} (available e.g. {extra.available[:6]})")
    s = r["trunk_after_embedding"].float()
    out = {"emb": s} if "emb" in want else {}
    for i, n in enumerate(names, 1):
        s = s + r[n + ".out"].float()
        if f"b{i}" in want:
            out[f"b{i}"] = s
    final = r["trunkfinal"].float()
    if "final" in want:
        out["final"] = final
    if check:
        mask = bf[:, 0:1].contiguous()
        with torch.autocast("cuda", enabled=False):
            again = model.act_trunkfinal(model.norm_trunkfinal(s, mask=mask, mask_sum_hw=mask.sum((2, 3), keepdim=True), mask_sum=mask.sum()))
        d = (again.float() - final).abs()
        err = float(d.mean()), float(final.abs().mean()), float(d.max())
        print(f"check: the rebuilt last stream, normalised and activated, against trunkfinal: mean difference {err[0]:.5f} of mean |trunkfinal| {err[1]:.4f} "
              f"(largest {err[2]:.3f}; the model runs in bfloat16); {len(names)} blocks", flush=True)
        if not err[0] < 0.05 * max(err[1], 1e-6):
            raise RuntimeError("the residual stream was not rebuilt correctly (another trunk layout?)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", nargs="+", default=list(DEFAULT))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-parts", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(PT.read_keys(a.keys)))
    mine = keys[a.shard::a.nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"layers_s{a.shard:03d}_p{k:04d}.npz").exists()]
    if a.max_parts:
        todo = todo[: a.max_parts]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do; layers {' '.join(a.layers)}", flush=True)
    if not todo:
        return 0
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        first, games_done = True, 0
        for k, part in todo:
            sgf = load_games(part, aicup_pid.TRAIN_DIR)
            done = list(pool.imap(PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))
            lens = np.array([len(d[3]) for d in done])
            some = [d for d in done if len(d[3])]
            G_, L_ = len(done), len(a.layers)
            res = None
            if some:
                bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])
                pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
                gid = torch.as_tensor(np.repeat(np.arange(G_), lens)).cuda()
                n = len(pos)
                sum_g = sum_l = cnt_l = None
                with torch.no_grad():
                    for i in range(0, n, a.batch):
                        j = min(n, i + a.batch)
                        b, g = torch.as_tensor(bins[i:j]).cuda(), torch.as_tensor(globs[i:j]).cuda()
                        c = torch.as_tensor(mover[i:j]).cuda().long()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            st = states(model, b, g, c, rows, a.layers, check=first)
                        first = False
                        p = torch.as_tensor(pos[i:j]).cuda().long()
                        on = p < te.PASS_POS
                        ix = on.nonzero()[:, 0]
                        if sum_g is None:
                            C = st[a.layers[0]].shape[1]
                            sum_g = torch.zeros(L_, G_, C, device="cuda"); sum_l = torch.zeros(L_, G_, C, device="cuda"); cnt_l = torch.zeros(G_, device="cuda")
                        cnt_l.index_add_(0, gid[i:j][on], torch.ones(int(on.sum()), device="cuda"))
                        for li, name in enumerate(a.layers):
                            s = st[name]
                            sum_g[li].index_add_(0, gid[i:j], s.mean(dim=(2, 3)))
                            sum_l[li].index_add_(0, gid[i:j][on], s[ix, :, p[on] // 19, p[on] % 19])
                cnt_g = torch.as_tensor(lens, device="cuda").float().clamp(min=1)
                res = torch.cat([sum_g / cnt_g[None, :, None], sum_l / cnt_l.clamp(min=1)[None, :, None]], dim=2).permute(1, 0, 2).cpu().numpy()
            if res is None:
                res = np.zeros((G_, L_, 2 * 384), np.float32)
            if not np.isfinite(res).all():
                raise ValueError(f"part {k}: non-finite rows")
            path = out / f"layers_s{a.shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            np.savez(tmp, keys=np.array([d[0] for d in done]), n_target=lens.astype(np.int16), rows=res.astype(np.float16), layers=np.array(a.layers))
            os.replace(tmp, path)
            games_done += len(part)
            if (k + 1) % 10 == 0 or games_done == len(part):
                print(f"part {k + 1}/{len(parts)}: {games_done} games, {games_done / (time.time() - t0):.1f} games/s", flush=True)
    finally:
        pool.terminate()
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
