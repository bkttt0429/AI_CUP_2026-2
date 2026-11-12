#!/usr/bin/env python
"""Game vectors of the rank session's sequence identity model (AI_CUP_2026-1 logs/nano4/m88_seqid.py, METHOD_NOTES 2.30) for
competition 1's v2 pools. The model reads a whole game ply by ply (both players, first 256 plies: trunk vectors, played point,
colour, displacements, which side is the subject) and returns a unit vector for the subject side; it was trained on training
players only (net v never saw fold v's players, net -1 saw every training player; no official game).
  --stage folds     training games that have stored per-ply vectors, fold v's games under net v -> <out>/fold<v>.npz   (simulation)
  --stage official  the added pools' games (scripts/pid_v2_ply_extract.py parts) under net -1 -> <out>/official.npz      (prediction)
Run from the AI_CUP_2026-1 checkout (m88_seqid.py resolves its paths from the working directory). Read-only on their files."""
import argparse
import glob
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

AICUP1 = Path.cwd()
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4")]
import m88_seqid as m88  # noqa: E402


def load_net(ckpt):
    net = m88.build_model()
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(st["best_state"])
    return net.cuda().eval(), st["best"]


def embed(net, GL, POS, MOV, LEN, SUBJ, batch=1024):
    out = np.zeros((len(LEN), m88.D_OUT), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(LEN), batch):
            s = slice(i, i + batch)
            out[s] = net(torch.as_tensor(GL[s]).cuda(), torch.as_tensor(POS[s].astype(np.int64)).cuda(), torch.as_tensor(MOV[s].astype(np.int64)).cuda(),
                         torch.as_tensor(LEN[s].astype(np.int64)).cuda(), torch.as_tensor(SUBJ[s].astype(np.int64)).cuda()).float().cpu().numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("folds", "official"))
    ap.add_argument("--ckpt-dir", default=str(AICUP1 / "results/m88"))
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seq")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "official":
        net, best = load_net(Path(a.ckpt_dir) / "ckpt_-1.pt")
        keys, embs = [], []
        for f in sorted(glob.glob(f"{a.ply_dir}/ply_*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            keys.append(z["key"].astype(str))
            embs.append(embed(net, z["GL"], z["POS"], z["MOV"], z["LEN"], z["SUBJ"]))
        keys, embs = np.concatenate(keys), np.concatenate(embs)
        np.savez(out / "official.npz", key=keys, emb=embs.astype(np.float16), net=np.array(-1), best=np.array(str(best)))
        print(f"official: {len(keys)} games under net -1 (best {best}) -> {out / 'official.npz'} ({time.time() - t0:.0f} s)", flush=True)
        return 0
    for v in a.folds:
        net, best = load_net(Path(a.ckpt_dir) / f"ckpt_{v}.pt")
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{v}").glob("games_s*_p*.csv"))]
        keys, pids, embs = [], [], []
        with mp.get_context("fork").Pool(12) as pool:
            for i in range(0, len(files), 48):
                for key, pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                    keys.append(key); pids.append(pid)
                    embs.append(embed(net, GL, POS, MOV, LEN, SUBJ))
                print(f"  fold {v}: {min(i + 48, len(files))}/{len(files)} parts, {sum(len(k) for k in keys)} games ({time.time() - t0:.0f} s)", flush=True)
        keys, pids, embs = np.concatenate(keys), np.concatenate(pids), np.concatenate(embs)
        np.savez(out / f"fold{v}.npz", key=keys, pid=pids, emb=embs.astype(np.float16), best=np.array(str(best)))
        print(f"fold {v}: {len(keys)} games of {len(set(pids.tolist()))} players under net {v} (never saw them; best {best}) -> {out / f'fold{v}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
