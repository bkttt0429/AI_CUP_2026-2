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


STATE = "best"  # --state last: the state after the final step instead of the one with the best held-out top-1


ARCH0 = (m88.D, m88.LAYERS, m88.HEADS, m88.FFN)


def load_net(ckpt):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = st.get("arch", ARCH0)  # a network trained with pid_seq_train.py --arch is rebuilt at its own size
    net = m88.build_model()
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = ARCH0
    net.load_state_dict(st["best_state"] if STATE == "best" else st["net"])
    return net.cuda().eval(), (st["best"] if STATE == "best" else {"step": st["step"], "state": "last"})


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
    ap.add_argument("--nets", nargs="+", type=int, default=[-1], help="official stage: which networks")
    ap.add_argument("--tag", default="", help="suffix of the official output files (e.g. _m90)")
    ap.add_argument("--state", choices=("best", "last"), default="best")
    ap.add_argument("--ckpt-suffix", default="", help="checkpoint file ckpt_<v><suffix>.pt (the rank session's further seeds: _s1, _s2)")
    ap.add_argument("--name", default="", help="folds stage: output fold<v><name>.npz")
    ap.add_argument("--sup-ply", nargs="*", default=None, help="folds stage: supervised per-ply features (scripts/pid_sup_ply_extract.py; METHOD_NOTES 2.88)")
    ap.add_argument("--sup-dims", type=int, nargs=2, default=None, help="the input dimensions they replace, as the network was trained (pid_seq_train.py --sup-dims)")
    ap.add_argument("--sup-only", action="store_true")
    ap.add_argument("--sup-stats", default=None)
    ap.add_argument("--flip", action="store_true", help="vectors of the OTHER side of every game (the subject's opponent; METHOD_NOTES 2.38); give --name / --tag a suffix such as _opp")
    a = ap.parse_args(argv)
    global STATE
    STATE = a.state
    sfx = "" if a.state == "best" else "_last"
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "official":  # no official player is in any net's training set, so every net is usable here
        for v in a.nets:
            net, best = load_net(Path(a.ckpt_dir) / f"ckpt_{v}{a.ckpt_suffix}.pt")
            keys, embs = [], []
            for f in sorted(glob.glob(f"{a.ply_dir}/ply_*.npz")):
                if f.endswith(".tmp.npz"):
                    continue
                z = np.load(f)
                keys.append(z["key"].astype(str))
                embs.append(embed(net, z["GL"], z["POS"], z["MOV"], z["LEN"], 1 - z["SUBJ"] if a.flip else z["SUBJ"]))
            keys, embs = np.concatenate(keys), np.concatenate(embs)
            dest = out / ("official.npz" if v == -1 and a.tag == "" and not sfx else f"official{a.tag}_net{v}{sfx}.npz")
            np.savez(dest, key=keys, emb=embs.astype(np.float16), net=np.array(v), best=np.array(str(best)))
            print(f"official: {len(keys)} games under net {v} (best {best}) -> {dest} ({time.time() - t0:.0f} s)", flush=True)
        return 0
    sup = None
    if a.sup_ply and a.sup_dims:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pid_sup
        sup = pid_sup.load_sup(a.sup_ply, a.sup_stats)
    for v in a.folds:
        net, best = load_net(Path(a.ckpt_dir) / f"ckpt_{v}{a.ckpt_suffix}.pt")
        mu_np, sd_np = net.mu.float().cpu().numpy(), net.sd.float().cpu().numpy()
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{v}").glob("games_s*_p*.csv"))]
        keys, pids, embs = [], [], []
        with mp.get_context("fork").Pool(12) as pool:
            for i in range(0, len(files), 48):
                for key, pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                    keys.append(key); pids.append(pid)
                    if sup is not None:  # the games without supervised features keep zeros (the mean) there: only the pool games are scored
                        idx, D = pid_sup.dense(sup, key.tolist(), GL.shape[1])
                        z = np.zeros((len(key), GL.shape[1], D.shape[2]), np.float32)
                        z[idx >= 0] = D[idx[idx >= 0]]
                        GL = pid_sup.put(GL.astype(np.float32), z, mu_np, sd_np, a.sup_dims, a.sup_only)
                    embs.append(embed(net, GL, POS, MOV, LEN, 1 - SUBJ if a.flip else SUBJ))
                print(f"  fold {v}: {min(i + 48, len(files))}/{len(files)} parts, {sum(len(k) for k in keys)} games ({time.time() - t0:.0f} s)", flush=True)
        keys, pids, embs = np.concatenate(keys), np.concatenate(pids), np.concatenate(embs)
        np.savez(out / f"fold{v}{a.name}{sfx}.npz", key=keys, pid=pids, emb=embs.astype(np.float16), best=np.array(str(best)))
        print(f"fold {v}: {len(keys)} games of {len(set(pids.tolist()))} players under net {v} (never saw them; best {best}) -> {out / f'fold{v}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
