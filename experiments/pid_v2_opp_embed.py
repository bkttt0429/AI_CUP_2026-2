#!/usr/bin/env python
"""Opponent-side vectors of every game of the added official pools (METHOD_NOTES 2.117): the sequence networks as they are
(no pool fine-tune), with the OTHER colour flagged as the subject. One file per network and pool,
  <out>/official_<pool>_net-1_<name>_opp_t0.npz   key, emb [128] float16   (candidate games c:<row>:<game_id> and question games)
for aicup_pid.oppside. Plain-input networks only: the supervised per-ply features exist for the subject's plies only.
Run from the AI_CUP_2026-1 checkout, one GPU:
  python -I <this> --nets o7Fs0e o7Fs1e ... [--out /work/u2507115/pid_v2/seq_opp]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
import pid_v2_seq_finetune as FT  # noqa: E402  (load_net, embed)


def load_pool(pool, tests_dir, ply_dir):
    """Per-ply arrays of every game of the pool: its candidate games and its question games."""
    tagname = "public" if pool == "pub2" else "private"
    c = pd.read_csv(Path(tests_dir) / f"player_identification_candidates2_{tagname}.csv", dtype=str, keep_default_na=False, usecols=["game_id"])
    own = {f"c:{i}:{g}" for i, g in enumerate(c.game_id)}
    parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
    for f in sorted(glob.glob(f"{ply_dir}/ply_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        key = z["key"].astype(str)
        m = np.array([(k in own) or k.startswith(f"q:{pool}_") for k in key])
        if m.any():
            for name in parts:
                parts[name].append(key[m] if name == "key" else z[name][m])
    A = {k: np.concatenate(x) for k, x in parts.items()}
    if not set(np.unique(A["SUBJ"]).tolist()) <= {0, 1}:
        raise ValueError("SUBJ is expected to be 0 / 1")
    return A, int(sum(k in own for k in A["key"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", required=True, help="checkpoint names without ckpt_-1_ and .pt")
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--tests-dir", default=str(R2 / "data" / "tests_1111"))
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seq_opp")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = {}
    for pool in ("pub2", "priv2"):
        A, n_c = load_pool(pool, a.tests_dir, a.ply_dir)
        B = dict(A)
        B["SUBJ"] = (1 - A["SUBJ"]).astype(A["SUBJ"].dtype)
        pools[pool] = B
        print(f"{pool}: {len(A['key'])} games ({n_c} candidate games, {len(A['key']) - n_c} question games) ({time.time() - t0:.0f} s)", flush=True)
    for name in a.nets:
        net = FT.load_net(Path(a.ckpt_dir) / f"ckpt_-1_{name}.pt")
        if net.sup_cfg:
            raise ValueError(f"{name} reads supervised per-ply features, which exist for the subject's plies only")
        for pool, B in pools.items():
            vec = FT.embed(net, B, np.arange(len(B["key"])))
            dest = out / f"official_{pool}_net-1_{name}_opp_t0.npz"
            tmp = dest.with_name(dest.name + ".tmp.npz")
            np.savez(tmp, key=B["key"], emb=vec.astype(np.float16), step=np.array(0))
            os.replace(tmp, dest)
        del net
        print(f"{name}: written ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
