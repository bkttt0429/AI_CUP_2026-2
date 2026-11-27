#!/usr/bin/env python
"""Probe T2 on the OLD-POOL simulation (METHOD_NOTES 2.123, part O): the lending of scripts/pid_probe_aug.py with multi-game
questions of the REAL sizes and true answers.

Pools of scripts/pid_v2_oldsim.py: the original pools' candidates (official-population players no network was trained on),
10-game candidate sets, questions of 1 / 1 / 1 / 2 / 2 / 3 / 5 games, Black only, 400 candidates per side. Per draw half of
the candidates own a multi-game question (a third each of 5 / 3 / 2 games: the proportions of the real added pools; distinct
owners), assigned as the pipeline assigns them (soft, lam 1); every question whose Top-1 leads its second choice by at least
m0 lends its games to that candidate; the first 1-game question of EVERY candidate is then re-scored against the enlarged
reference sets. Reported: the score difference of the 1-game questions whose owner has a multi-game question, has none, all.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_aug_old_v2o13c_9.json
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_aug import fuse_v, lend, margins  # noqa: E402
from pid_probe_qq import boot, own_scores  # noqa: E402

G = {}
SIZES = (5, 3, 2)


def draw(task):
    d, m0s, side = task
    P, H = G["P"], len(G["have"])
    rng = np.random.default_rng(9950 + 13 * d + (0 if side == "pub" else 1))
    perm = rng.permutation(H)
    n5 = n3 = H // 6
    n2 = H // 2 - n5 - n3
    pick = [(int(h), 5) for h in perm[:n5]] + [(int(h), 3) for h in perm[n5:n5 + n3]] + [(int(h), 2) for h in perm[n5 + n3:n5 + n3 + n2]]
    Fm = np.stack([G["Fmulti"][m][h] for h, m in pick])
    qs = [G["multi"][m][h] for h, m in pick]
    truth = G["have"][[h for h, _ in pick]]
    a, mg = assign_soft(Fm, 1.0, 5)[:, 0], margins(Fm)
    present = np.zeros(H, dtype=bool)
    present[[h for h, _ in pick]] = True
    out = {"present": present, "right": float((a == truth).mean()), "diff": {}, "lent": {}, "lent_right": {}}
    for m0 in m0s:
        aug, n = lend(P, G["ref"], qs, a, mg, m0)
        s1, _h = own_scores(fuse_v(P, aug, G["one"], G["vecs"], G["row"]), G["have"])
        out["diff"][m0] = s1 - G["base"]
        out["lent"][m0] = n
        out["lent_right"][m0] = float((a == truth)[mg >= m0].mean()) if n else float("nan")
    return d, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=18)
    ap.add_argument("--m0", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    torch.set_num_threads(1)  # the workers are forked: no CUDA and no thread pool in the parent
    T = Path(a.tests_dir)
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    comps = list(raw["components"])
    assert comps[-1] == "seq"
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(a.store)
    weight = {}
    for x in raw["seq_official"]:
        key = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0] if isinstance(x, list) else x).name).groups()
        weight[key] = weight.get(key, 0) + 1
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    rng = np.random.default_rng(9955)
    out, keep = {"doc": a.doc, "draws": a.draws, "m0": a.m0, "sides": {}}, {}
    for side, tag in (("pub", "public"), ("priv", "private")):
        nets = list(weight)
        first = load(Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{nets[0][0]}{nets[0][1]}_t{a.step}.npz")
        cands, questions, owner, sgfs = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(first), a.seed)
        players = sorted(cands)
        truth_all = np.array([players.index(p) for p in owner])
        size = np.array([len(q) for q in questions])
        keys = [k for p in players for k, _ in cands[p]] + [k for q in questions for k, _ in q]
        row = {k: i for i, k in enumerate(keys)}
        vecs = []
        for v, tg in nets:
            one = first if (v, tg) == nets[0] else load(Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{v}{tg}_t{a.step}.npz")
            vecs.append(np.stack([one[k] for k in keys]))
        P = SimpleNamespace(moves=parse_all(sgfs, a.workers), sgfs=sgfs, cfg=raw["cfg"], store=store, emb=emb, comps=comps, other=[c for c in comps if c != "seq"],
                            w=np.asarray(raw["weights"], dtype=float), wts=[weight[k] for k in nets], players=players, C=len(players))
        qrow = {}
        for j, (o, m) in enumerate(zip(truth_all, size)):
            qrow.setdefault((int(o), int(m)), j)  # the first question of that size of the candidate
        have = np.array([i for i in range(len(players)) if all((i, m) in qrow for m in (1, *SIZES))])
        one_q = [questions[qrow[(int(i), 1)]] for i in have]
        multi = {m: [questions[qrow[(int(i), m)]] for i in have] for m in SIZES}
        F1 = fuse_v(P, cands, one_q, vecs, row)
        base, hit = own_scores(F1, have)
        Fmulti = {m: fuse_v(P, cands, multi[m], vecs, row) for m in SIZES}
        print(f"{side}: {len(players)} candidates ({len(have)} with questions of every size), {len(nets)} networks; 1-game score {base.mean():.4f} (Top-1 {hit.mean():.3f}); "
              + "multi-game Top-1 before the assignment " + " ".join(f"n={m} {(Fmulti[m].argmax(1) == have).mean():.3f}" for m in SIZES) + f" ({time.time() - t0:.0f} s)", flush=True)
        G.update(P=P, have=have, ref=cands, one=one_q, multi=multi, Fmulti=Fmulti, vecs=vecs, row=row, base=base)
        with mp.get_context("fork").Pool(min(a.procs, a.draws)) as ex:
            got = dict(ex.map(draw, [(d, a.m0, side) for d in range(a.draws)]))
        res = {"candidates": int(len(have)), "base": float(base.mean()), "assigned_right": float(np.mean([got[d]["right"] for d in got])), "m0": {}}
        pr = np.stack([got[d]["present"] for d in range(a.draws)])
        for m0 in a.m0:
            dd = np.stack([got[d]["diff"][m0] for d in range(a.draws)])
            cp, ca = pr.sum(0), (~pr).sum(0)
            dp, da = (dd * pr).sum(0) / np.maximum(cp, 1), (dd * ~pr).sum(0) / np.maximum(ca, 1)
            r = {"all": boot(dd.mean(0), rng), "present": boot(dp[cp > 0], rng), "absent": boot(da[ca > 0], rng), "draws_up": int((dd.mean(1) > 0).sum()),
                 "lent": float(np.mean([got[d]["lent"][m0] for d in got])), "lent_right": float(np.nanmean([got[d]["lent_right"][m0] for d in got]))}
            res["m0"][f"{m0:g}"] = r
            keep.setdefault(m0, []).append(dd.mean(0))
            print(f"  m0 {m0:g}: {r['lent']:.0f} of {len(have) // 2} questions lent ({r['lent_right']:.3f} of them to the owner) | all 1-game questions {r['all'][0]:+.4f} [{r['all'][1]:+.4f}, {r['all'][2]:+.4f}]  "
                  f"owner has a multi-game question {r['present'][0]:+.4f} [{r['present'][1]:+.4f}, {r['present'][2]:+.4f}]  has none {r['absent'][0]:+.4f} [{r['absent'][1]:+.4f}, {r['absent'][2]:+.4f}]  "
                  f"draws up {r['draws_up']}/{a.draws}", flush=True)
        out["sides"][side] = res
    out["both"] = {}
    for m0, arrs in sorted(keep.items()):
        both = boot(np.concatenate(arrs), rng)
        out["both"][f"{m0:g}"] = both
        print(f"both sides, m0 {m0:g}: {both[0]:+.4f} [{both[1]:+.4f}, {both[2]:+.4f}] per 1-game question", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
