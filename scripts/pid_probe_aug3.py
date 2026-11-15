#!/usr/bin/env python
"""Probe T3 on the official pools (METHOD_NOTES 2.127, parts P and F): the lending of probe T2 with what was learnt since.

See scripts/pid_probe_aug3_old.py for the two changes (the 3-game questions lend nothing and their owners have no 1-game
question; every candidate's column is put back where it was after lending) and for the de-biasing itself.
Part P (official pseudo-pools, true answers): per candidate 3 games held out; the first is his 1-game question, the other
  two a 2-game question. Per draw half of the candidates own a multi-game question; a third of those stand in for the
  3-game questions (they lend nothing, their owners' 1-game questions are left out, the prior of the candidate file sits on
  the candidates they are assigned to); the rest lend when their margin is at least m0. Real multi-game questions have
  2 / 3 / 5 games, these have 2: conservative. Reported against "prior only": the score difference of the 1-game questions
  whose owner has a lending question, has none, and their mixture at the estimated real shares (0.344 / 0.656, probe S3).
Part F (real pools, no answers): what each variant changes in the real 1-game questions.
  python -I <this> --doc v2o16c_b --out results/final/v2_probe_aug3_v2o16c_b.json
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import topk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_aug import fuse_v, lend, margins  # noqa: E402
from pid_probe_aug3_old import VARIANTS, debias  # noqa: E402
from pid_probe_overlap import log_prior  # noqa: E402
from pid_probe_qq import Pool, boot, own_scores  # noqa: E402

G = {}
SHARES = {2: 0.344, 0: 0.656}  # probe S3: the owner of a real 1-game question owns a 2- or 5-game question / none


def pseudo(task):
    """One hold-out seed of the pseudo-pool of G['P']: per draw the score differences of the 1-game questions, per variant."""
    s, draws, m0s, ft_dir, pool = task
    P = G["P"]
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    one, two = [[hold[p][0]] for p in P.players], [list(hold[p][1:3]) for p in P.players]
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    vecs = []
    for key in P.net_keys:
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        vecs.append(np.stack([d[k] for k in ckeys]))
    F1, F2 = fuse_v(P, ref, one, vecs, row), fuse_v(P, ref, two, vecs, row)
    rng = np.random.default_rng(9820 + 31 * s + (0 if pool == "pub2" else 1))
    k3, outs = P.C // 6, []
    for _ in range(draws):
        M = rng.permutation(P.C)[:P.C // 2]  # the owners of the multi-game questions, in random order
        is3 = np.arange(len(M)) < k3  # the first third stands in for the 3-game questions
        a, mg = assign_soft(F2[M], G["lam"], 5)[:, 0], margins(F2[M])
        cls = np.zeros(P.C, dtype=int)
        cls[M[~is3]], cls[M[is3]] = 2, 3
        qi = np.flatnonzero(cls != 3)  # the 1-game queries; the candidate index is also the true answer
        in3 = np.zeros(P.C, dtype=bool)
        in3[a[is3]] = True
        lp = G["T"] * log_prior(in3, G["w3"])[None]
        Fb, q1 = F1[qi], [one[i] for i in qi]
        s_plain, _h = own_scores(Fb, qi)
        s_prior, _h = own_scores(Fb + lp, qi)
        o = {"qi": qi, "cls": cls[qi], "right": float((a == M).mean()), "prior": s_prior - s_plain, "diff": {}, "lent": {}, "lent_right": {}}
        for m0 in m0s:
            for src in ("all", "25"):
                aug, n = lend(P, ref, [two[i] for i in M], a, mg, m0, skip=None if src == "all" else is3)
                Fa = fuse_v(P, aug, q1, vecs, row)
                ok = (mg >= m0) & (~is3 if src == "25" else True)
                o["lent"][(m0, src)] = n
                o["lent_right"][(m0, src)] = float((a == M)[ok].mean()) if n else float("nan")
                for how in ("none", "shift", "scale"):
                    s1, _h = own_scores(debias(Fa, Fb, how) + lp, qi)
                    o["diff"][(m0, src, how)] = s1 - s_prior
        outs.append(o)
    return s, outs


def by_class(outs, key, C, rng):
    """Per candidate the mean over the draws in which he was of a class; the classes' means and their mixture at the real shares."""
    acc = {c: np.zeros(C) for c in SHARES}
    cnt = {c: np.zeros(C) for c in SHARES}
    for o in outs:
        x = key(o)
        for c in SHARES:
            m = o["cls"] == c
            acc[c][o["qi"][m]] += x[m]
            cnt[c][o["qi"][m]] += 1
    full = np.all([cnt[c] > 0 for c in SHARES], axis=0)
    mean = {c: acc[c] / np.maximum(cnt[c], 1) for c in SHARES}
    mix = sum(SHARES[c] * mean[c] for c in SHARES)[full]
    return {"lending": boot(mean[2][cnt[2] > 0], rng), "none": boot(mean[0][cnt[0] > 0], rng), "mixture": boot(mix, rng)}, mix


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--m0", nargs="+", type=float, default=[0.0, 2.0])
    ap.add_argument("--lam", type=float, default=2.0, help="soft assignment of the multi-game questions (the candidate files: 2)")
    ap.add_argument("--w3", type=float, default=0.05)
    ap.add_argument("--prior-T", type=float, default=1.6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    torch.set_num_threads(1)  # the workers are forked: no CUDA and no thread pool in the parent
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    rng = np.random.default_rng(9797)
    out = {"doc": a.doc, "m0": a.m0, "lam": a.lam, "w3": a.w3, "prior_T": a.prior_T, "shares": {str(k): v for k, v in SHARES.items()}, "pools": {}}
    keep = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        G.update(P=P, lam=a.lam, T=a.prior_T, w3=a.w3)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            ps = dict(ex.map(pseudo, [(s, a.draws, a.m0, a.ft_dir, pool) for s in a.seeds]))
        allo = [o for s in a.seeds for o in ps[s]]
        res = {"P": {}, "F": {}}
        r, _m = by_class(allo, lambda o: o["prior"], P.C, rng)
        res["P"]["prior_only_vs_plain"] = r
        print(f"{pool} part P: stand-in and lending questions assigned to their owner {np.mean([o['right'] for o in allo]):.3f}; the prior alone against no prior: "
              f"owner has a lending question {r['lending'][0]:+.4f}, none {r['none'][0]:+.4f}, mixture {r['mixture'][0]:+.4f} [{r['mixture'][1]:+.4f}, {r['mixture'][2]:+.4f}] ({time.time() - t0:.0f} s)", flush=True)
        for m0 in a.m0:
            for src, how in VARIANTS:
                k = (m0, src, how)
                r, mix = by_class(allo, lambda o: o["diff"][k], P.C, rng)
                r["per_seed"] = [by_class(ps[s], lambda o: o["diff"][k], P.C, rng)[0]["mixture"][0] for s in a.seeds]
                r["lent"] = float(np.mean([o["lent"][(m0, src)] for o in allo]))
                r["lent_right"] = float(np.nanmean([o["lent_right"][(m0, src)] for o in allo]))
                name = f"m0 {m0:g} | lend {'2/3/5' if src == 'all' else '2/5'} | {how}"
                res["P"][name] = r
                keep.setdefault(name, []).append(mix)
                print(f"  {name}: {r['lent']:.0f} lent ({r['lent_right']:.3f} to the owner) | against the prior alone: owner has a lending question {r['lending'][0]:+.4f} [{r['lending'][1]:+.4f}, {r['lending'][2]:+.4f}]  "
                      f"none {r['none'][0]:+.4f} [{r['none'][1]:+.4f}, {r['none'][2]:+.4f}]  mixture {r['mixture'][0]:+.4f} [{r['mixture'][1]:+.4f}, {r['mixture'][2]:+.4f}]  per seed "
                      + " ".join(f"{x:+.4f}" for x in r["per_seed"]), flush=True)
        # ---- part F: the real 1-game questions
        F = fuse_v(P, P.ref, P.qs, P.vec["t50"], P.row)
        n = np.array([len(q) for q in P.qs])
        big, one = np.flatnonzero(n > 1), np.flatnonzero(n == 1)
        nb = n[big]
        a_big, mg_big = assign_soft(F[big], a.lam, 5)[:, 0], margins(F[big])
        in3 = np.zeros(P.C, dtype=bool)
        in3[a_big[nb == 3]] = True
        lp = a.prior_T * log_prior(in3, a.w3)[None]
        q_one, q_big, Fb = [P.qs[int(j)] for j in one], [P.qs[int(j)] for j in big], F[one]
        t5 = topk(Fb + lp, 5)
        for m0 in a.m0:
            for src in ("all", "25"):
                skip = None if src == "all" else nb == 3
                aug, k = lend(P, P.ref, q_big, a_big, mg_big, m0, skip=skip)
                Fa = fuse_v(P, aug, q_one, P.vec["t50"], P.row)
                lent_to = np.zeros(P.C, dtype=bool)
                lent_to[a_big[(mg_big >= m0) & (nb != 3 if src == "25" else True)]] = True
                for how in ("none", "shift", "scale"):
                    n5 = topk(debias(Fa, Fb, how) + lp, 5)
                    name = f"m0 {m0:g} | lend {'2/3/5' if src == 'all' else '2/5'} | {how}"
                    r = {"lent": int(k), "top1_changed": int((n5[:, 0] != t5[:, 0]).sum()), "top5_changed": int((n5 != t5).any(1).sum()),
                         "top1_enlarged_before": int(lent_to[t5[:, 0]].sum()), "top1_enlarged_after": int(lent_to[n5[:, 0]].sum()), "top1_on_3game_after": int(in3[n5[:, 0]].sum())}
                    res["F"][name] = r
                    print(f"{pool} part F {name}: {k} of {len(big)} multi-game questions lent; of the {len(one)} 1-game questions {r['top1_changed']} change their Top-1, {r['top5_changed']} their Top-5; "
                          f"Top-1 is an enlarged candidate: {r['top1_enlarged_before']} before, {r['top1_enlarged_after']} after", flush=True)
        out["pools"][pool] = res
    out["both"] = {}
    for name, arrs in keep.items():
        both = boot(np.concatenate(arrs), rng)
        out["both"][name] = both
        print(f"both pools, part P, {name}: mixture {both[0]:+.4f} [{both[1]:+.4f}, {both[2]:+.4f}] per 1-game question", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
