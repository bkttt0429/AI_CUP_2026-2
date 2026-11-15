#!/usr/bin/env python
"""Probe T3 on the OLD-POOL simulation (METHOD_NOTES 2.127, part O): the lending of probe T2 with what was learnt since.

T2 (scripts/pid_probe_aug_old.py) lent the games of the confidently assigned multi-game questions to their Top-1 candidates
and re-scored the 1-game questions: +0.02 where the owner has a multi-game question, -0.006 where he has none, because an
enlarged candidate scores higher on EVERY query. Two changes:
  (1) structure (probe S3): the owner of a 1-game question never owns a 3-game question. So the 3-game questions lend
      nothing, their owners have no 1-game question here, and the prior of the candidate file (log(w / |A3|) on the
      candidates assigned to the 3-game questions, times T) is part of the baseline and of every variant;
  (2) de-biasing: after lending, every candidate's column of the [1-game queries x candidates] matrix is put back where it
      was - the median over the queries of (after - before) is removed ("shift"), or the column's median and inter-quartile
      range are restored ("scale"). No answers are used: at most one or two of a candidate's queries are his own.
Pools of scripts/pid_v2_oldsim.py (official-population players no network was trained on; 10-game candidate sets, true
answers). Per draw half of the candidates own a multi-game question (a third each of 5 / 3 / 2 games, distinct owners),
assigned as the pipeline assigns them (soft, Top-1 at --lam); the 1-game queries are the first 1-game question of every
candidate that owns no 3-game question. Reported per variant, against "prior only": the score difference of the queries
whose owner has a 2-game question, a 5-game question, none, and their mixture at the estimated real shares
(0.172 / 0.172 / 0.656, probe S3).
  python -I <this> --doc v2o16c_b --out results/final/v2_probe_aug3_old_v2o16c_b.json
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
from pid_probe_overlap import log_prior  # noqa: E402
from pid_probe_qq import boot, own_scores  # noqa: E402

G = {}
SIZES = (5, 3, 2)
SHARES = {2: 0.172, 5: 0.172, 0: 0.656}  # probe S3: where the owner of a real 1-game question is
VARIANTS = [(src, how) for src in ("all", "25") for how in ("none", "shift", "scale")]


def debias(Fa, Fb, how):
    """Fa [Q, C] after lending, Fb before: every candidate's column put back (see the module text)."""
    if how == "none":
        return Fa
    if how == "shift":
        return Fa - np.median(Fa - Fb, axis=0, keepdims=True)
    qa, qb = np.percentile(Fa, [25, 50, 75], axis=0), np.percentile(Fb, [25, 50, 75], axis=0)
    return qb[1][None] + (Fa - qa[1][None]) * ((qb[2] - qb[0]) / np.maximum(qa[2] - qa[0], 1e-9))[None]


def draw(task):
    d, m0s, side = task
    P, H = G["P"], len(G["have"])
    rng = np.random.default_rng(9970 + 13 * d + (0 if side == "pub" else 1))
    perm = rng.permutation(H)
    n5 = n3 = H // 6
    n2 = H // 2 - n5 - n3
    pick = [(int(h), 5) for h in perm[:n5]] + [(int(h), 3) for h in perm[n5:n5 + n3]] + [(int(h), 2) for h in perm[n5 + n3:n5 + n3 + n2]]
    hh, sz = np.array([h for h, _ in pick]), np.array([m for _, m in pick])
    Fm = np.stack([G["Fmulti"][m][h] for h, m in pick])
    qs = [G["multi"][m][h] for h, m in pick]
    truth = G["have"][hh]
    a, mg = assign_soft(Fm, G["lam"], 5)[:, 0], margins(Fm)
    cls = np.zeros(H, dtype=int)  # per candidate with questions: 0 = owns no multi-game question, else its size
    cls[hh] = sz
    qi = np.flatnonzero(cls != 3)  # the 1-game queries: every candidate but the owners of the 3-game questions
    own, one, Fb = G["have"][qi], [G["one"][i] for i in qi], G["F1"][qi]
    in3 = np.zeros(P.C, dtype=bool)
    in3[a[sz == 3]] = True  # the candidates the 3-game questions are assigned to
    lp = G["T"] * log_prior(in3, G["w3"])[None]
    s_plain, _h = own_scores(Fb, own)
    s_prior, _h = own_scores(Fb + lp, own)
    out = {"qi": qi, "cls": cls[qi], "right": float((a == truth).mean()), "prior": s_prior - s_plain, "diff": {}, "lent": {}, "lent_right": {}}
    for m0 in m0s:
        for src in ("all", "25"):
            skip = None if src == "all" else sz == 3
            aug, n = lend(P, G["ref"], qs, a, mg, m0, skip=skip)
            Fa = fuse_v(P, aug, one, G["vecs"], G["row"])
            ok = (mg >= m0) & (sz != 3 if src == "25" else True)
            out["lent"][(m0, src)] = n
            out["lent_right"][(m0, src)] = float((a == truth)[ok].mean()) if n else float("nan")
            for how in ("none", "shift", "scale"):
                s1, _h = own_scores(debias(Fa, Fb, how) + lp, own)
                out["diff"][(m0, src, how)] = s1 - s_prior
    return d, out


def by_class(got, key, H, rng):
    """Per candidate the mean over the draws in which he was of a class; the classes' means and their mixture at the real shares."""
    acc = {c: np.zeros(H) for c in SHARES}
    cnt = {c: np.zeros(H) for c in SHARES}
    for o in got.values():
        x = key(o)
        for c in SHARES:
            m = o["cls"] == c
            acc[c][o["qi"][m]] += x[m]
            cnt[c][o["qi"][m]] += 1
    full = np.all([cnt[c] > 0 for c in SHARES], axis=0)
    mean = {c: acc[c] / np.maximum(cnt[c], 1) for c in SHARES}
    mix = sum(SHARES[c] * mean[c] for c in SHARES)[full]
    return {"own2": boot(mean[2][cnt[2] > 0], rng), "own5": boot(mean[5][cnt[5] > 0], rng), "none": boot(mean[0][cnt[0] > 0], rng), "mixture": boot(mix, rng)}, mix


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=36)
    ap.add_argument("--m0", nargs="+", type=float, default=[0.0, 2.0])
    ap.add_argument("--lam", type=float, default=2.0, help="soft assignment of the multi-game questions (the candidate files: 2)")
    ap.add_argument("--w3", type=float, default=0.05)
    ap.add_argument("--prior-T", type=float, default=1.6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=12)
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
    rng = np.random.default_rng(9975)
    out, keep = {"doc": a.doc, "draws": a.draws, "m0": a.m0, "lam": a.lam, "w3": a.w3, "prior_T": a.prior_T, "shares": {str(k): v for k, v in SHARES.items()}, "sides": {}}, {}
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
        H = len(have)
        print(f"{side}: {len(players)} candidates ({H} with questions of every size), {len(nets)} networks; 1-game score {base.mean():.4f} (Top-1 {hit.mean():.3f}) ({time.time() - t0:.0f} s)", flush=True)
        G.update(P=P, have=have, ref=cands, one=one_q, multi=multi, Fmulti=Fmulti, vecs=vecs, row=row, F1=F1, lam=a.lam, T=a.prior_T, w3=a.w3)
        with mp.get_context("fork").Pool(min(a.procs, a.draws)) as ex:
            got = dict(ex.map(draw, [(d, a.m0, side) for d in range(a.draws)]))
        res = {"candidates": H, "base": float(base.mean()), "assigned_right": float(np.mean([o["right"] for o in got.values()])), "variants": {}}
        r, _m = by_class(got, lambda o: o["prior"], H, rng)
        res["prior_only_vs_plain"] = r
        print(f"  the prior alone against no prior: owner has a 2-game question {r['own2'][0]:+.4f}, a 5-game question {r['own5'][0]:+.4f}, none {r['none'][0]:+.4f}; "
              f"mixture {r['mixture'][0]:+.4f} [{r['mixture'][1]:+.4f}, {r['mixture'][2]:+.4f}]", flush=True)
        for m0 in a.m0:
            for src, how in VARIANTS:
                r, mix = by_class(got, lambda o: o["diff"][(m0, src, how)], H, rng)
                r["lent"] = float(np.mean([o["lent"][(m0, src)] for o in got.values()]))
                r["lent_right"] = float(np.nanmean([o["lent_right"][(m0, src)] for o in got.values()]))
                r["draws_up"] = int(sum(float(sum(SHARES[c] * o["diff"][(m0, src, how)][o["cls"] == c].mean() for c in SHARES)) > 0 for o in got.values()))
                name = f"m0 {m0:g} | lend {'2/3/5' if src == 'all' else '2/5'} | {how}"
                res["variants"][name] = r
                keep.setdefault(name, []).append(mix)
                print(f"  {name}: {r['lent']:.0f} lent ({r['lent_right']:.3f} to the owner) | against the prior alone: own 2-game {r['own2'][0]:+.4f} [{r['own2'][1]:+.4f}, {r['own2'][2]:+.4f}]  "
                      f"own 5-game {r['own5'][0]:+.4f} [{r['own5'][1]:+.4f}, {r['own5'][2]:+.4f}]  none {r['none'][0]:+.4f} [{r['none'][1]:+.4f}, {r['none'][2]:+.4f}]  "
                      f"mixture {r['mixture'][0]:+.4f} [{r['mixture'][1]:+.4f}, {r['mixture'][2]:+.4f}]  draws up {r['draws_up']}/{a.draws}", flush=True)
        out["sides"][side] = res
    out["both"] = {}
    for name, arrs in keep.items():
        both = boot(np.concatenate(arrs), rng)
        out["both"][name] = both
        print(f"both sides, {name}: mixture {both[0]:+.4f} [{both[1]:+.4f}, {both[2]:+.4f}] per 1-game question", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
