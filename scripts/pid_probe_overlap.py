#!/usr/bin/env python
"""Probe S (METHOD_NOTES 2.123): how many owners of the 1-game questions also own a multi-game question, and what a prior
built on that share is worth.

The questions of 2 / 3 / 5 games have distinct owners (2.37); the 1-game questions were taken as independent draws (half of
their owners would then own a multi-game question). Probe T2 found the Top-1 of the real 1-game questions among the candidates
assigned to a multi-game question only 35-40 times in 100, not 48-50.
Estimate (no answers): P_q(c) = softmax(F_qc / T) for a 1-game question q (T fitted on the pseudo-pools' true answers),
A = the candidates assigned to the multi-game questions, m_q = sum over A of P_q(c). With pi = P(owner in A):
  log L(pi) = sum_q log(pi * m_q / |A| + (1 - pi) * (1 - m_q) / (C - |A|)).
Use: F_qc + T * log prior, prior = pi / |A| inside A and (1 - pi) / (C - |A|) outside; only the 1-game questions change.
Part 1 (pseudo-pools, true answers): T; the estimator on planted structures (a share p of 100 one-game owners taken from the
  300 owners of a 2-game question); the gain of the prior for every pair (planted p, pi used).
Part 2 (real pools): pi-hat with its likelihood-ratio interval, per pool and pooled; shares by size of the multi-game question.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_overlap_v2o13c_9.json
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
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_aug import fuse_v  # noqa: E402
from pid_probe_qq import Pool, own_scores  # noqa: E402

G = {}
GRID = np.linspace(0.001, 0.999, 999)


def softmax(F, T):
    z = (F - F.max(axis=1, keepdims=True)) / T
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def pi_mle(m, nA, C):
    """(pi-hat, lower, upper of the 95% likelihood-ratio interval) from the posterior masses m [Q] on a set of nA of C candidates."""
    ll = np.log(GRID[:, None] * m[None] / nA + (1 - GRID[:, None]) * (1 - m[None]) / (C - nA) + 1e-300).sum(1)
    ok = GRID[ll >= ll.max() - 1.92]
    return float(GRID[ll.argmax()]), float(ok.min()), float(ok.max())


def log_prior(inA, pi):
    nA, C = int(inA.sum()), len(inA)
    return np.where(inA, np.log(max(pi, 1e-6) / nA), np.log(max(1 - pi, 1e-6) / (C - nA)))


def pseudo_scores(task):
    s, ft_dir, pool = task
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
    return s, fuse_v(P, ref, one, vecs, row).astype(np.float32), fuse_v(P, ref, two, vecs, row).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--planted", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--used", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    torch.set_num_threads(1)
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    out = {"doc": a.doc, "pools": {}}
    pooled = {"m": [], "nA": [], "C": []}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        C = P.C
        G.update(P=P)
        with mp.get_context("fork").Pool(len(a.seeds)) as ex:
            got = {s: (f1, f2) for s, f1, f2 in ex.map(pseudo_scores, [(s, a.ft_dir, pool) for s in a.seeds])}
        own = np.arange(C)
        # ---- temperature: the true owners of the pseudo-pools' 1-game questions under a uniform prior
        Ts = np.exp(np.linspace(np.log(0.3), np.log(6.0), 61))
        nll = [float(np.mean([-np.log(softmax(got[s][0], T)[own, own] + 1e-300).mean() for s in a.seeds])) for T in Ts]
        T = float(Ts[int(np.argmin(nll))])
        res = {"T": T, "nll_at_T": float(min(nll)), "planted": {}, "gain": {}}
        print(f"{pool}: temperature {T:.2f} (mean -log P of the owner {min(nll):.3f}; at T = 1: {nll[int(np.argmin(np.abs(Ts - 1.0)))]:.3f}) ({time.time() - t0:.0f} s)", flush=True)
        # ---- planted structures
        rng = np.random.default_rng(9600 + (0 if pool == "pub2" else 1))
        for p in a.planted:
            est, true_pi, share, gains = [], [], [], {u: [] for u in a.used}
            oracle = []
            for s in a.seeds:
                F1, F2 = got[s]
                Pq = softmax(F1, T)
                for _ in range(a.draws // len(a.seeds)):
                    M = rng.permutation(C)[:C // 2]
                    A = np.unique(assign_soft(F2[M], 1.0, 5)[:, 0])
                    inA = np.zeros(C, dtype=bool)
                    inA[A] = True
                    k = int(round(100 * p))
                    rest = np.setdiff1d(own, M)
                    sel = np.concatenate([rng.choice(M, k, replace=False), rng.choice(rest, 100 - k, replace=False)])
                    m = Pq[sel][:, inA].sum(1)
                    ph, _lo, _hi = pi_mle(m, int(inA.sum()), C)
                    est.append(ph)
                    true_pi.append(float(inA[sel].mean()))
                    share.append(float(inA[F1[sel].argmax(1)].mean()))
                    base, _h = own_scores(F1[sel], sel)
                    for u in a.used:
                        s1, _h = own_scores(F1[sel] + T * log_prior(inA, u)[None], sel)
                        gains[u].append(float((s1 - base).mean()))
                    s2, _h = own_scores(F1[sel] + T * log_prior(inA, float(np.clip(ph, 0.02, 0.98)))[None], sel)
                    oracle.append(float((s2 - base).mean()))
            r = {"true_pi": float(np.mean(true_pi)), "estimate_mean": float(np.mean(est)), "estimate_sd": float(np.std(est)), "top1_share_in_A": float(np.mean(share)),
                 "gain_by_pi_used": {f"{u:g}": [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))] for u, v in gains.items()},
                 "gain_with_own_estimate": [float(np.mean(oracle)), float(np.std(oracle) / np.sqrt(len(oracle)))]}
            res["planted"][f"{p:g}"] = r
            print(f"  planted share {p:g}: true P(owner in A) {r['true_pi']:.3f}, estimate {r['estimate_mean']:.3f} (sd of one pool of 100 questions {r['estimate_sd']:.3f}); Top-1 in A {r['top1_share_in_A']:.3f} | "
                  f"gain per 1-game question with pi used = " + "  ".join(f"{u}: {v[0]:+.4f}" for u, v in r["gain_by_pi_used"].items())
                  + f"  own estimate: {r['gain_with_own_estimate'][0]:+.4f} (se {r['gain_with_own_estimate'][1]:.4f})", flush=True)
        # ---- the real pool
        F = fuse_v(P, P.ref, P.qs, P.vec["t50"], P.row)
        n = np.array([len(q) for q in P.qs])
        big, one = np.flatnonzero(n > 1), np.flatnonzero(n == 1)
        a_big = assign_soft(F[big], 1.0, 5)[:, 0]
        inA = np.zeros(C, dtype=bool)
        inA[a_big] = True
        Pq = softmax(F[one], T)
        m = Pq[:, inA].sum(1)
        ph, lo, hi = pi_mle(m, int(inA.sum()), C)
        top2 = np.sort(F[one], axis=1)[:, -2:]
        mg = top2[:, 1] - top2[:, 0]
        t1 = F[one].argmax(1)
        res["real"] = {"one_game_questions": int(len(one)), "A": int(inA.sum()), "pi_hat": ph, "pi_95": [lo, hi], "independent_value": float(inA.sum() / C),
                       "top1_in_A": {f"{t:g}": [int(inA[t1[mg > t]].sum()), int((mg > t).sum())] for t in (0.0, 1.0, 2.0, 3.0)}, "mean_mass_on_A": float(m.mean())}
        # by the size of the multi-game question: mixture weights of the four groups (EM)
        groups = {str(sz): np.unique(a_big[n[big] == sz]) for sz in (2, 3, 5)}
        gm = np.stack([Pq[:, g].sum(1) / len(g) for g in groups.values()] + [Pq[:, ~inA].sum(1) / int((~inA).sum())], axis=1)
        w = np.full(4, 0.25)
        for _ in range(500):
            r_ = w[None] * gm
            r_ /= r_.sum(1, keepdims=True)
            w = r_.mean(0)
        res["real"]["weights_by_group"] = {**{f"owners of {k}-game questions": [float(x), float(len(g) / C)] for (k, g), x in zip(groups.items(), w[:3])}, "no multi-game question": [float(w[3]), float((~inA).sum() / C)]}
        pooled["m"].append(m); pooled["nA"].append(int(inA.sum())); pooled["C"].append(C)
        print(f"{pool} REAL: {len(one)} 1-game questions, {int(inA.sum())} candidates assigned to a multi-game question (independent draws would give pi = {inA.sum() / C:.3f}); "
              f"pi-hat {ph:.3f} [{lo:.3f}, {hi:.3f}]; Top-1 in A by margin > 0/1/2/3: " + "  ".join(f"{v[0]}/{v[1]}" for v in res["real"]["top1_in_A"].values())
              + "; weights (estimate vs independent): " + "  ".join(f"{k} {v[0]:.3f} vs {v[1]:.3f}" for k, v in res["real"]["weights_by_group"].items()), flush=True)
        out["pools"][pool] = res
    # both pools: one pi (the groups have nearly the same sizes)
    ll = sum(np.log(GRID[:, None] * m[None] / nA + (1 - GRID[:, None]) * (1 - m[None]) / (C - nA) + 1e-300).sum(1) for m, nA, C in zip(pooled["m"], pooled["nA"], pooled["C"]))
    ok = GRID[ll >= ll.max() - 1.92]
    out["both"] = {"pi_hat": float(GRID[ll.argmax()]), "pi_95": [float(ok.min()), float(ok.max())], "loglik_gain_over_half": float(ll.max() - ll[int(np.argmin(np.abs(GRID - 0.5)))])}
    print(f"both pools REAL: pi-hat {out['both']['pi_hat']:.3f} [{ok.min():.3f}, {ok.max():.3f}]; log-likelihood above pi = 0.5: {out['both']['loglik_gain_over_half']:.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
