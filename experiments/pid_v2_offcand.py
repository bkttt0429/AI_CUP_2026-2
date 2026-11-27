#!/usr/bin/env python
"""Which sequence networks identify OFFICIAL players best? Leave-games-out among each added pool's own candidates
(METHOD_NOTES 2.40). The organisers give the owner of every candidate game (600 players x 10 games per pool; candidate games
may be used for calibration, Q&A Q10), so a candidate's games can be split into a labelled question and a reference:
  n = 1   every candidate game is a question; its owner's prototype = normalised sum of the other nine games
  n = 3   the candidate's first three games (file order) are one question; prototype = the other seven
All other candidates keep their ten games. Vectors BEFORE the pool fine-tune only (the fine-tune trains on these games).
Pools are never mixed. Scores: e^-(rank-1) over the top five, as the competition, and top-1.

Networks of the official files: the rank session's M88 (one seed) and M90 (three seeds), each with net -1 (trained on every
training player: "full") and nets 1-4 (each misses one fold: "cross"). Official players are unseen by all of them, so a cross
network is simply a network with fewer training players. Reported per pool:
  every network alone; the ensembles (mean of row-standardised matrices) full4 / full3 (M90) / cross16 / cross12 (M90) / all20;
  all20 with the full networks weighted w times; M90 size curves (mean over subsets) for full and cross networks and their
  1/N fits; equal-size mixtures (k full + (3-k) cross of M90). Differences with a bootstrap over candidates.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401  (puts the AI_CUP_2026-1 package on the path)
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402

unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731


def matrices(E, owner, C, n):
    """Row-standardised [Q, C] score matrix of one network. E [G, 128] game vectors, owner [G] candidate index.
    n = 1: Q = G (every game); n = 3: Q = C (the first three games of every candidate)."""
    P = np.zeros((C, E.shape[1]))
    np.add.at(P, owner, E)
    if n == 1:
        q, truth = unit(E), owner
        S = q @ unit(P).T
        S[np.arange(len(E)), owner] = np.einsum("ij,ij->i", q, unit(P[owner] - E))
    else:
        first = np.zeros(len(E), bool)
        seen = np.zeros(C, int)
        for i, c in enumerate(owner):
            first[i] = seen[c] < n
            seen[c] += 1
        Qs = np.zeros((C, E.shape[1]))
        np.add.at(Qs, owner[first], E[first])
        truth = np.arange(C)
        S = unit(Qs) @ unit(P).T
        S[truth, truth] = np.einsum("ij,ij->i", unit(Qs), unit(P - Qs))
    return standardise(S), truth


def per_question(M, truth):
    rank = (M > M[np.arange(len(truth)), truth][:, None]).sum(1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0), (rank == 0).astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--subsets", type=int, default=40, help="random subsets per size of the cross-network size curve")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float64))))(np.load(f))  # noqa: E731
    m88 = {v: load(ROOT / "data" / "seq" / ("official_m88.npz" if v == -1 else f"official_m88_net{v}.npz")) for v in (-1, 1, 2, 3, 4)}
    out = {"pools": {}}
    for tag, prefix in (("public", "pub2"), ("private", "priv2")):
        cands, _, _, _ = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        keys = [k for p in players for k, _ in cands[p]]
        owner = np.array([i for i, p in enumerate(players) for _ in cands[p]])
        C = len(players)
        nets = {f"m88_{v}": m88[v] for v in m88}
        nets.update({f"m90s{k}_{v}": load(ROOT / "data" / "seq_ft" / f"official_{prefix}_net{v}_m90s{k}e_t0.npz") for k in (0, 1, 2) for v in (-1, 1, 2, 3, 4)})
        full = [x for x in nets if x.endswith("_-1")]
        cross = [x for x in nets if not x.endswith("_-1")]
        full90, cross90 = [x for x in full if x.startswith("m90")], [x for x in cross if x.startswith("m90")]
        res = {"candidates": C, "games": len(keys), "n": {}}
        for n in (1, 3):
            Z, truth = {}, None
            for name, vec in nets.items():
                Z[name], truth = matrices(np.stack([vec[k] for k in keys]), owner, C, n)
            grp = truth  # questions of one candidate are resampled together
            sc = lambda names, w=None: per_question(np.average([Z[x] for x in names], axis=0, weights=w), truth)  # noqa: E731

            def boot(x, y):
                d = x - y
                per = np.bincount(grp, weights=d, minlength=C) / np.maximum(np.bincount(grp, minlength=C), 1)
                bs = per[rng.integers(0, C, (a.boot, C))].mean(1)
                return {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

            r = {"single": {x: {"score": float(sc([x])[0].mean()), "top1": float(sc([x])[1].mean())} for x in nets}, "ens": {}, "curves": {}, "diffs": {}}
            mean_of = lambda xs: float(np.mean([r["single"][x]["score"] for x in xs]))  # noqa: E731
            print(f"{prefix} n={n}: single network score (top-1): M90 full {mean_of(full90):.4f} ({np.mean([r['single'][x]['top1'] for x in full90]):.4f}), M90 cross {mean_of(cross90):.4f} "
                  f"({np.mean([r['single'][x]['top1'] for x in cross90]):.4f}); M88 full {mean_of(['m88_-1']):.4f}, M88 cross {mean_of([x for x in cross if x.startswith('m88')]):.4f}", flush=True)
            print("   M90 by network: " + "  ".join(f"{x[3:]} {r['single'][x]['score']:.3f}" for x in sorted(nets) if x.startswith("m90")), flush=True)
            ens = {"full4": full, "full3": full90, "cross16": cross, "cross12": cross90, "all20": [*full, *cross], "m90_15": [*full90, *cross90]}
            Q = {k: sc(v) for k, v in ens.items()}
            for w in (2, 3, 4, 6, 10):
                Q[f"all20_w{w}"] = sc([*full, *cross], [w] * len(full) + [1] * len(cross))
            for k, (s_, t_) in Q.items():
                r["ens"][k] = {"score": float(s_.mean()), "top1": float(t_.mean())}
            print("   ensembles: " + "  ".join(f"{k} {x['score']:.4f}" for k, x in r["ens"].items()), flush=True)
            # size curves of the M90 networks: full (all subsets of the three seeds) and cross (random subsets)
            cur = {"full": {}, "cross": {}, "cross_same_fold": {}, "cross_same_seed": {}}
            for m in (1, 2, 3):
                cur["full"][m] = float(np.mean([sc(list(s))[0].mean() for s in itertools.combinations(full90, m)]))
            for m in (1, 2, 3, 4, 6, 8, 12):
                subs = [list(rng.choice(cross90, m, replace=False)) for _ in range(a.subsets)] if m < 12 else [cross90]
                cur["cross"][m] = float(np.mean([sc(s)[0].mean() for s in subs]))
            cur["cross_same_fold"][3] = float(np.mean([sc([f"m90s{k}_{v}" for k in (0, 1, 2)])[0].mean() for v in (1, 2, 3, 4)]))  # three seeds, one training set
            cur["cross_same_seed"][4] = float(np.mean([sc([f"m90s{k}_{v}" for v in (1, 2, 3, 4)])[0].mean() for k in (0, 1, 2)]))  # four training sets, one seed
            fit = {}
            for kind in ("full", "cross"):
                x = np.array([1 / m for m in cur[kind]])
                y = np.array(list(cur[kind].values()))
                slope, lim = np.polyfit(x, y, 1)
                fit[kind] = {"limit": float(lim), "c": float(-slope)}
            r["curves"], r["fit"] = cur, fit
            print(f"   M90 size curves: full {({m: round(x, 4) for m, x in cur['full'].items()})}; cross {({m: round(x, 4) for m, x in cur['cross'].items()})}; cross 3 seeds of one fold {cur['cross_same_fold'][3]:.4f}, "
                  f"4 folds of one seed {cur['cross_same_seed'][4]:.4f}; 1/N fit: full limit {fit['full']['limit']:.4f} (c {fit['full']['c']:.4f}), cross limit {fit['cross']['limit']:.4f} (c {fit['cross']['c']:.4f})", flush=True)
            # equal size 3: k full + (3 - k) cross
            mixes = {}
            for kf in (0, 1, 2, 3):
                vals = []
                for _ in range(a.subsets):
                    s = list(rng.choice(full90, kf, replace=False)) + list(rng.choice(cross90, 3 - kf, replace=False))
                    vals.append(sc(s)[0].mean())
                mixes[kf] = float(np.mean(vals))
            r["mix3"] = mixes
            print(f"   three M90 networks, k of them full: {({k: round(x, 4) for k, x in mixes.items()})}", flush=True)
            # paired differences
            c3 = np.mean([sc(list(rng.choice(cross90, 3, replace=False)))[0] for _ in range(a.subsets)], axis=0)
            c1 = np.mean([sc([x])[0] for x in cross90], axis=0)
            f1 = np.mean([sc([x])[0] for x in full90], axis=0)
            pairs = {"full vs cross, one M90 network": (f1, c1), "full3 vs three cross (M90)": (Q["full3"][0], c3), "full4 vs all20": (Q["full4"][0], Q["all20"][0]), "cross16 vs all20": (Q["cross16"][0], Q["all20"][0]),
                     "all20_w3 vs all20": (Q["all20_w3"][0], Q["all20"][0]), "all20_w6 vs all20": (Q["all20_w6"][0], Q["all20"][0]), "all20 vs m90_15": (Q["all20"][0], Q["m90_15"][0])}
            for k, (x, y) in pairs.items():
                r["diffs"][k] = boot(x, y)
                print(f"   {k}: {r['diffs'][k]['diff']:+.4f} [{r['diffs'][k]['ci95'][0]:+.4f}, {r['diffs'][k]['ci95'][1]:+.4f}]", flush=True)
            res["n"][str(n)] = r
        out["pools"][prefix] = res
        print(f"{prefix} done ({time.time() - t0:.0f} s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
