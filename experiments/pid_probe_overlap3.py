#!/usr/bin/env python
"""Probe S3 (METHOD_NOTES 2.123): the owners of the 1-game questions are (almost) never the owners of the 3-game questions.

Probe S (scripts/pid_probe_overlap.py) found the 1-game questions' owners among the owners of the 2-game and of the 5-game
questions at the rate of independent draws, and among the owners of the 3-game questions at a rate near zero, in both added
pools. Here: (a) the real pools, no answers: w-hat = P(owner of a 1-game question is in A_k) with its likelihood-ratio
interval for A_k = the candidates assigned to the k-game questions (k = 2, 3, 5), per pool and pooled, and the counts of the
confident 1-game questions by group; what the prior would change. (c) pseudo-pools with true answers: a third of the 2-game
questions of a draw stand in for the 3-game questions (their assignment is less accurate than a real 3-game question's:
conservative); a share w_true of the 100 one-game owners is taken from their owners; gain of the prior for every w_used.
Prior: log(w / |A|) inside A, log((1 - w) / (C - |A|)) outside, added times T to the fused scores of the 1-game questions.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_overlap3_v2o13c_9.json
"""
import argparse
import json
import multiprocessing as mp
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
from aicup_pid.assign import assign_soft  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import topk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_aug import fuse_v  # noqa: E402
from pid_probe_overlap import G, GRID, log_prior, pi_mle, pseudo_scores, softmax  # noqa: E402
from pid_probe_qq import Pool, boot, own_scores  # noqa: E402


def loglik(m, nA, C):
    return np.log(GRID[:, None] * m[None] / nA + (1 - GRID[:, None]) * (1 - m[None]) / (C - nA) + 1e-300).sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--planted", nargs="+", type=float, default=[0.0, 0.03, 0.06, 0.10, 0.167])
    ap.add_argument("--used", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.08])
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
    pooled = {k: [] for k in (2, 3, 5)}
    rngb = np.random.default_rng(9650)
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        C = P.C
        G.update(P=P)
        with mp.get_context("fork").Pool(len(a.seeds)) as ex:
            got = {s: (f1, f2) for s, f1, f2 in ex.map(pseudo_scores, [(s, a.ft_dir, pool) for s in a.seeds])}
        own = np.arange(C)
        Ts = np.exp(np.linspace(np.log(0.3), np.log(6.0), 61))
        nll = [float(np.mean([-np.log(softmax(got[s][0], T)[own, own] + 1e-300).mean() for s in a.seeds])) for T in Ts]
        T = float(Ts[int(np.argmin(nll))])
        res = {"T": T, "planted": {}}
        # ---- (c) planted structure in the pseudo-pools: a third of the 2-game questions stand in for the 3-game questions
        rng = np.random.default_rng(9660 + (0 if pool == "pub2" else 1))
        print(f"{pool}: temperature {T:.2f}; planted structure in the pseudo-pools ({a.draws} draws of 100 one-game questions per planted share) ({time.time() - t0:.0f} s)", flush=True)
        acc_x, est, gains = [], {w: [] for w in a.planted}, {w: {u: [] for u in a.used} for w in a.planted}
        for s in a.seeds:  # one assignment per draw; every planted share is read on it
            F1, F2 = got[s]
            Pq = softmax(F1, T)
            base_all, _h = own_scores(F1, own)
            for _ in range(a.draws // len(a.seeds)):
                M = rng.permutation(C)[:C // 2]
                asg = assign_soft(F2[M], 1.0, 5)[:, 0]
                X = M[:C // 6]  # the owners of the stand-in questions (M is in random order)
                inA = np.zeros(C, dtype=bool)
                inA[asg[:C // 6]] = True
                acc_x.append(float((asg[:C // 6] == X).mean()))
                lp = {u: T * log_prior(inA, u)[None] for u in a.used}
                rest = np.setdiff1d(own, X)
                for w_true in a.planted:
                    k = int(round(100 * w_true))
                    sel = np.concatenate([rng.choice(X, k, replace=False), rng.choice(rest, 100 - k, replace=False)])
                    est[w_true].append(pi_mle(Pq[sel][:, inA].sum(1), int(inA.sum()), C)[0])
                    for u in a.used:
                        s1, _h = own_scores(F1[sel] + lp[u], sel)
                        gains[w_true][u].append(float((s1 - base_all[sel]).mean()))
        res["stand_in_assigned_right"] = float(np.mean(acc_x))
        print(f"  stand-in questions assigned to their owner {np.mean(acc_x):.3f}", flush=True)
        for w_true in a.planted:
            r = {"estimate_mean": float(np.mean(est[w_true])), "estimate_sd": float(np.std(est[w_true])),
                 "gain_by_w_used": {f"{u:g}": boot(np.array(v), rngb) for u, v in gains[w_true].items()}}
            res["planted"][f"{w_true:g}"] = r
            print(f"  planted share {w_true:g}: estimate {r['estimate_mean']:.3f} (sd {r['estimate_sd']:.3f}) | gain per 1-game question, w used = "
                  + "  ".join(f"{u}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for u, v in r["gain_by_w_used"].items()), flush=True)
        # ---- (a) the real pool
        F = fuse_v(P, P.ref, P.qs, P.vec["t50"], P.row)
        n = np.array([len(q) for q in P.qs])
        big, one = np.flatnonzero(n > 1), np.flatnonzero(n == 1)
        a_big = assign_soft(F[big], 1.0, 5)[:, 0]
        Pq = softmax(F[one], T)
        top2 = np.sort(F[one], axis=1)[:, -2:]
        mg, t1 = top2[:, 1] - top2[:, 0], F[one].argmax(1)
        res["real"] = {"one_game_questions": int(len(one)), "groups": {}}
        t5 = topk(F[one], 5)
        for k in (2, 3, 5):
            Ak = np.unique(a_big[n[big] == k])
            inA = np.zeros(C, dtype=bool)
            inA[Ak] = True
            m = Pq[:, inA].sum(1)
            wh, lo, hi = pi_mle(m, len(Ak), C)
            pooled[k].append((m, len(Ak), C))
            g = {"candidates": int(len(Ak)), "independent_value": float(len(Ak) / C), "w_hat": wh, "w_95": [lo, hi],
                 "top1_in_group": {f"{t:g}": [int(inA[t1[mg > t]].sum()), int((mg > t).sum())] for t in (0.0, 1.0, 2.0, 3.0)}}
            if k == 3:
                g["effect_of_prior"] = {}
                for u in a.used:
                    n5 = topk(F[one] + T * log_prior(inA, u)[None], 5)
                    g["effect_of_prior"][f"{u:g}"] = {"top1_changed": int((n5[:, 0] != t5[:, 0]).sum()), "top5_changed": int((n5 != t5).any(1).sum()),
                                                      "top1_in_group_after": int(inA[n5[:, 0]].sum())}
            res["real"]["groups"][str(k)] = g
            print(f"{pool} REAL, owners of the {k}-game questions ({len(Ak)} candidates; independent draws give {len(Ak) / C:.3f}): w-hat {wh:.3f} [{lo:.3f}, {hi:.3f}]; "
                  f"1-game Top-1 in the group by margin > 0/1/2/3: " + "  ".join(f"{v[0]}/{v[1]}" for v in g["top1_in_group"].values())
                  + ("; prior on the real questions (Top-1 changed / Top-5 changed), w used = " + "  ".join(f"{u}: {v['top1_changed']}/{v['top5_changed']}" for u, v in g["effect_of_prior"].items()) if k == 3 else ""), flush=True)
        out["pools"][pool] = res
    out["both"] = {}
    for k in (2, 3, 5):
        ll = sum(loglik(m, nA, C) for m, nA, C in pooled[k])
        ok = GRID[ll >= ll.max() - 1.92]
        ind = float(np.mean([nA / C for _m, nA, C in pooled[k]]))
        out["both"][str(k)] = {"w_hat": float(GRID[ll.argmax()]), "w_95": [float(ok.min()), float(ok.max())], "independent_value": ind,
                               "loglik_above_independent": float(ll.max() - ll[int(np.argmin(np.abs(GRID - ind)))])}
        print(f"both pools REAL, owners of the {k}-game questions: w-hat {GRID[ll.argmax()]:.3f} [{ok.min():.3f}, {ok.max():.3f}] (independent {ind:.3f}; log-likelihood above it {out['both'][str(k)]['loglik_above_independent']:.2f})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
