#!/usr/bin/env python
"""Probe S3 on the OLD-POOL simulation (METHOD_NOTES 2.123, part b): the prior "the owner of a 1-game question does not own a
3-game question" with multi-game questions of the real sizes and true answers.

Pools of scripts/pid_v2_oldsim.py (400 official-population candidates per side, 10-game candidate sets, questions of
1 / 1 / 1 / 2 / 2 / 3 / 5 games). Per draw a sixth of the candidates each own a 5-, a 3- and a 2-game question (distinct
owners, assigned as the pipeline assigns them: soft, lam 1); a sixth of the candidates own a 1-game question, a share w_true
of them taken from the owners of the 3-game questions and the others from everybody else. Prior on the 1-game questions:
log(w / |A3|) for the candidates assigned to the 3-game questions, log((1 - w) / (C - |A3|)) for the others, times T.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_overlap3_old_v2o13c_9.json
"""
import argparse
import json
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
from pid_probe_aug import fuse_v  # noqa: E402
from pid_probe_overlap import log_prior, pi_mle, softmax  # noqa: E402
from pid_probe_qq import boot, own_scores  # noqa: E402

SIZES = (5, 3, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--planted", nargs="+", type=float, default=[0.0, 0.03, 0.06, 0.10, 0.167])
    ap.add_argument("--used", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.08])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    torch.set_num_threads(1)
    T_ = Path(a.tests_dir)
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    comps = list(raw["components"])
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(a.store)
    weight = {}
    for x in raw["seq_official"]:
        key = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0] if isinstance(x, list) else x).name).groups()
        weight[key] = weight.get(key, 0) + 1
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    rngb = np.random.default_rng(9670)
    out = {"doc": a.doc, "draws": a.draws, "sides": {}}
    for side, tag in (("pub", "public"), ("priv", "private")):
        nets = list(weight)
        first = load(Path(a.ft_dir) / f"oldsim_{side}_s{a.seed}_net{nets[0][0]}{nets[0][1]}_t{a.step}.npz")
        cands, questions, owner, sgfs = v2.original_sim(T_ / f"player_identification_candidates_{tag}.csv", set(first), a.seed)
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
            qrow.setdefault((int(o), int(m)), j)
        C = len(players)
        assert all((i, m) in qrow for i in range(C) for m in (1, *SIZES)), "a candidate lacks a question size"
        F1 = fuse_v(P, cands, [questions[qrow[(i, 1)]] for i in range(C)], vecs, row)
        Fm = {m: fuse_v(P, cands, [questions[qrow[(i, m)]] for i in range(C)], vecs, row) for m in SIZES}
        own = np.arange(C)
        Ts = np.exp(np.linspace(np.log(0.3), np.log(6.0), 61))
        nll = [float(-np.log(softmax(F1, T)[own, own] + 1e-300).mean()) for T in Ts]
        T = float(Ts[int(np.argmin(nll))])
        Pq = softmax(F1, T)
        base_all, hit = own_scores(F1, own)
        n6 = C // 6
        n2 = C // 2 - 2 * n6
        res = {"candidates": C, "T": T, "one_game_score": float(base_all.mean()), "planted": {}}
        print(f"{side}: {C} candidates, 1-game score {base_all.mean():.4f} (Top-1 {hit.mean():.3f}), temperature {T:.2f}; per draw {n6} five-, {n6} three-, {n2} two-game and {n6} one-game questions ({time.time() - t0:.0f} s)", flush=True)
        rng = np.random.default_rng(9680 + (0 if side == "pub" else 1))
        acc3, est, gains = [], {w: [] for w in a.planted}, {w: {u: [] for u in a.used} for w in a.planted}
        for _ in range(a.draws):  # one assignment per draw; every planted share is read on it
            perm = rng.permutation(C)
            o5, o3, o2 = perm[:n6], perm[n6:2 * n6], perm[2 * n6:2 * n6 + n2]
            asg = assign_soft(np.concatenate([Fm[5][o5], Fm[3][o3], Fm[2][o2]]), 1.0, 5)[:, 0]
            a3 = asg[n6:2 * n6]
            inA = np.zeros(C, dtype=bool)
            inA[a3] = True
            acc3.append(float((a3 == o3).mean()))
            lp = {u: T * log_prior(inA, u)[None] for u in a.used}
            for w_true in a.planted:
                k = int(round(n6 * w_true))
                sel = np.concatenate([rng.choice(o3, k, replace=False), rng.choice(np.setdiff1d(own, o3), n6 - k, replace=False)])
                est[w_true].append(pi_mle(Pq[sel][:, inA].sum(1), int(inA.sum()), C)[0])
                for u in a.used:
                    s1, _h = own_scores(F1[sel] + lp[u], sel)
                    gains[w_true][u].append(float((s1 - base_all[sel]).mean()))
        res["three_game_assigned_right"] = float(np.mean(acc3))
        print(f"  3-game questions assigned to their owner {np.mean(acc3):.3f}", flush=True)
        for w_true in a.planted:
            r = {"estimate_mean": float(np.mean(est[w_true])), "estimate_sd": float(np.std(est[w_true])),
                 "gain_by_w_used": {f"{u:g}": boot(np.array(v), rngb) for u, v in gains[w_true].items()}}
            res["planted"][f"{w_true:g}"] = r
            print(f"  planted share {w_true:g}: estimate {r['estimate_mean']:.3f} (sd {r['estimate_sd']:.3f}) | gain per 1-game question, w used = "
                  + "  ".join(f"{u}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for u, v in r["gain_by_w_used"].items()), flush=True)
        out["sides"][side] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
