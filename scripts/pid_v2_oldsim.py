#!/usr/bin/env python
"""Fusion documents on simulated pools cut from the ORIGINAL pools' candidates (METHOD_NOTES 2.55, item 7).

Why: my strongest networks have seen every training player, so the training-fold simulations cannot score them, and the
official pseudo-pools (3 of a candidate's 10 games held out) have 7-game candidate sets and no 5-game question. The original
pools hold 400 candidates x 100 games per side: official-population players that no network was trained on. A simulated pool
(aicup_pid.v2.original_sim): every candidate's 10-game set plus questions of 1 / 1 / 1 / 2 / 2 / 3 / 5 games from its other
games; the networks are fine-tuned on the candidate sets exactly as on a real pool (scripts/pid_v2_seq_finetune.py --mode
oldsim). Limits: 400 candidates instead of 600, Black games only. Evaluation only: no answer of any question is used and the
two sides are never mixed.

  --docs a b c ...     names of results/pid_fusion_<name>.json; the first is the reference, the others are compared with it
                       (paired bootstrap over candidates); the reference's base components and weights are used for all
  --fit-weights        also refit the fusion weights of the reference: fitted on one side's simulated pool, scored on the
                       other side's (and the weights fitted on each side, which is what that side's added pool would use)
Reported per side and for both: score by question size (1, 2, 3, 5) and their mean, differences from the reference.
"""
import argparse
import json
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
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402

SIZES = (1, 2, 3, 5)


def per_question(F, truth):
    rank = (F > F[np.arange(len(truth)), truth][:, None]).sum(1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def mix(x, n):
    return float(np.mean([x[n == m].mean() for m in SIZES]))


def boot(d, n, cluster, B, rng):
    """Size-balanced mean of d with a bootstrap over candidates."""
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    bs = (num[:, pick].sum(2) / den[:, pick].sum(2)).mean(0)
    return {"diff": float((num.sum(1) / den.sum(1)).mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "by_n": {str(m): float(d[n == m].mean()) for m in SIZES}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--step", type=int, default=50, help="fine-tune steps of the vectors used")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--fit-weights", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T, rng = Path(a.tests_dir), np.random.default_rng(0)
    docs = {n: json.loads((ROOT / "results" / f"pid_fusion_{n}.json").read_text()) for n in a.docs}
    ref = docs[a.docs[0]]
    comps = list(ref["components"])
    assert comps[-1] == "seq"
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / ref["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(a.store)
    nets_of = lambda d_: [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in d_["seq_official"]]  # noqa: E731
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    cache, runs = {}, []
    for side, tag in (("pub", "public"), ("priv", "private")):
        for s in a.seeds:
            first = load(Path(a.ft_dir) / f"oldsim_{side}_s{s}_net{nets_of(ref)[0][0]}{nets_of(ref)[0][1]}_t{a.step}.npz")
            assert all(k in store for k in first), "an original-pool game with per-ply vectors has no trunk row"
            cands, questions, owner, sgfs = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(first), s)
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            n = np.array([len(q) for q in questions])
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, ref["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            Z = {c: standardise(S[c]) for c in S}
            seq = {}
            for name, d_ in docs.items():
                vecs = []
                for v, tg in nets_of(d_):
                    f = Path(a.ft_dir) / f"oldsim_{side}_s{s}_net{v}{tg}_t{a.step}.npz"
                    if f not in cache:
                        cache[f] = load(f)
                    vecs.append(cache[f])
                seq[name] = standardise(v2.components(cands, questions, moves, sgfs, ref["cfg"], which=("seq",), seq=vecs)["seq"])
            cache.clear()
            runs.append({"side": side, "seed": s, "Z": Z, "seq": seq, "truth": truth, "n": n})
            print(f"{side} split {s}: {len(players)} candidates, {len(questions)} questions; sequence component alone: "
                  + "  ".join(f"{name} {mix(per_question(x, truth), n):.4f}" for name, x in seq.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    mats = lambda r, name: [r["seq"][name] if c == "seq" else r["Z"][c] for c in comps]  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(ref["weights"], dtype=float)
    sc = {(i, name): per_question(fus.scores(mats(r, name)), r["truth"]) for i, r in enumerate(runs) for name in docs}
    out = {"docs": a.docs, "step": a.step, "seeds": a.seeds, "scores": {}, "diff": {}, "weights": {}}
    groups = {"pub": [i for i, r in enumerate(runs) if r["side"] == "pub"], "priv": [i for i, r in enumerate(runs) if r["side"] == "priv"], "both": list(range(len(runs)))}
    for g, idx in groups.items():
        n = np.concatenate([runs[i]["n"] for i in idx])
        cl = np.concatenate([np.char.add(f"{runs[i]['side']}|", runs[i]["truth"].astype(str)) for i in idx])
        out["scores"][g] = {}
        for name in docs:
            x = np.concatenate([sc[(i, name)] for i in idx])
            out["scores"][g][name] = {"mix": mix(x, n), **{str(m): float(x[n == m].mean()) for m in SIZES}}
        print(f"{g}: " + "  ".join(f"{name} {v_['mix']:.4f} (n=1 {v_['1']:.4f}, n=2 {v_['2']:.4f}, n=3 {v_['3']:.4f}, n=5 {v_['5']:.4f})" for name, v_ in out["scores"][g].items()), flush=True)
        for name in a.docs[1:]:
            d = np.concatenate([sc[(i, name)] - sc[(i, a.docs[0])] for i in idx])
            b = boot(d, n, cl, a.boot, rng)
            out["diff"][f"{g}: {name} vs {a.docs[0]}"] = b
            print(f"   {name} vs {a.docs[0]}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)
    if a.fit_weights:
        name = a.docs[0]
        fits = {}
        for side in ("pub", "priv"):
            rs = [runs[i] for i in groups[side]]
            f2 = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, name) for r in rs])], np.concatenate([r["truth"] for r in rs]))
            fits[side] = f2
            out["weights"][side] = dict(zip(comps, f2.w.tolist()))
            print(f"weights fitted on the {side} simulated pool: {({c: round(w, 3) for c, w in zip(comps, f2.w)})}", flush=True)
        print(f"document weights: {({c: round(w, 3) for c, w in zip(comps, fus.w)})}", flush=True)
        d_all, n_all, cl_all = [], [], []
        for side, other in (("pub", "priv"), ("priv", "pub")):  # scored on the side the weights were NOT fitted on
            for i in groups[side]:
                r = runs[i]
                d = per_question(fits[other].scores(mats(r, name)), r["truth"]) - sc[(i, name)]
                d_all.append(d); n_all.append(r["n"]); cl_all.append(np.char.add(f"{side}|", r["truth"].astype(str)))
                b = boot(d, r["n"], cl_all[-1], a.boot, rng)
                print(f"   weights of {other} scored on {side} (split {r['seed']}), against the document weights: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)
        b = boot(np.concatenate(d_all), np.concatenate(n_all), np.concatenate(cl_all), a.boot, rng)
        out["weights"]["cross_side_vs_document"] = b
        print(f"refitted weights (fitted on the other side) against the document weights, both sides: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
