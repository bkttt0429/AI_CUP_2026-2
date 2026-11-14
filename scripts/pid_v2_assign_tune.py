#!/usr/bin/env python
"""How hard should the one-to-one assignment of the multi-game questions be NOW (METHOD_NOTES 2.122)?

The added pools give every candidate at most one question among those of 2 / 3 / 5 games (METHOD_NOTES 2.37: compatible
with 0-5% repeated owners). The submission applies aicup_pid.assign.assign_soft with lam = 1 to those questions; lam was
chosen on 11-13 with the scores of that day (fused score about 0.78 before the assignment). The scores are much sharper now
(median Top-1 margin 3.7), so the same lam resolves fewer collisions. This script re-measures the assignment on OFFICIAL
players with the current scores:
  the official pseudo-pools of scripts/pid_v2_offsim.py (3 games of every candidate held out; questions of 2 and 3 games) under
  the fusion document's weights; per draw, 150 candidates give their 3-game question and 150 OTHER candidates their 2-game
  question (300 questions over 600 candidates, the density of the real pools); with contamination p a share p of the
  questions is replaced by a question of a candidate who already owns one (a repeated owner).
Methods: none; soft assignment with lam in --lams; the hard assignment; Sinkhorn prices (the score minus tau x a price per
candidate from the entropic one-to-one problem; changes the whole Top-5 order, not only the Top-1) with tau in --taus; and
the hard Top-1 followed by the Sinkhorn order.
Reported: the score (e^-(rank-1) over the Top-5) of every method by contamination, the difference from --ref (default
soft lam 1) with the spread over the (pool, seed) runs, and the number of Top-1 answers each method changes.
  python -I <this> --fusion results/pid_fusion_v2o13c_9.json --seeds 0 1 2 3 --out results/final/v2_assign_tune.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft, assign_topk  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402

DECAY = np.exp(-np.arange(5.0))


def score(idx, truth):
    """Mean e^-(rank-1) of the truth inside the Top-5 lists idx [Q, 5]."""
    hit = idx == truth[:, None]
    return float((hit * DECAY[None, :idx.shape[1]]).sum(1).mean())


def sinkhorn_prices(S, tau, iters=300):
    """log v [C] <= 0: the price of every candidate in the entropic assignment with row sums 1 and column sums <= 1."""
    K = np.exp((S - S.max(axis=1, keepdims=True)) / tau)
    v = np.ones(S.shape[1])
    for _ in range(iters):
        u = 1.0 / np.maximum(K @ v, 1e-300)
        v = np.minimum(1.0, 1.0 / np.maximum(K.T @ u, 1e-300))
    return np.log(np.maximum(v, 1e-300))


def methods(S, lams, taus):
    """{name: Top-5 indices [Q, 5]}."""
    out = {"none": topk(S, 5), "hard": assign_topk(S, 5)}
    for lam in lams:
        out[f"soft{lam:g}"] = assign_soft(S, lam, 5)
    for tau in taus:
        adj = S + tau * sinkhorn_prices(S, tau)[None, :]
        out[f"sink{tau:g}"] = topk(adj, 5)
        rows, cols = linear_sum_assignment(S, maximize=True)
        base = topk(adj, 5)
        mix = base.copy()
        for q, c in zip(rows, cols):
            if mix[q, 0] != c:
                mix[q] = np.array([c, *[x for x in base[q].tolist() if x != c]][:5])
        out[f"hard+sink{tau:g}"] = mix
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--fusion", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--per-size", type=int, default=150)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--contam", nargs="+", type=float, default=[0.0, 0.05, 0.15])
    ap.add_argument("--ref", default="soft1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T = Path(a.tests_dir)
    doc = json.loads(Path(a.fusion).read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq" and isinstance(doc["emb_model"], str)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    print(f"document {Path(a.fusion).name}: {len(nets)} sequence networks at t{a.step}; {a.draws} draws of {a.per_size} + {a.per_size} questions per run", flush=True)
    res, margins = {}, []  # res[(contam, method)] = list over runs of (mean score, mean changed Top-1)
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        C = len(players)
        ckeys = {k: c for g in cands.values() for k, c in g}
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)
        for s in a.seeds:
            hold = v2.candidate_holdout(cands, a.holdout, s)
            held = {k for g in hold.values() for k, _ in g}
            ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
            questions, owner, size = [], [], []
            for i, p in enumerate(players):
                h = [(k, c) for k, c in hold[p] if k not in bad]
                for m in (2, 3):
                    if len(h) >= m:
                        questions.append(h[:m]); owner.append(i); size.append(m)
            owner, size = np.array(owner), np.array(size)
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            # the sequence component as aicup_pid.v2.components computes it (mean of the networks' row-standardised cosine
            # matrices); a network repeated in the document (its weight) is read and scored once
            unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
            weight = {}
            for key in nets:
                weight[key] = weight.get(key, 0) + 1
            seq_acc, tot = 0.0, 0
            for (v, tg), w in weight.items():
                one = load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.step}.npz")
                Cm = unit(np.stack([np.sum([one[k] for k, _ in ref[p]], axis=0) for p in players]).astype(np.float64))
                Qm = unit(np.stack([np.sum([one[k] for k, _ in q], axis=0) for q in questions]).astype(np.float64))
                seq_acc = seq_acc + w * standardise(Qm @ Cm.T)
                tot += w
            S["seq"] = seq_acc / tot
            F = fus.scores([standardise(S[c]) for c in comps])
            row = {(int(o), int(m)): j for j, (o, m) in enumerate(zip(owner, size))}
            top2 = np.sort(F, axis=1)[:, -2:]
            margins.append(float(np.median(top2[:, 1] - top2[:, 0])))
            have = [i for i in range(C) if (i, 2) in row and (i, 3) in row]
            rng = np.random.default_rng(9100 + 17 * s + (0 if pool == "pub2" else 1))
            acc = {}
            for _ in range(a.draws):
                perm = rng.permutation(have)
                g3, g2 = perm[:a.per_size], perm[a.per_size:2 * a.per_size]
                for p in a.contam:
                    q = [(int(c), 3) for c in g3] + [(int(c), 2) for c in g2]
                    k = int(round(p * len(q)))
                    if k:  # k questions are replaced by the OTHER-size question of a candidate that already owns one
                        drop = set(rng.choice(len(q), k, replace=False).tolist())
                        keep = [x for j, x in enumerate(q) if j not in drop]
                        donors = rng.choice(len(keep), k, replace=False)
                        q = keep + [(keep[j][0], 5 - keep[j][1]) for j in donors]
                    rows = np.array([row[x] for x in q])
                    truth = np.array([x[0] for x in q])
                    Sq = F[rows]
                    base1 = Sq.argmax(1)
                    for name, idx in methods(Sq, a.lams, a.taus).items():
                        acc.setdefault((p, name), []).append((score(idx, truth), float((idx[:, 0] != base1).sum()), float((idx[:, 0] == truth).mean())))
            for key, vals in acc.items():
                res.setdefault(key, []).append(np.mean(np.array(vals), axis=0))
            print(f"{pool} split {s}: {len(have)} candidates with both questions; median Top-1 margin {margins[-1]:.2f}; "
                  + "  ".join(f"{n} {np.mean([x[0] for x in acc[(0.0, n)]]):.4f}" for n in ("none", a.ref, "hard")) + f" ({time.time() - t0:.0f} s)", flush=True)
    out = {"document": Path(a.fusion).name, "seeds": a.seeds, "draws": a.draws, "per_size": a.per_size, "ref": a.ref, "median_margin": margins, "table": {}}
    names = sorted({n for _p, n in res}, key=lambda n: (re.sub(r"[0-9.]+$", "", n), float(re.search(r"[0-9.]+$", n).group()) if re.search(r"[0-9.]+$", n) else 0.0))
    for p in a.contam:
        refv = np.array(res[(p, a.ref)])[:, 0]
        print(f"contamination {p:g}: score of the {2 * a.per_size} multi-game questions; difference from {a.ref} = mean over the {len(refv)} runs [min, max], runs up")
        for n in names:
            v = np.array(res[(p, n)])
            dlt = v[:, 0] - refv
            out["table"][f"{p:g}|{n}"] = {"score": float(v[:, 0].mean()), "top1": float(v[:, 2].mean()), "changed_top1": float(v[:, 1].mean()), "diff": float(dlt.mean()),
                                         "diff_min": float(dlt.min()), "diff_max": float(dlt.max()), "se": float(dlt.std(ddof=1) / np.sqrt(len(dlt))), "runs_up": int((dlt > 0).sum())}
            print(f"  {n:<12} {v[:, 0].mean():.4f} (Top-1 {v[:, 2].mean():.4f}, changes {v[:, 1].mean():5.1f} answers)  {dlt.mean():+.4f} [{dlt.min():+.4f}, {dlt.max():+.4f}]  se {dlt.std(ddof=1) / np.sqrt(len(dlt)):.4f}  up {int((dlt > 0).sum())}/{len(dlt)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
