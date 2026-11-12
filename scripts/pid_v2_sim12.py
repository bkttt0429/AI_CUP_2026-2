#!/usr/bin/env python
"""Owner structure of the v2 added pools (METHOD_NOTES 2.37): which structure the official pools' label-free statistics
match, and what each way of using it does under each structure. Multi-network pools as pid_v2_sim11.py.

Structures, cut from one simulated pool's two blocks (D = 400 questions of distinct owners, R = 400 drawn with replacement):
  distinct   block D: no player owns two questions
  replace    block R: every question's owner drawn independently
  bysize     distinct inside a size group, independent across size groups
  one_free   sizes 2 / 3 / 5 of block D (distinct), size 1 of block R (independent of the others)
Procedures on the fused scores of the questions of one structure:
  none       per-question Top-5
  global     one-to-one assignment over all questions (aicup_pid.assign.assign_topk)
  by_size    one-to-one inside every size group separately
  big3       one-to-one over the questions of sizes 2 / 3 / 5 together; size 1 per question
  big3_1     big3, and one-to-one inside size 1
  big3_s<L>  as big3 with the soft assignment (aicup_pid.assign.assign_soft): a second question on one candidate costs L of
             fused score (L = 1, 2, 3, 5), so two questions both far ahead on one candidate keep it
  mix<p>     (structure) one_free with p % of its size 2 / 3 / 5 questions swapped for block-R questions of the same size:
             how much repetition among the larger sizes each procedure tolerates
Models: `multi` (all networks, --steps of the pool fine-tune, emb2: the current official configuration) and `v2a` (the base
components only: the model of the 11-12 leaderboard check, forced assignment -0.0114 on the added Public pool).
Label-free statistics per structure (model multi): pairs of questions with the same Top-1 by size cell, distinct Top-1, the
confident-half share; --official <submission csv> ... gives the same cells of the official pools and their Poisson
log-likelihood under every structure's mean cells.
"""
import argparse
import json
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_blocks, assign_soft, assign_topk  # noqa: E402
from aicup_pid.checks import confident_distinct  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
CELLS = list(combinations_with_replacement(SIZES, 2))
PER_SIZE = 100


def cells(t1, n):
    """{(a, b): pairs of questions of sizes a and b that name the same Top-1 candidate}."""
    out = {}
    for a, b in CELLS:
        ca, cb = pd.Series(t1[n == a]).value_counts(), pd.Series(t1[n == b]).value_counts()
        out[(a, b)] = int((ca * (ca - 1) // 2).sum()) if a == b else int((ca.reindex(cb.index).fillna(0) * cb).sum())
    return out


BIG = [(x, y) for x, y in CELLS if x != 1]  # the cells among sizes 2 / 3 / 5
SOFT = (1.0, 2.0, 3.0, 5.0)
MIX = (5, 15, 30)


def structures(r, seed=0):
    n, owner, block = r["n"], r["owner"], r["block"]
    D, R = np.flatnonzero(block == "D"), np.flatnonzero(block == "R")
    pick = []
    for m in SIZES:  # distinct inside the size group: the R questions of the size without repeated owners, topped up from D
        seen, grp = set(), []
        for i in (*R[n[R] == m], *D[n[D] == m]):
            if owner[i] not in seen and len(grp) < PER_SIZE:
                seen.add(owner[i])
                grp.append(int(i))
        pick += grp
    out = {"distinct": D, "replace": R, "bysize": np.array(sorted(pick)), "one_free": np.array(sorted([*D[n[D] != 1].tolist(), *R[n[R] == 1].tolist()]))}
    rng = np.random.RandomState(1000 + seed)
    big_d, big_r, one = D[n[D] != 1], R[n[R] != 1], R[n[R] == 1].tolist()
    for pct in MIX:
        drop = rng.choice(len(big_d), int(round(pct / 100 * len(big_d))), replace=False)
        add = [int(i) for m in SIZES if m != 1 for i in rng.choice(big_r[n[big_r] == m], int((n[big_d[drop]] == m).sum()), replace=False)]
        out[f"mix{pct:02d}"] = np.array(sorted([*np.delete(big_d, drop).tolist(), *add, *one]))
    return out


def procedures(F, n):
    big = np.where(n == 1, "one", "big")
    out = {"none": topk(F, 5), "global": assign_topk(F, 5), "by_size": assign_blocks(F, n, {int(m): True for m in SIZES}, 5),
           "big3": assign_blocks(F, big, {"big": True, "one": False}, 5), "big3_1": assign_blocks(F, big, {"big": True, "one": True}, 5)}
    rows = np.flatnonzero(n != 1)
    for lam in SOFT:
        out[f"big3_s{lam:g}"] = out["none"].copy()
        out[f"big3_s{lam:g}"][rows] = assign_soft(F[rows], lam, 5)
    return out


def official_cells(csv, tests_dir):
    sub = pd.read_csv(csv, dtype=str).set_index("question_id")
    out = {}
    for tag, prefix in (("public", "pub2"), ("private", "priv2")):
        q = pd.read_csv(Path(tests_dir) / f"player_identification_test2_{tag}.csv", dtype=str, usecols=["question_id", "num_games"])
        t1, n = sub.loc[q.question_id, "top_1"].values, q.num_games.astype(int).values
        out[prefix] = {"cells": cells(t1, n), "distinct": int(len(set(t1.tolist())))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--official", nargs="*", default=[])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731

    def ft_file(v_, s, name):
        if a.steps == 0:
            return ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz"
        if name == "m88":
            return Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if a.steps == 50 else f"sim_f{v_}_s{s}_t100.npz")
        return Path(a.ft_dir) / f"sim_f{v_}_s{s}{nets[name][1]}_t{a.steps}.npz"

    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            seq = standardise(np.mean([standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(ft_file(v_, s, name)))["seq"]) for name in nets], axis=0))
            players = sorted(cands)
            Z = {c: standardise(x) for c, x in S.items()}
            runs.append({"multi": [seq if c == "seq" else Z[c] for c in COMPS], "v2a": [Z[c] for c in BASE], "truth": np.array([players.index(p) for p in owner]),
                         "n": np.array([len(q) for q in questions]), "owner": owner, "block": block, "fold": v_})
            print(f"pool fold {v_} seed {s}: {len(questions)} questions ({time.time() - t0:.0f} s)", flush=True)

    out = {"nets": {k: list(x) for k, x in nets.items()}, "steps": a.steps, "models": {}, "label_free": {}}
    stats = {}
    for model in ("multi", "v2a"):
        res = {}  # (structure, procedure) -> per-question scores per pool
        for ri, r in enumerate(runs):
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[o[model] for o in others])], np.concatenate([o["truth"] for o in others]))
            F = fus.scores(r[model])
            for st, idx in structures(r, ri).items():
                n, truth = r["n"][idx], r["truth"][idx]
                for pr, top in procedures(F[idx], n).items():
                    res.setdefault((st, pr), []).append((score(top, truth), n, r["owner"][idx]))
                if model == "multi":
                    t1 = F[idx].argmax(axis=1)
                    stats.setdefault(st, []).append({"cells": cells(t1, n), "distinct": len(set(t1.tolist())) / len(idx) * 400, "confident": confident_distinct(F[idx]), "questions": len(idx)})
        out["models"][model] = {}
        for st in ("distinct", "replace", "bysize", "one_free", *[f"mix{p:02d}" for p in MIX]):
            base = res[(st, "none")]
            pid = np.concatenate([np.char.add(o.astype(str), f"|{i}") for i, (_, _, o) in enumerate(base)])
            nn = np.concatenate([x[1] for x in base])
            b = np.concatenate([x[0] for x in base])
            row = {"none": {"score": float(b.mean()), "by_n": {str(m): float(b[nn == m].mean()) for m in SIZES}}}
            print(f"[{model}] {st}: none {b.mean():.4f}", flush=True)
            for pr in ("global", "by_size", "big3", "big3_1", *[f"big3_s{lam:g}" for lam in SOFT]):
                x = np.concatenate([y[0] for y in res[(st, pr)]])
                d = paired_player_bootstrap(x, b, pid, n=2000)
                d["pools_positive"] = int(sum(y[0].mean() > z[0].mean() for y, z in zip(res[(st, pr)], base)))
                d["per_pool"] = [float(y[0].mean() - z[0].mean()) for y, z in zip(res[(st, pr)], base)]
                d["by_n"] = {str(m): float(x[nn == m].mean() - b[nn == m].mean()) for m in SIZES}
                row[pr] = d
                print(f"   {pr:8s} {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(base)}, per pool {min(d['per_pool']):+.4f}..{max(d['per_pool']):+.4f}, "
                      f"by n {({k: round(v, 4) for k, v in d['by_n'].items()})}", flush=True)
            out["models"][model][st] = row
    for st, L in stats.items():
        mean = {f"{x}x{y}": float(np.mean([s_["cells"][(x, y)] for s_ in L])) for x, y in CELLS}
        out["label_free"][st] = {"cells_mean": mean, "cells_per_pool": [{f"{x}x{y}": s_["cells"][(x, y)] for x, y in CELLS} for s_ in L], "distinct_per_400": [s_["distinct"] for s_ in L],
                                 "confident": [s_["confident"] for s_ in L], "questions": [s_["questions"] for s_ in L]}
        bigs = [sum(s_["cells"][c] for c in BIG) for s_ in L]
        out["label_free"][st]["big_cells_sum"] = bigs
        print(f"label-free {st}: pairs among sizes 2/3/5 {np.mean(bigs):.1f} ({min(bigs)}-{max(bigs)}); cells {({k: round(v, 1) for k, v in mean.items()})}; distinct Top-1 per 400 {np.mean([s_['distinct'] for s_ in L]):.0f} "
              f"({min(s_['distinct'] for s_ in L):.0f}-{max(s_['distinct'] for s_ in L):.0f}); confident half {np.mean([s_['confident'] for s_ in L]):.3f} "
              f"({min(s_['confident'] for s_ in L):.3f}-{max(s_['confident'] for s_ in L):.3f})", flush=True)
    out["official"] = {}
    for csv in a.official:
        off = official_cells(csv, a.tests_dir)
        for pool, o in off.items():
            ll = {}
            for st in out["label_free"]:
                mu = out["label_free"][st]["cells_mean"]
                ll[st] = float(sum(o["cells"][(x, y)] * np.log(max(mu[f"{x}x{y}"], 0.25)) - max(mu[f"{x}x{y}"], 0.25) for x, y in CELLS))
            best = max(ll.values())
            out["official"][f"{Path(csv).name}:{pool}"] = {"cells": {f"{x}x{y}": o["cells"][(x, y)] for x, y in CELLS}, "distinct": o["distinct"], "loglik_minus_best": {k: v - best for k, v in ll.items()}}
            out["official"][f"{Path(csv).name}:{pool}"]["big_cells_sum"] = int(sum(o["cells"][c] for c in BIG))
            print(f"official {Path(csv).name} {pool}: pairs among sizes 2/3/5 {sum(o['cells'][c] for c in BIG)}; cells {({f'{x}x{y}': o['cells'][(x, y)] for x, y in CELLS})}; distinct {o['distinct']}; "
                  f"Poisson log-likelihood minus best {({k: round(v - best, 1) for k, v in ll.items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
