#!/usr/bin/env python
"""Cohort score normalisation per candidate (METHOD_NOTES 2.153) - the registered test.

Every question is scored against 600 candidates and the fused scores are compared as they are. But candidates differ in
how strangers' games score on them: a "hub" collects high scores from everybody, a candidate with a generic style gets
modest scores even from his own games (notes 2.139: 28 % of the outcome variance is the player). Speaker verification
corrects this with a cohort: the scores that KNOWN impostors get on each enrolled speaker (Z-norm; adaptive: only the
most similar impostors). Here the cohort is free and inside the rules (candidate-pool games may be used for calibration
and normalisation): every REFERENCE game of the other candidates, scored as a 1-game question against candidate c, is an
impostor trial of c. Per candidate: mu_c, sd_c of the impostor scores (all of them, or the top N = his look-alikes).
The column statistics tried before (notes 2.139, measurement O) came from the questions themselves - few, and
contaminated by the true owners.
  variants  centre N g   : F_std - g * z(mu_c^N)             (z over the candidates; F_std = row-standardised fused)
            znorm N g    : (1 - g) * F_std + g * rowstd((F - mu_c^N) / sd_c^N)
Official pseudo-pools (3 of 10 games held out; 1- / 2- / 3-game questions; seeds 0-3, both pools): the variant is chosen
on seeds 0, 1 and the verdict is read on seeds 2, 3 (bootstrap over candidates). Then, with the chosen variant fixed, the
old-pool simulation (another population, 10-game reference sets, questions of 1 / 2 / 3 / 5 games).
  python -I <this> --doc v2bgm_d2 --out results/final/v2_probe_asnorm_v2bgm_d2.json
From the AI_CUP_2026-1 checkout, CPU only. Evaluation only: no answer of a real question is used.
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
from aicup_pid.fusion import Fusion, standardise  # noqa: E402

G = {}
_STATS = {}  # (id of a cohort matrix, N) -> (mu, sd); the matrices live as long as the run
NS = (None, 300, 100, 30, 10)
GAMMAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def score_of(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def per_question(F, own):
    return score_of((F > F[np.arange(len(F)), own][:, None]).sum(1))


def one_seed(task):
    from pid_probe_aug import fuse_v
    s, ft_dir, pool = task
    P = G["P"]
    C = P.C
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    qs = {1: [[g] for p in P.players for g in hold[p]], 2: [list(hold[p][:2]) for p in P.players], 3: [list(hold[p]) for p in P.players]}
    own = {1: np.repeat(np.arange(C), 3), 2: np.arange(C), 3: np.arange(C)}
    cohort = [[g] for p in P.players for g in ref[p]]
    cown = np.repeat(np.arange(C), [len(ref[p]) for p in P.players])
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    vecs = []
    for key in P.net_keys:
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d_ = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        vecs.append(np.stack([d_[k] for k in ckeys]))
    A = fuse_v(P, ref, qs[1] + qs[2] + qs[3] + cohort, vecs, row)  # every row is scored on its own: one call for all of them
    cut = np.cumsum([len(qs[1]), len(qs[2]), len(qs[3])])
    out = {"F": {1: A[:cut[0]], 2: A[cut[0]:cut[1]], 3: A[cut[1]:cut[2]]}, "own": own, "G": A[cut[2]:].astype(np.float32), "cown": cown}
    print(f"  {pool} seed {s}: 1-game score {per_question(out['F'][1], own[1]).mean():.4f}; cohort of {len(cohort)} reference games ({time.time() - G['t0']:.0f} s)", flush=True)
    return (pool, s), out


def old_side(a, doc, side, tag, emb, store):
    """The same quantities on the old-pool simulation of one side (as scripts/pid_pair_old.py)."""
    from aicup_pid.components import parse_all
    T = Path(a.tests_dir)
    comps = list(doc["components"])
    nets = [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in doc["seq_official"]]
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    cache, vecs = {}, []
    for v, tg in nets:
        f = Path(a.old_ft_dir) / f"oldsim_{side}_s0_net{v}{tg}_t50.npz"
        if f not in cache:
            cache[f] = load(f)
        vecs.append(cache[f])
    cands, questions, owner, sgfs = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(vecs[0]), 0)
    players = sorted(cands)
    truth = np.array([players.index(p) for p in owner])
    n = np.array([len(q) for q in questions])
    cohort = [[g] for p in players for g in cands[p]]
    cown = np.repeat(np.arange(len(players)), [len(cands[p]) for p in players])
    moves = parse_all(sgfs, a.workers)
    allq = questions + cohort
    S = v2.components(cands, allq, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
    S["seq"] = v2.components(cands, allq, moves, sgfs, doc["cfg"], which=("seq",), seq=vecs)["seq"]
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    A = fus.scores([standardise(np.nan_to_num(S[c], nan=0.0, posinf=0.0, neginf=0.0)) for c in comps])
    Q = len(questions)
    sizes = sorted(set(n.tolist()))
    return {"F": {m: A[:Q][n == m] for m in sizes}, "own": {m: truth[n == m] for m in sizes}, "G": A[Q:].astype(np.float32), "cown": cown}


def cohort_stats(Gm, cown, N):
    """Per candidate: mean and spread of the impostor scores (reference games of the OTHER candidates), all or the top N."""
    key = (id(Gm), N)
    if key not in _STATS:
        X = Gm.astype(np.float64).copy()
        X[np.arange(len(cown)), cown] = np.nan
        if N is None:
            _STATS[key] = (np.nanmean(X, 0), np.nanstd(X, 0))
        else:
            top = np.sort(np.where(np.isnan(X), -np.inf, X), axis=0)[-N:]
            _STATS[key] = (top.mean(0), top.std(0))
    return _STATS[key]


def apply(F, Gm, cown, variant):
    kind, N, g = variant
    Fs = standardise(F)
    if kind == "none":
        return Fs
    mu, sd = cohort_stats(Gm, cown, N)
    if kind == "centre":
        return Fs - g * (mu - mu.mean()) / max(mu.std(), 1e-9)
    return (1 - g) * Fs + g * standardise((F - mu[None]) / np.maximum(sd[None], 1e-6))


def diffs(tasks, keys, variant, sizes):
    d, n_, cl = [], [], []
    for key in keys:
        t = tasks[key]
        for m in sizes:
            F, own = t["F"][m], t["own"][m]
            d.append(per_question(apply(F, t["G"], t["cown"], variant), own) - per_question(F, own))
            n_.append(np.full(len(own), m))
            cl.append(np.char.add(str(key[0]) + "|", own.astype(str)))
    return np.concatenate(d), np.concatenate(n_), np.concatenate(cl)


def balanced(d, n, sizes):
    return float(np.mean([d[n == m].mean() for m in sizes]))


def boot(d, n, cluster, sizes, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in sizes])
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in sizes])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    per = num[:, pick].sum(2) / den[:, pick].sum(2)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"diff": balanced(d, n, sizes), "ci95": ci(per.mean(0)), "by_n": {str(m): float(d[n == m].mean()) for m in sizes}, "ci95_by_n": {str(m): ci(per[i]) for i, m in enumerate(sizes)}}


def name(v):
    return f"{v[0]} N={'all' if v[1] is None else v[1]} g={v[2]:g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", default="v2bgm_d2")
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--old-ft-dir", default="/work/u2507115/pid_v2/seq_ft_oldsim")
    ap.add_argument("--old-store", default=str(ROOT / "data" / "trunk" / "official"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--select-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    G["t0"] = t0
    import torch
    from aicup_pid.embed import PlayerEmbedding
    from aicup_pid.trunkstyle import load_store
    from pid_probe_qq import Pool
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    torch.set_num_threads(1)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    tasks = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        G.update(P=P)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            tasks.update(ex.map(one_seed, [(s, a.ft_dir, pool) for s in a.seeds]))
    rng = np.random.default_rng(0)
    sizes = (1, 2, 3)
    keys = sorted(tasks)
    sel, ver = [k for k in keys if k[1] in a.select_seeds], [k for k in keys if k[1] not in a.select_seeds]
    variants = [(kind, N, g) for kind in ("centre", "znorm") for N in NS for g in GAMMAS]
    out = {"doc": a.doc, "select_seeds": a.select_seeds, "grid": {}}
    # what the cohort says about the candidates
    t = tasks[keys[0]]
    mu_all, sd_all = cohort_stats(t["G"], t["cown"], None)
    mu_top, _ = cohort_stats(t["G"], t["cown"], 30)
    print(f"\ncohort of {keys[0]}: impostor mean per candidate (raw fused units) 5/50/95 %: {np.percentile(mu_all, [5, 50, 95]).round(2).tolist()}, spread {np.percentile(sd_all, [5, 50, 95]).round(2).tolist()}; "
          f"mean of the top 30 impostors: {np.percentile(mu_top, [5, 50, 95]).round(2).tolist()} (row spread of the fused scores {t['F'][1].std(1).mean():.2f})", flush=True)
    grid = {}
    for v in variants:
        d, n, _ = diffs(tasks, sel, v, sizes)
        grid[v] = (balanced(d, n, sizes), float(d[n == 1].mean()))
        out["grid"][name(v)] = {"select": grid[v][0], "select_n1": grid[v][1]}
    print(f"\nseeds {a.select_seeds} (selection), size-balanced gain / 1-game gain, by variant:")
    for kind in ("centre", "znorm"):
        for N in NS:
            print(f"  {kind:6s} N={'all' if N is None else N:<4} " + "  ".join(f"g={g:g}: {grid[(kind, N, g)][0]:+.4f}/{grid[(kind, N, g)][1]:+.4f}" for g in GAMMAS), flush=True)
    best = max(grid, key=lambda v: grid[v][0])
    res = boot(*diffs(tasks, ver, best, sizes), sizes, a.boot, rng)
    by_pool = {p: balanced(*diffs(tasks, [k for k in ver if k[0] == p], best, sizes)[:2], sizes) for p in sorted({k[0] for k in ver})}
    allv = boot(*diffs(tasks, keys, best, sizes), sizes, a.boot, rng)
    a_ok = res["diff"] >= 0.004 and res["ci95"][0] > 0
    b_ok = res["by_n"]["1"] > 0 and res["ci95_by_n"]["1"][0] > 0
    c_ok = all(x > 0 for x in by_pool.values())
    word = "PASS" if a_ok and b_ok and c_ok else "PARTIAL" if res["diff"] > 0 and res["ci95"][0] > 0 else "FAIL"
    out.update(chosen=name(best), verdict=res, verdict_by_pool=by_pool, all_seeds=allv, rule={"a": bool(a_ok), "b": bool(b_ok), "c": bool(c_ok), "verdict": word})
    print(f"\nchosen on seeds {a.select_seeds}: {name(best)} ({grid[best][0]:+.4f})")
    print(f"verdict on seeds {sorted({k[1] for k in ver})}: {res['diff']:+.4f} [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]; 1 game {res['by_n']['1']:+.4f} [{res['ci95_by_n']['1'][0]:+.4f}, {res['ci95_by_n']['1'][1]:+.4f}], "
          f"2 games {res['by_n']['2']:+.4f}, 3 games {res['by_n']['3']:+.4f}; pools {({p: round(x, 4) for p, x in by_pool.items()})}")
    print(f"all four seeds (the selection seeds included): {allv['diff']:+.4f} [{allv['ci95'][0]:+.4f}, {allv['ci95'][1]:+.4f}]")
    print(f"rule 2.153: (a) balanced gain >= +0.004 with the interval above 0: {a_ok}; (b) 1-game gain with the interval above 0: {b_ok}; (c) both pools above 0: {c_ok} -> {word}", flush=True)
    # the old-pool simulation at the chosen variant (and its neighbours, for information)
    old_doc = {"components": raw["components"], "weights": raw["weights"], "cfg": raw["cfg"], "seq_official": raw["seq_official"]}
    torch.set_num_threads(8)
    old_store = load_store(a.old_store)
    emb_old = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False))
    old = {(f"old{side}", 0): old_side(a, old_doc, side, tag, emb_old, old_store) for side, tag in (("pub", "public"), ("priv", "private"))}
    osz = (1, 2, 3, 5)
    okeys = sorted(old)
    base1 = np.mean([per_question(old[k]["F"][1], old[k]["own"][1]).mean() for k in okeys])
    ro = boot(*diffs(old, okeys, best, osz), osz, a.boot, rng)
    ro["by_side"] = {k[0]: balanced(*diffs(old, [k], best, osz)[:2], osz) for k in okeys}
    out["old_pool"] = {"chosen": ro, "one_game_score": float(base1), "grid": {}}
    print(f"\nold-pool simulation (1-game score {base1:.4f}) at the chosen variant: {ro['diff']:+.4f} [{ro['ci95'][0]:+.4f}, {ro['ci95'][1]:+.4f}]  by n " + " ".join(f"{m}:{ro['by_n'][str(m)]:+.4f}" for m in osz)
          + f"  sides {({p: round(x, 4) for p, x in ro['by_side'].items()})}")
    print("old-pool simulation, the whole grid (information only):")
    for kind in ("centre", "znorm"):
        for N in NS:
            row_ = []
            for g in GAMMAS:
                d, n, _ = diffs(old, okeys, (kind, N, g), osz)
                out["old_pool"]["grid"][name((kind, N, g))] = balanced(d, n, osz)
                row_.append(f"g={g:g}: {balanced(d, n, osz):+.4f}/{d[n == 1].mean():+.4f}")
            print(f"  {kind:6s} N={'all' if N is None else N:<4} " + "  ".join(row_), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
