#!/usr/bin/env python
"""Do a linked candidate's TRAINING games help to recognise his question games? (METHOD_NOTES 2.167) - registered test.

The linkage probe (notes 2.166) found for 9-13 % of the added pools' candidates a training player at same-person level
(cosine of the 10-game prototype to the training prototype >= 0.88 / 0.85; two-game queries as high as a known player's
held-out games). If that is the same person, the candidate has dozens of further reference games - in the training set.
Test on the official pseudo-pools (3 of the 10 reference games held out as questions of 1 / 2 / 3 games; four seeds, both
pools; fused scores of one fusion document):
  - the link is decided from the SEVEN remaining reference games only (never from a question game): best training
    prototype under the probe networks (o22Ss0-3, the vectors before any pool fine-tune), linked when cosine >= tau
  - a sequence component from the same four networks is computed twice: with every candidate's own prototype (plain),
    and with  unit(own prototype + lam * training twin's prototype)  for the linked candidates (augmented)
  - new score = row-standardised fused score + w * (augmented - plain): only the CHANGE the twin's games make enters
(tau, lam, w) are chosen on hold-out seeds 0, 1; the verdict is read on seeds 2, 3 (bootstrap over candidates). Printed
with it: the share of questions whose owner is linked, their score against the others', and the change on each part
(the unlinked questions show whether an enlarged prototype steals answers, the failure of the lending tests).
From the AI_CUP_2026-1 checkout, CPU only. Evaluation only: no answer of a real question is used; the sides are not mixed.
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
from aicup_pid.fusion import standardise  # noqa: E402

G = {}
SIZES = (1, 2, 3)
TAUS = (0.85, 0.88, 0.9)
LAMS = (0.5, 1.0, 2.0, 4.0)
WS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def score_of(rank):
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def per_question(F, own):
    return score_of((F > F[np.arange(len(F)), own][:, None]).sum(1))


def one_seed(task):
    from pid_probe_aug import fuse_v
    s, ft_dir, pool = task
    P, U, vec0 = G["P"], G["U"], G["vec0"]
    C, K = P.C, len(U)
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    qs = {1: [[g] for p in P.players for g in hold[p]], 2: [list(hold[p][:2]) for p in P.players], 3: [list(hold[p]) for p in P.players]}
    own = {1: np.repeat(np.arange(C), 3), 2: np.arange(C), 3: np.arange(C)}
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    vecs = []
    for key in P.net_keys:
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d_ = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        vecs.append(np.stack([d_[k] for k in ckeys]))
    allq = qs[1] + qs[2] + qs[3]
    cohort = [[g] for p in P.players for g in ref[p]] if G["znorm"] else []  # the reference games as 1-game trials (notes 2.153)
    Araw = fuse_v(P, ref, allq + cohort, vecs, row)
    cut = np.cumsum([len(qs[1]), len(qs[2]), len(qs[3])])
    A = standardise(Araw[:cut[2]])
    Fz = {1: A[:cut[0]], 2: A[cut[0]:cut[1]], 3: A[cut[1]:cut[2]]}
    Zn = None
    if G["znorm"]:  # cohort normalisation with the setting chosen in notes 2.154: the top N impostor scores per candidate
        N_, g_ = G["znorm"]
        Gc = Araw[cut[2]:].astype(np.float64)
        cown = np.repeat(np.arange(C), [len(ref[p]) for p in P.players])
        Gc[np.arange(len(cown)), cown] = -np.inf
        top = np.sort(Gc, axis=0)[-int(N_):]
        mu_c, sd_c = top.mean(0), np.maximum(top.std(0), 1e-6)
        Zr = (1 - g_) * A + g_ * standardise((Araw[:cut[2]] - mu_c[None]) / sd_c[None])
        Zn = {1: Zr[:cut[0]].astype(np.float32), 2: Zr[cut[0]:cut[1]].astype(np.float32), 3: Zr[cut[1]:cut[2]].astype(np.float32)}
    # the probe networks: prototypes of the seven reference games, the link, the two components
    proto = [unit(np.stack([np.sum([vec0[k][g] for g, _ in ref[p]], 0) for p in P.players])) for k in range(K)]
    St = np.mean([proto[k] @ U[k].T for k in range(K)], 0)
    tb, tc = St.argmax(1), St.max(1)
    Q = {n: [unit(np.stack([np.sum([vec0[k][g] for g, _ in q], 0) for q in qs[n]])) for k in range(K)] for n in SIZES}
    plain = {n: np.mean([standardise(Q[n][k] @ proto[k].T) for k in range(K)], 0) for n in SIZES}
    D, alone = {}, {(0.0, 0.0, n): per_question(plain[n], own[n]).astype(np.float32) for n in SIZES}
    # --cal (notes 2.168): an enlarged prototype scores higher on EVERY game, strangers' included (the unlinked questions
    # lost 0.003-0.005 without it). The impostor trials of a candidate are the reference games of the other candidates
    # (never a question game): the enlarged prototype's scores are mapped so that its impostor mean and spread are those
    # of the candidate's own prototype - the stranger sees the same candidate as before, the owner the better one.
    rown = np.repeat(np.arange(C), [len(ref[p]) for p in P.players])
    R = [unit(np.stack([vec0[k][g] for p in P.players for g, _ in ref[p]])) for k in range(K)] if G["cal"] else None

    def imp_stats(k, protos):
        Gm = R[k] @ protos.T
        Gm[np.arange(len(rown)), rown] = np.nan
        return np.nanmean(Gm, 0), np.nanstd(Gm, 0) + 1e-9

    base = [imp_stats(k, proto[k]) for k in range(K)] if G["cal"] else None
    for tau in TAUS:
        linked = tc >= tau
        for lam in LAMS:
            comps = {n: [] for n in SIZES}
            for k in range(K):
                p_ = proto[k].copy()
                p_[linked] = unit(proto[k][linked] + lam * U[k][tb[linked]])
                if G["cal"]:
                    mu_a, sd_a = imp_stats(k, p_)
                for n in SIZES:
                    S = Q[n][k] @ p_.T
                    if G["cal"]:
                        S = (S - mu_a) / sd_a * base[k][1] + base[k][0]  # the identity for an unlinked candidate
                    comps[n].append(standardise(S))
            for n in SIZES:
                comp = np.mean(comps[n], 0)
                D[(tau, lam, n)] = (comp - plain[n]).astype(np.float32)
                alone[(tau, lam, n)] = per_question(comp, own[n]).astype(np.float32)  # the four-network component on its own, with the twins
    r1 = per_question(Fz[1], own[1])
    print(f"  {pool} seed {s}: 1-game score {r1.mean():.4f}; candidates linked at {TAUS}: {[int((tc >= t).sum()) for t in TAUS]} of {C} ({time.time() - G['t0']:.0f} s)", flush=True)
    return (pool, s), {"Fz": {n: Fz[n].astype(np.float32) for n in SIZES}, "own": own, "D": D, "tc": tc, "tb": tb, "alone": alone, "Zn": Zn}


def diffs(tasks, keys, tau, lam, w, part=None, zn=None):
    zn = (G.get("znorm") is not None) if zn is None else zn
    d, n_, cl = [], [], []
    for key in keys:
        t = tasks[key]
        linked = t["tc"] >= tau
        for n in SIZES:
            F, own = t["Fz"][n], t["own"][n]
            start = t["Zn"][n] if zn else F  # with --znorm the add-on sits on the cohort-normalised score; the gain is always against the plain fused score
            x = per_question(start + w * t["D"][(tau, lam, n)], own) - per_question(F, own)
            m = np.ones(len(own), bool) if part is None else (linked[own] if part == "linked" else ~linked[own])
            d.append(x[m]); n_.append(np.full(int(m.sum()), n)); cl.append(np.char.add(str(key[0]) + "|", own[m].astype(str)))
    return np.concatenate(d), np.concatenate(n_), np.concatenate(cl)


def balanced(d, n):
    return float(np.mean([d[n == m].mean() for m in SIZES if (n == m).any()]))


def boot(d, n, cluster, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    per = num[:, pick].sum(2) / np.maximum(den[:, pick].sum(2), 1e-9)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"diff": balanced(d, n), "ci95": ci(per.mean(0)), "by_n": {str(m): float(d[n == m].mean()) for m in SIZES}, "ci95_by_n": {str(m): ci(per[i]) for i, m in enumerate(SIZES)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", default="v2bgm_d2")
    ap.add_argument("--protos", default="/work/u2507115/pid_v2/pair_eval/link_protos.npz")
    ap.add_argument("--vec-dir", default=str(ROOT / "data" / "seq_ft"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--select-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-gain", type=float, default=0.004)
    ap.add_argument("--znorm", nargs=2, type=float, default=None, metavar=("N", "G"), help="also apply the cohort normalisation of notes 2.153 (top N impostors, mixing weight g) under the add-on")
    ap.add_argument("--cal", action="store_true", help="map an enlarged prototype's scores to the impostor mean and spread of the candidate's own prototype")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    G["t0"] = t0
    G["cal"] = a.cal
    G["znorm"] = tuple(a.znorm) if a.znorm else None
    import torch
    from aicup_pid.embed import PlayerEmbedding
    from aicup_pid.trunkstyle import load_store
    from pid_probe_qq import Pool
    Z = np.load(a.protos)
    G["U"] = [u.astype(np.float64) for u in Z["U"]]
    alias = Z["alias"].astype(str).tolist()
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
        vec0 = []
        for al in alias:
            z = np.load(Path(a.vec_dir) / f"official_{pool}_net-1_{al}e_t0.npz")
            vec0.append(dict(zip(z["key"].astype(str).tolist(), z["emb"].astype(np.float64))))
        G.update(P=P, vec0=vec0)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            tasks.update(ex.map(one_seed, [(s, a.ft_dir, pool) for s in a.seeds]))
    rng = np.random.default_rng(0)
    keys = sorted(tasks)
    print("\nthe four-network component ON ITS OWN (no fusion), all four seeds: score of the questions whose owner is linked / not linked, without and with the twin's prototype")
    for tau in TAUS:
        for lam in LAMS:
            cells = []
            for n in SIZES:
                lk = np.concatenate([(tasks[k]["tc"] >= tau)[tasks[k]["own"][n]] for k in keys])
                p0 = np.concatenate([tasks[k]["alone"][(0.0, 0.0, n)] for k in keys]); p1 = np.concatenate([tasks[k]["alone"][(tau, lam, n)] for k in keys])
                cells.append(f"{n} game(s): linked {p0[lk].mean():.3f} -> {p1[lk].mean():.3f} ({p1[lk].mean() - p0[lk].mean():+.3f}), others {p0[~lk].mean():.3f} -> {p1[~lk].mean():.3f} ({p1[~lk].mean() - p0[~lk].mean():+.4f}), all {p1.mean() - p0.mean():+.4f}")
            print(f"  tau {tau} lam {lam:g}: " + " | ".join(cells), flush=True)
    sel, ver = [k for k in keys if k[1] in a.select_seeds], [k for k in keys if k[1] not in a.select_seeds]
    grid = {}
    print(f"\nseeds {a.select_seeds} (selection): size-balanced gain / 1-game gain")
    for tau in TAUS:
        for lam in LAMS:
            row_ = []
            for w in WS:
                d, n, _ = diffs(tasks, sel, tau, lam, w)
                grid[(tau, lam, w)] = (balanced(d, n), float(d[n == 1].mean()))
                row_.append(f"w={w:g}: {grid[(tau, lam, w)][0]:+.4f}/{grid[(tau, lam, w)][1]:+.4f}")
            print(f"  tau {tau} lam {lam:g}  " + "  ".join(row_), flush=True)
    best = max(grid, key=lambda v: grid[v][0])
    tau, lam, w = best
    res = boot(*diffs(tasks, ver, tau, lam, w), a.boot, rng)
    by_pool = {p: balanced(*diffs(tasks, [k for k in ver if k[0] == p], tau, lam, w)[:2]) for p in sorted({k[0] for k in ver})}
    allv = boot(*diffs(tasks, keys, tau, lam, w), a.boot, rng)
    parts = {}
    for part in ("linked", "other"):
        d, n, _ = diffs(tasks, ver, tau, lam, w, part)
        base = np.concatenate([per_question(tasks[k]["Fz"][m], tasks[k]["own"][m])[((tasks[k]["tc"] >= tau)[tasks[k]["own"][m]]) == (part == "linked")] for k in ver for m in SIZES])
        parts[part] = {"questions": int(len(d)), "base_score": float(base.mean()) if len(base) else 0.0, "gain": balanced(d, n) if len(d) else 0.0, "gain_by_n": {str(m): float(d[n == m].mean()) if (n == m).any() else 0.0 for m in SIZES}}
    a_ok = res["diff"] >= a.min_gain and res["ci95"][0] > 0
    b_ok = res["by_n"]["1"] > 0 and res["ci95_by_n"]["1"][0] > 0
    c_ok = all(x > 0 for x in by_pool.values())
    word = "PASS" if a_ok and b_ok and c_ok else "PARTIAL" if res["diff"] > 0 and res["ci95"][0] > 0 else "FAIL"
    out = {"doc": a.doc, "chosen": {"tau": tau, "lam": lam, "w": w, "select": grid[best][0]}, "verdict": res, "verdict_by_pool": by_pool, "all_seeds": allv, "parts": parts,
           "grid": {f"{t} {l_} {w_}": v[0] for (t, l_, w_), v in grid.items()}, "rule": {"a": bool(a_ok), "b": bool(b_ok), "c": bool(c_ok), "verdict": word}}
    print(f"\nchosen on seeds {a.select_seeds}: tau {tau}, lam {lam:g}, w {w:g} ({grid[best][0]:+.4f})")
    print(f"verdict on seeds {sorted({k[1] for k in ver})}: {res['diff']:+.4f} [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]; 1 game {res['by_n']['1']:+.4f} [{res['ci95_by_n']['1'][0]:+.4f}, {res['ci95_by_n']['1'][1]:+.4f}], "
          f"2 games {res['by_n']['2']:+.4f} [{res['ci95_by_n']['2'][0]:+.4f}, {res['ci95_by_n']['2'][1]:+.4f}], 3 games {res['by_n']['3']:+.4f}; pools {({p: round(x, 4) for p, x in by_pool.items()})}")
    print(f"all seeds (the selection seeds included): {allv['diff']:+.4f} [{allv['ci95'][0]:+.4f}, {allv['ci95'][1]:+.4f}]")
    if G.get("znorm") and ver:
        za = boot(*diffs(tasks, ver, tau, lam, 0.0, zn=True), a.boot, rng)
        la = boot(*diffs(tasks, ver, tau, lam, w, zn=False), a.boot, rng)
        out_extra = {"znorm_alone": za, "link_alone": la}
        print(f"   on the verdict seeds, the two parts alone: cohort normalisation {za['diff']:+.4f} [{za['ci95'][0]:+.4f}, {za['ci95'][1]:+.4f}]; linkage add-on {la['diff']:+.4f} [{la['ci95'][0]:+.4f}, {la['ci95'][1]:+.4f}]")
    for part in ("linked", "other"):
        x = parts[part]
        print(f"   questions whose owner is {'LINKED' if part == 'linked' else 'not linked'}: {x['questions']} ({x['questions'] / max(parts['linked']['questions'] + parts['other']['questions'], 1):.3f}); score before {x['base_score']:.4f}; "
              f"change {x['gain']:+.4f} (1 / 2 / 3 games: {x['gain_by_n']['1']:+.4f} / {x['gain_by_n']['2']:+.4f} / {x['gain_by_n']['3']:+.4f})")
    print(f"rule: (a) balanced gain >= {a.min_gain:+.3f} with the interval above 0: {a_ok}; (b) 1-game gain with the interval above 0: {b_ok}; (c) both pools above 0: {c_ok} -> {word}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
