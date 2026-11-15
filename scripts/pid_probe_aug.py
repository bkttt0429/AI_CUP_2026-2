#!/usr/bin/env python
"""Probe T2 (METHOD_NOTES 2.123): lend the games of the confidently assigned multi-game questions to their Top-1 candidates
and re-score the 1-game questions against the enlarged reference sets.

Probe T (scripts/pid_probe_qq.py) showed that the question games and the reference games of the added pools are exchangeable
draws of one player's games, and that 2-4 more games of the owner are worth +0.02 to +0.09 to a 1-game question; linking a
1-game question to "its" multi-game question fails because the mate cannot be told from the best of 299 strangers. Here no
link is guessed: every multi-game question whose Top-1 leads its second choice by at least m0 gives its games to that
candidate (10 reference games -> 12 / 13 / 15). A correct assignment only sharpens that candidate's models; the harm comes
from wrong assignments alone.
Part P (official pseudo-pools, true answers): per candidate 3 games held out; the first is his 1-game question (600), the
  other two the 2-game question of a random half of the candidates (300, distinct owners as in the real pools; real multi-game
  questions have 2 / 3 / 5 games: conservative). The 2-game questions are assigned as the pipeline assigns them (soft, lam 1).
Part R (real question games, pseudo-truth = the full question's assigned candidate): one game of every multi-game question as
  a 1-game question, the other 1 / 2 / 4 as its multi-game question; in turn a fifth of the questions lose theirs ("the owner
  has none").
Part F (real pools, no answers): what the change does to the real 1-game questions.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_aug_v2o13c_9.json
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import standardise, topk  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_qq import Pool, boot, own_scores, unit  # noqa: E402

G = {}  # the pool of the running part, inherited by the forked workers


def fuse_v(P, ref, queries, vecs, row):
    """Fused [Q, C] of the queries against the sets ref (sorted names); the sequence vectors vecs (one matrix per network) by row."""
    names = sorted(ref)
    S = v2.components(ref, queries, P.moves, P.sgfs, P.cfg, P.store, P.emb, which=P.other)

    def member(sets):
        r = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        c = np.array([row[k] for s in sets for k, _ in s])
        return sparse.csr_matrix((np.ones(len(c), dtype=np.float32), (r, c)), shape=(len(sets), len(row)))
    A, B = member([ref[p] for p in names]), member(queries)
    acc = 0.0
    for V, w in zip(vecs, P.wts):
        acc = acc + w * standardise(unit(np.asarray(B @ V, dtype=np.float64)) @ unit(np.asarray(A @ V, dtype=np.float64)).T)
    S["seq"] = acc / sum(P.wts)
    return sum(wj * standardise(np.nan_to_num(S[c], nan=0.0, posinf=0.0, neginf=0.0)) for wj, c in zip(P.w, P.comps))


def margins(F):
    top2 = np.sort(F, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def lend(P, ref, questions, a, mg, m0, skip=None):
    """The reference sets with the games of every question whose margin is at least m0 added to its assigned candidate."""
    aug = {p: list(g) for p, g in ref.items()}
    n = 0
    for t, q in enumerate(questions):
        if mg[t] >= m0 and (skip is None or not skip[t]):
            aug[P.players[a[t]]] = aug[P.players[a[t]]] + list(q)
            n += 1
    return aug, n


def pseudo(task):
    """One hold-out seed of the pseudo-pool of G['P']: per draw and m0 the score differences of the 600 1-game questions."""
    s, draws, m0s, ft_dir, pool = task
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
    own = np.arange(P.C)
    F1, F2 = fuse_v(P, ref, one, vecs, row), fuse_v(P, ref, two, vecs, row)
    base, hit = own_scores(F1, own)
    rng = np.random.default_rng(9800 + 31 * s + (0 if pool == "pub2" else 1))
    out = {"base": base, "top1": float(hit.mean()), "diff": {m0: [] for m0 in m0s}, "present": [], "assigned_right": [], "lent": {m0: [] for m0 in m0s}, "lent_right": {m0: [] for m0 in m0s}}
    for _ in range(draws):
        M = np.sort(rng.permutation(P.C)[:P.C // 2])
        a = assign_soft(F2[M], 1.0, 5)[:, 0]
        mg = margins(F2[M])
        out["present"].append(np.isin(own, M))
        out["assigned_right"].append(float((a == M).mean()))
        for m0 in m0s:
            aug, n = lend(P, ref, [two[i] for i in M], a, mg, m0)
            s1, _h = own_scores(fuse_v(P, aug, one, vecs, row), own)
            out["diff"][m0].append(s1 - base)
            out["lent"][m0].append(n)
            out["lent_right"][m0].append(float((a == M)[mg >= m0].mean()) if n else float("nan"))
    return s, out


def real(task):
    """One draw of the emulation on the real question games of G['P']."""
    rep, m0s, pool = task
    P, big, truth = G["P"], G["big"], G["truth"]
    rng = np.random.default_rng(9900 + 17 * rep + (0 if pool == "pub2" else 1))
    J = [int(j) for j in np.flatnonzero(big)]
    gi = [int(rng.integers(len(P.qs[j]))) for j in J]
    qg = [[P.qs[j][g]] for j, g in zip(J, gi)]
    mates = [[x for i, x in enumerate(P.qs[j]) if i != g] for j, g in zip(J, gi)]
    vecs = P.vec["t50"]
    B, FM = fuse_v(P, P.ref, qg, vecs, P.row), fuse_v(P, P.ref, mates, vecs, P.row)
    a, mg = assign_soft(FM, 1.0, 5)[:, 0], margins(FM)
    s0, _h = own_scores(B, truth)
    fold = rng.permutation(len(J)) % 5
    out = {"base": s0, "size": np.array([len(P.qs[j]) for j in J]), "assigned_as_truth": float((a == truth).mean()), "present": {}, "absent": {}}
    for m0 in m0s:
        pres, absn = np.zeros(len(J)), np.zeros(len(J))
        for f in range(5):
            aug, _n = lend(P, P.ref, mates, a, mg, m0, skip=(fold == f))
            s1, _h = own_scores(fuse_v(P, aug, qg, vecs, P.row), truth)
            absn[fold == f] = s1[fold == f]
            pres[fold != f] += s1[fold != f] / 4.0
        out["present"][m0], out["absent"][m0] = pres - s0, absn - s0
    return rep, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--m0", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    raw = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    nets = {}
    for x in raw["seq_official"]:
        key = tuple(x) if isinstance(x, list) else (x,)
        nets[key] = nets.get(key, 0) + 1
    doc = {"cfg": raw["cfg"], "comps": raw["components"], "weights": raw["weights"], "nets": nets}
    torch.set_num_threads(1)  # the workers are forked: no CUDA and no thread pool in the parent
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    rng = np.random.default_rng(9777)
    out = {"doc": a.doc, "m0": a.m0, "pools": {}}
    keep = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        F = fuse_v(P, P.ref, P.qs, P.vec["t50"], P.row)
        n = np.array([len(q) for q in P.qs])
        big = n > 1
        a_big, mg_big = assign_soft(F[big], 1.0, 5)[:, 0], margins(F[big])
        G.update(P=P, big=big, truth=a_big)
        tasks = [("pseudo", (s, a.draws, a.m0, a.ft_dir, pool)) for s in a.seeds] + [("real", (r, a.m0, pool)) for r in range(a.reps)]
        with mp.get_context("fork").Pool(min(a.procs, len(tasks))) as ex:
            got = [ex.apply_async(pseudo if kind == "pseudo" else real, (t,)) for kind, t in tasks]
            got = [g.get() for g in got]
        ps = dict(got[:len(a.seeds)])
        rs = dict(got[len(a.seeds):])
        res = {"P": {}, "R": {}, "F": {}}
        # ---- part P
        print(f"{pool} part P: 1-game score without lending, per seed " + " ".join(f"{ps[s]['base'].mean():.4f}" for s in a.seeds)
              + f"; 2-game questions assigned to their owner {np.mean([np.mean(ps[s]['assigned_right']) for s in a.seeds]):.3f} ({time.time() - t0:.0f} s)", flush=True)
        for m0 in a.m0:
            # per candidate: mean over seeds and draws, separately for the draws where he owned a 2-game question and where he did not
            dp, da, dall, runs = np.zeros(P.C), np.zeros(P.C), np.zeros(P.C), []
            cp, ca = np.zeros(P.C), np.zeros(P.C)
            for s in a.seeds:
                dd = np.stack(ps[s]["diff"][m0])
                pr = np.stack(ps[s]["present"])
                dp += (dd * pr).sum(0); cp += pr.sum(0)
                da += (dd * ~pr).sum(0); ca += (~pr).sum(0)
                dall += dd.mean(0) / len(a.seeds)
                runs.append(float(dd.mean()))
            okp, oka = cp > 0, ca > 0
            r = {"all": boot(dall, rng), "present": boot(dp[okp] / cp[okp], rng), "absent": boot(da[oka] / ca[oka], rng), "per_seed": runs,
                 "lent": float(np.mean([np.mean(ps[s]["lent"][m0]) for s in a.seeds])), "lent_right": float(np.nanmean([np.nanmean(ps[s]["lent_right"][m0]) for s in a.seeds]))}
            res["P"][f"{m0:g}"] = r
            keep.setdefault(("P", m0), []).append(dall)
            print(f"  m0 {m0:g}: {r['lent']:.0f} of 300 questions lent ({r['lent_right']:.3f} of them to the owner) | all 1-game questions {r['all'][0]:+.4f} [{r['all'][1]:+.4f}, {r['all'][2]:+.4f}]  "
                  f"owner has a 2-game question {r['present'][0]:+.4f} [{r['present'][1]:+.4f}, {r['present'][2]:+.4f}]  has none {r['absent'][0]:+.4f} [{r['absent'][1]:+.4f}, {r['absent'][2]:+.4f}]  "
                  f"per seed " + " ".join(f"{x:+.4f}" for x in runs), flush=True)
        # ---- part R
        size = rs[0]["size"]
        base = np.mean([rs[r]["base"] for r in rs], axis=0)
        print(f"{pool} part R: 1-game score without lending {base.mean():.4f}; mates assigned as the full question {np.mean([rs[r]['assigned_as_truth'] for r in rs]):.3f} ({time.time() - t0:.0f} s)", flush=True)
        for m0 in a.m0:
            dp = np.mean([rs[r]["present"][m0] for r in rs], axis=0)
            da = np.mean([rs[r]["absent"][m0] for r in rs], axis=0)
            mixed = 0.5 * (dp + da)
            r = {"present": boot(dp, rng), "absent": boot(da, rng), "mixed": boot(mixed, rng), "present_by_size": {str(m): float(dp[size == m].mean()) for m in sorted(set(size.tolist()))}}
            res["R"][f"{m0:g}"] = r
            keep.setdefault(("R", m0), []).append(mixed)
            print(f"  m0 {m0:g}: owner has a multi-game question {r['present'][0]:+.4f} [{r['present'][1]:+.4f}, {r['present'][2]:+.4f}]  has none {r['absent'][0]:+.4f} [{r['absent'][1]:+.4f}, {r['absent'][2]:+.4f}]  "
                  f"half/half {r['mixed'][0]:+.4f} [{r['mixed'][1]:+.4f}, {r['mixed'][2]:+.4f}]  by question size {r['present_by_size']}", flush=True)
        # ---- part F: the real 1-game questions
        one = np.flatnonzero(n == 1)
        q_one, q_big = [P.qs[int(j)] for j in one], [P.qs[int(j)] for j in np.flatnonzero(big)]
        t5 = topk(F[one], 5)
        for m0 in a.m0:
            aug, k = lend(P, P.ref, q_big, a_big, mg_big, m0)
            Fa = fuse_v(P, aug, q_one, P.vec["t50"], P.row)
            n5 = topk(Fa, 5)
            lent_to = np.zeros(P.C, dtype=bool)
            lent_to[a_big[mg_big >= m0]] = True
            r = {"lent": int(k), "top1_changed": int((n5[:, 0] != t5[:, 0]).sum()), "top5_changed": int((n5 != t5).any(1).sum()),
                 "top1_enlarged_before": int(lent_to[t5[:, 0]].sum()), "top1_enlarged_after": int(lent_to[n5[:, 0]].sum())}
            res["F"][f"{m0:g}"] = r
            print(f"{pool} part F m0 {m0:g}: {k} of {int(big.sum())} multi-game questions lent; of the {len(one)} 1-game questions {r['top1_changed']} change their Top-1, {r['top5_changed']} their Top-5; "
                  f"Top-1 is an enlarged candidate: {r['top1_enlarged_before']} before, {r['top1_enlarged_after']} after", flush=True)
        out["pools"][pool] = res
    out["both"] = {}
    for (part, m0), arrs in sorted(keep.items()):
        both = boot(np.concatenate(arrs), rng)
        out["both"][f"{part}|{m0:g}"] = both
        print(f"both pools, part {part}, m0 {m0:g}: {both[0]:+.4f} [{both[1]:+.4f}, {both[2]:+.4f}] per 1-game question", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
