#!/usr/bin/env python
"""Probe T (METHOD_NOTES 2.123): do the QUESTION games of one owner resemble each other more than they resemble his REFERENCE
games, and can a 1-game question borrow the games of a multi-game question of the same owner?

Known since 2.37: the questions of 2 / 3 / 5 games have distinct owners; the owner of a 1-game question is drawn independently,
so about half of the 1-game questions share their owner with a multi-game question of the same pool. Nothing uses that. If the
question games come from another period than the reference games (2.9: the orientation habit drifted in the original pools),
two questions of one owner carry evidence about each other that the references do not carry.

Part A (no answers; matched sizes). For the questions of n = 5 / 3 / 2 games: one game g of the question is the query, the
other n-1 are its "mates". Four galleries of (n-1)-game sets, one entry per question of that size:
  QQ  query g, entries = the mates of every question                  (own entry: its own mates)
  QR  query g, entries = n-1 random reference games of every question's Top-1 candidate   (own: the question's Top-1 candidate)
  RR  query r = a reference game of the Top-1 candidate, entries = n-1 of the OTHER reference games of each of them
  RQ  query r, entries = the mates
If question and reference games are exchangeable draws of one player's games the four agree (QR and RQ a little lower: the
Top-1 candidate is not always the owner). QQ above RR and QR = games of the question period resemble each other more.
Part B (emulation on the real question games; pseudo-truth = the full question's assigned candidate): g as a 1-game question
against the 600 candidates, alone and with the link to the multi-game questions: score(g, c) + beta * f(z(g, mates assigned
to c)). Own mates present (the owner has a multi-game question) or removed (he has none: the link can only mislead).
Part C (no answers): the real 1-game questions against the real multi-game questions: how many have a strong link.
  python -I <this> --doc v2o13c_9 --out results/final/v2_probe_qq_v2o13c_9.json
"""
import argparse
import json
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
from aicup.submission import check_game  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.assign import assign_soft  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def own_scores(F, own):
    """(e^-(rank-1) inside the Top-5, Top-1 hit) of column own[q] in every row of F."""
    r = (F > F[np.arange(len(F)), own][:, None]).sum(1)
    return np.where(r < 5, np.exp(-r.astype(float)), 0.0), (r == 0).astype(float)


def boot(x, rng, n=2000):
    """Mean and 95% interval of the mean of x over resampled rows."""
    x = np.asarray(x, dtype=float)
    m = x[rng.integers(len(x), size=(n, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


class Pool:
    """One added pool: games, component inputs and the sequence vectors of every network of the document."""

    def __init__(self, a, doc, tag, pool, emb, store, t0):
        T_ = Path(a.tests_dir)
        self.cands, real_q, _ids, self.sgfs = v2.official_pool(T_ / f"player_identification_candidates2_{tag}.csv", T_ / f"player_identification_test2_{tag}.csv")
        self.players = sorted(self.cands)
        self.C = len(self.players)
        colour = {k: c for games in (*self.cands.values(), *real_q) for k, c in games}
        bad = {k for k in self.sgfs if not check_game(self.sgfs[k], colour[k])[0] or k not in store}
        self.moves = parse_all({k: s_ for k, s_ in self.sgfs.items() if k not in bad}, a.workers)
        self.ref = {p: [(k, c) for k, c in g if k not in bad] for p, g in self.cands.items()}
        qs = [[(k, c) for k, c in q if k not in bad] for q in real_q]
        self.qs = [q for q in qs if q]
        self.cfg, self.store, self.emb = doc["cfg"], store, emb
        self.comps = list(doc["comps"])
        self.other = [c for c in self.comps if c != "seq"]
        self.w = np.asarray(doc["weights"], dtype=float)
        keys = [k for g in self.ref.values() for k, _ in g] + [k for q in self.qs for k, _ in q]
        self.row = {k: i for i, k in enumerate(keys)}
        self.vec, self.wts, missing = {"t50": [], "t0": []}, [], 0
        pick = 0 if pool == "pub2" else 1
        for key, w in doc["nets"].items():
            f = ([x for x in key if f"official_{pool}_" in Path(x).name] or [key[min(pick, len(key) - 1)]])[0]
            self.wts.append(w)
            for step in ("t50", "t0"):
                g = f if step == "t50" else f.replace("_t50.npz", "_t0.npz")
                if step == "t0" and (g == f or not Path(g).exists()):
                    missing += 1
                    self.vec[step].append(None)
                    continue
                z = np.load(g)
                one = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
                self.vec[step].append(np.stack([one[k] for k in keys]))
        self.steps = ["t50"] + (["t0"] if missing == 0 else [])
        print(f"{pool}: {self.C} candidates, {len(self.qs)} questions, {len(keys)} games ({len(bad)} unusable), {len(self.wts)} networks ({sum(self.wts)} with weights); "
              f"vectors before the fine-tune missing for {missing} ({time.time() - t0:.0f} s)", flush=True)

    def member(self, sets):
        rows = np.repeat(np.arange(len(sets)), [len(s) for s in sets])
        cols = np.array([self.row[k] for s in sets for k, _ in s])
        return sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(sets), len(self.row)))

    def seq(self, ref_sets, queries, step):
        A, B = self.member(ref_sets), self.member(queries)
        acc = 0.0
        for V, w in zip(self.vec[step], self.wts):
            acc = acc + w * standardise(unit(np.asarray(B @ V, dtype=np.float64)) @ unit(np.asarray(A @ V, dtype=np.float64)).T)
        return acc / sum(self.wts)

    def fuse(self, ref, queries):
        """(fused [Q, C], {component: row-standardised [Q, C]}) of the queries against the sets ref {name: [(key, colour)]}, sorted names."""
        names = sorted(ref)
        try:
            S = v2.components(ref, queries, self.moves, self.sgfs, self.cfg, self.store, self.emb, which=self.other)
        except Exception as e:  # a model that cannot be fitted on very small sets: fit them one by one, leave the failing ones out (zero)
            print(f"  components together failed ({type(e).__name__}: {e}); one by one", flush=True)
            S = {}
            for c in self.other:
                try:
                    S.update(v2.components(ref, queries, self.moves, self.sgfs, self.cfg, self.store, self.emb, which=[c]))
                except Exception as e2:
                    print(f"  component {c} left out ({type(e2).__name__}: {e2})", flush=True)
                    S[c] = np.zeros((len(queries), len(names)))
        S["seq"] = self.seq([ref[p] for p in names], queries, "t50")
        Z = {c: standardise(np.nan_to_num(S[c], nan=0.0, posinf=0.0, neginf=0.0)) for c in self.comps}
        return sum(wj * Z[c] for wj, c in zip(self.w, self.comps)), Z


def part_a(P, top1, sizes, reps, rng):
    out = {}
    for n in sizes:
        J = [j for j, q in enumerate(P.qs) if len(q) == n]
        cnt = np.bincount(top1[J], minlength=P.C)
        J = [j for j in J if cnt[top1[j]] == 1 and len(P.ref[P.players[top1[j]]]) >= n]  # one question per Top-1 candidate: "his references" is one entry
        names, own, acc = [f"e{t:03d}" for t in range(len(J))], np.arange(len(J)), {}
        for _ in range(reps):
            gi = rng.integers(n, size=len(J))
            qg = [[P.qs[j][g]] for j, g in zip(J, gi)]
            mates = {nm: [x for i, x in enumerate(P.qs[j]) if i != g] for nm, j, g in zip(names, J, gi)}
            refs = [P.ref[P.players[top1[j]]] for j in J]
            perm = [rng.permutation(len(r)) for r in refs]
            rq = [[r[p[0]]] for r, p in zip(refs, perm)]
            sub_rr = {nm: [r[i] for i in p[1:n]] for nm, r, p in zip(names, refs, perm)}
            sub_qr = {nm: [r[i] for i in rng.permutation(len(r))[:n - 1]] for nm, r in zip(names, refs)}
            for gal, ref_, qry in (("QQ", mates, qg), ("QR", sub_qr, qg), ("RR", sub_rr, rq), ("RQ", mates, rq)):
                F, Z = P.fuse(ref_, qry)
                mats = [("fused", F), *Z.items()]
                if "t0" in P.steps:
                    mats.append(("seq_t0", P.seq([ref_[nm] for nm in names], qry, "t0")))
                for what, M in mats:
                    acc.setdefault((gal, what), []).append(np.stack(own_scores(M, own)))
        res = {"questions": len(J)}
        mean = {k: np.mean(v, axis=0) for k, v in acc.items()}  # [2, J]: score, Top-1
        for what in sorted({w for _g, w in mean}):
            row = {gal: [float(mean[(gal, what)][0].mean()), float(mean[(gal, what)][1].mean())] for gal in ("QQ", "QR", "RR", "RQ")}
            for x, y in (("QQ", "RR"), ("QQ", "QR"), ("RR", "QR"), ("RQ", "QR")):
                row[f"{x}-{y}"] = boot(mean[(x, what)][0] - mean[(y, what)][0], rng)
            res[what] = row
        out[str(n)] = res
        for what in ("fused", "seq", *(["seq_t0"] if "t0" in P.steps else []), *[c for c in P.comps if c != "seq"]):
            r = res[what]
            print(f"  n={n} ({len(J)} questions; 1 game against {n - 1}-game sets) {what:<15} score QQ {r['QQ'][0]:.3f}  QR {r['QR'][0]:.3f}  RR {r['RR'][0]:.3f}  RQ {r['RQ'][0]:.3f} | "
                  f"QQ-RR {r['QQ-RR'][0]:+.3f} [{r['QQ-RR'][1]:+.3f}, {r['QQ-RR'][2]:+.3f}]  QQ-QR {r['QQ-QR'][0]:+.3f} [{r['QQ-QR'][1]:+.3f}, {r['QQ-QR'][2]:+.3f}]  "
                  f"RR-QR {r['RR-QR'][0]:+.3f} [{r['RR-QR'][1]:+.3f}, {r['RR-QR'][2]:+.3f}]", flush=True)
    return out


VARIANTS = [("lin", 0.0), ("hinge", 1.0), ("hinge", 2.0), ("hinge", 3.0)]


def link_matrix(Lz, a_m, ok, C, kind, t):
    """[Q, C]: the link evidence of every query for every candidate = sum over the gallery questions assigned to him."""
    K = Lz if kind == "lin" else np.maximum(Lz - t, 0.0)
    M = np.zeros((len(Lz), C))
    np.add.at(M.T, a_m[ok], K[:, ok].T)
    return M


def zrow(L, drop_diag=False):
    L = L.astype(float).copy()
    if drop_diag:
        np.fill_diagonal(L, np.nan)
    z = (L - np.nanmean(L, axis=1, keepdims=True)) / np.maximum(np.nanstd(L, axis=1, keepdims=True), 1e-12)
    return np.nan_to_num(z, nan=0.0)


def part_b(P, big, truth, margin, reps, betas, m0s, rng):
    J = [int(j) for j in np.flatnonzero(big)]
    size = np.array([len(P.qs[j]) for j in J])
    acc, link = {}, {"own": [], "other": []}
    for rep in range(reps):
        gi = [int(rng.integers(len(P.qs[j]))) for j in J]
        qg = [[P.qs[j][g]] for j, g in zip(J, gi)]
        mates = [[x for i, x in enumerate(P.qs[j]) if i != g] for j, g in zip(J, gi)]
        B, _ = P.fuse(P.ref, qg)
        FM, _ = P.fuse(P.ref, mates)
        a_m = assign_soft(FM, 1.0, 5)[:, 0]
        top2 = np.sort(FM, axis=1)[:, -2:]
        mm = top2[:, 1] - top2[:, 0]
        L, _ = P.fuse({f"e{t:03d}": m for t, m in enumerate(mates)}, qg)
        zp, za = zrow(L), zrow(L, drop_diag=True)
        link["own"].append(np.diag(zp).copy())
        link["other"].append(za.max(axis=1))
        s0, h0 = own_scores(B, truth)
        acc.setdefault(("base", "", 0.0, 0.0, 0.0), []).append(np.stack([s0, s0, h0, h0]))
        for kind, t in VARIANTS:
            for m0 in m0s:
                ok = mm >= m0
                Mp, Ma = link_matrix(zp, a_m, ok, P.C, kind, t), link_matrix(za, a_m, ok, P.C, kind, t)
                for beta in betas:
                    sp, hp = own_scores(B + beta * Mp, truth)
                    sa, ha = own_scores(B + beta * Ma, truth)
                    acc.setdefault(("link", kind, t, m0, beta), []).append(np.stack([sp, sa, hp, ha]))
        print(f"  emulation draw {rep}: 1-game score alone {s0.mean():.4f} (Top-1 {h0.mean():.3f}); mates assigned to the pseudo-owner {np.mean(a_m == truth):.3f}; "
              f"own-mate link z median {np.median(np.diag(zp)):.2f}, best other question median {np.median(za.max(1)):.2f}", flush=True)
    mean = {k: np.mean(v, axis=0) for k, v in acc.items()}  # [4, J]
    base = mean[("base", "", 0.0, 0.0, 0.0)][0]
    conf = margin > 2.0
    out = {"questions": len(J), "confident": int(conf.sum()), "base": {"all": float(base.mean()), "confident": float(base[conf].mean()),
                                                                          "by_size": {str(m): float(base[size == m].mean()) for m in sorted(set(size.tolist()))}}, "variants": {}}
    own, oth = np.mean(link["own"], axis=0), np.mean(link["other"], axis=0)
    out["link_z"] = {"own_mate": {str(m - 1): [float(np.quantile(own[size == m], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] for m in sorted(set(size.tolist()))},
                     "best_other": [float(np.quantile(oth, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)],
                     "own_all_draws": np.concatenate(link["own"]).tolist(), "other_all_draws": np.concatenate(link["other"]).tolist()}
    print(f"  link z of the own mates (10/25/50/75/90%), by mate size: " + "; ".join(f"{k} game(s) " + "/".join(f"{x:.1f}" for x in v) for k, v in out["link_z"]["own_mate"].items())
          + " | best OTHER question: " + "/".join(f"{x:.1f}" for x in out["link_z"]["best_other"]), flush=True)
    rows = []
    for key, v in mean.items():
        if key[0] != "link":
            continue
        _l, kind, t, m0, beta = key
        dp, da = v[0] - base, v[1] - base
        mixed = 0.5 * (dp + da)
        r = {"present": boot(dp, rng), "absent": boot(da, rng), "mixed": boot(mixed, rng), "mixed_confident": boot(mixed[conf], rng),
             "present_by_size": {str(m): float(dp[size == m].mean()) for m in sorted(set(size.tolist()))}}
        out["variants"][f"{kind}{t:g}|m{m0:g}|b{beta:g}"] = r
        rows.append((r["mixed"][0], f"{kind}{t:g} m0 {m0:g} beta {beta:g}", r))
    for _s, name, r in sorted(rows, key=lambda x: -x[0])[:12]:
        print(f"  {name:<24} owner has a multi-game question {r['present'][0]:+.4f} [{r['present'][1]:+.4f}, {r['present'][2]:+.4f}]  has none {r['absent'][0]:+.4f} "
              f"[{r['absent'][1]:+.4f}, {r['absent'][2]:+.4f}]  half/half {r['mixed'][0]:+.4f} [{r['mixed'][1]:+.4f}, {r['mixed'][2]:+.4f}]  (confident only {r['mixed_confident'][0]:+.4f}) "
              f"by question size {r['present_by_size']}", flush=True)
    return out


def part_c(P, big, n):
    one = [int(j) for j in np.flatnonzero(n == 1)]
    gal = {f"e{t:03d}": P.qs[int(j)] for t, j in enumerate(np.flatnonzero(big))}
    L, _ = P.fuse(gal, [P.qs[j] for j in one])
    z = zrow(L)
    best = z.max(axis=1)
    out = {"questions": len(one), "gallery": len(gal), "best_z_quantiles": [float(np.quantile(best, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
           "share_above": {str(t): float((best > t).mean()) for t in (3.0, 4.0, 5.0, 6.0, 8.0)}, "best_z": best.tolist(),
           "second_z": np.sort(z, axis=1)[:, -2].tolist()}
    print(f"  real 1-game questions against the {len(gal)} real multi-game questions: best link z 10/25/50/75/90% " + "/".join(f"{x:.1f}" for x in out["best_z_quantiles"])
          + "; share above 3/4/5/6/8: " + "/".join(f"{v:.2f}" for v in out["share_above"].values()), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--reps-a", type=int, default=8)
    ap.add_argument("--reps-b", type=int, default=6)
    ap.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--m0", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
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
    assert doc["comps"][-1] == "seq" and isinstance(raw["emb_model"], str)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / raw["trunk_store"]))
    rng = np.random.default_rng(9700 + a.seed)
    out = {"doc": a.doc, "pools": {}}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        F, _ = P.fuse(P.ref, P.qs)
        n = np.array([len(q) for q in P.qs])
        cache = Path(f"/work/u2507115/pid_v2/assign_F/real_{pool}_{a.doc}.npz")
        if cache.exists():
            print(f"  fused scores against the cached ones of the assignment study: largest difference {np.abs(F - np.load(cache)['F']).max():.2e}", flush=True)
        big = n > 1
        top1 = F.argmax(1)
        top1[big] = assign_soft(F[big], 1.0, 5)[:, 0]  # the final file's Top-1 of the multi-game questions
        top2 = np.sort(F, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        res = {"sizes": {str(m): int((n == m).sum()) for m in sorted(set(n.tolist()))}}
        print(f"{pool} part A ({time.time() - t0:.0f} s)", flush=True)
        res["A"] = part_a(P, top1, (5, 3, 2), a.reps_a, rng)
        print(f"{pool} part B ({time.time() - t0:.0f} s)", flush=True)
        res["B"] = part_b(P, big, top1[big], margin[big], a.reps_b, a.betas, a.m0, rng)
        print(f"{pool} part C ({time.time() - t0:.0f} s)", flush=True)
        res["C"] = part_c(P, big, n)
        out["pools"][pool] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
