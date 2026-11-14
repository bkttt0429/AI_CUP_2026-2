#!/usr/bin/env python
"""How many repeated owners can the multi-game questions of the REAL added pools have, and which assignment is best below
that bound (METHOD_NOTES 2.122, second step)?

scripts/pid_v2_assign_tune.py showed that the best assignment depends on the share of repeated owners among the questions of
2 / 3 / 5 games: with none, harder assignments gain +0.005 to +0.009 per question over the soft assignment with lam = 1; with
5% they lose. The 11-13 statistics (weaker scores) could only say "0-5%". With the current scores a label-free statistic
bounds it more tightly:
  T_m = number of pairs of multi-game questions that name the SAME Top-1 candidate while BOTH lead their second choice by
        more than m (fused score, before any assignment).
Two confident questions naming one candidate are either a repeated owner or a confident mistake, so a small T_m bounds the
number of repeated-owner pairs: with f_m the share of multi-game questions whose margin exceeds m, a repeated pair shows up
in T_m with probability about f_m^2 (taken without the positive correlation inside one owner: conservative).
Part A (emulation, official pseudo-pools as pid_v2_assign_tune.py, fused scores cached): the methods' gains over soft lam 1
at contamination 0 / 1 / 2 / 3 / 5%, and T_m of the draws. Part B (real pools, no answers): T_m and f_m of each pool under
the same fusion document, and the bound.
  python -I <this> --doc v2o13c_9 --out results/final/v2_assign_struct_v2o13c_9.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

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
from pid_v2_assign_tune import score, sinkhorn_prices  # noqa: E402

MARGINS = (0.0, 1.0, 2.0, 3.0, 4.0)


def pair_stats(S):
    """({m: T_m}, {m: f_m}) of a score matrix [Q, C]."""
    top2 = np.sort(S, axis=1)[:, -2:]
    margin, t1 = top2[:, 1] - top2[:, 0], S.argmax(1)
    T, f = {}, {}
    for m in MARGINS:
        k = np.bincount(t1[margin > m], minlength=S.shape[1])
        T[m] = int((k * (k - 1) // 2).sum())
        f[m] = float((margin > m).mean())
    return T, f


def with_first(first, order):
    """Top-5 lists: `first` [Q] as the Top-1, then the row order of `order` [Q, 5+]."""
    out = order[:, :5].copy()
    for q, c in enumerate(first):
        if out[q, 0] != c:
            out[q] = np.array([c, *[x for x in order[q].tolist() if x != c]][:5])
    return out


def methods(S, lams=(1.0, 2.0, 3.0, 4.0), taus=(0.5, 1.0)):
    out = {"none": topk(S, 5), "hard": assign_topk(S, 5)}
    soft = {lam: assign_soft(S, lam, 5) for lam in lams}
    for lam, idx in soft.items():
        out[f"soft{lam:g}"] = idx
    for tau in taus:
        order = topk(S + tau * sinkhorn_prices(S, tau)[None, :], 8)
        out[f"sink{tau:g}"] = order[:, :5]
        for lam, idx in soft.items():
            if lam > 1:
                out[f"soft{lam:g}+sink{tau:g}"] = with_first(idx[:, 0], order)
    return out


def seq_component(vec_files, players, ref, questions):
    """Mean of the networks' row-standardised cosine matrices; vec_files [(path, weight)]."""
    unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
    acc, tot = 0.0, 0
    for paths, w in vec_files:
        one = {}
        for f in paths:
            z = np.load(f)
            one.update(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        Cm = unit(np.stack([np.sum([one[k] for k, _ in ref[p]], axis=0) for p in players]).astype(np.float64))
        Qm = unit(np.stack([np.sum([one[k] for k, _ in q], axis=0) for q in questions]).astype(np.float64))
        acc = acc + w * standardise(Qm @ Cm.T)
        tot += w
    return acc / tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--cache", default="/work/u2507115/pid_v2/assign_F")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--draws", type=int, default=120)
    ap.add_argument("--per-size", type=int, default=150)
    ap.add_argument("--contam", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05])
    ap.add_argument("--ref", default="soft1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    T_ = Path(a.tests_dir)
    doc = json.loads((ROOT / "results" / f"pid_fusion_{a.doc}.json").read_text())
    comps = list(doc["components"])
    assert comps[-1] == "seq" and isinstance(doc["emb_model"], str)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))
    store = load_store(str(ROOT / doc["trunk_store"]))
    weight = {}
    for x in doc["seq_official"]:  # [pub2 file, priv2 file] per network; a repeated entry is its weight
        key = tuple(x) if isinstance(x, list) else (x,)
        weight[key] = weight.get(key, 0) + 1
    fus = Fusion()
    fus.w = np.asarray(doc["weights"], dtype=float)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"document {a.doc}: {len(weight)} distinct sequence networks ({sum(weight.values())} with weights)", flush=True)
    res, emu_T, real = {}, {p: {m: [] for m in MARGINS} for p in a.contam}, {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, real_q, _ids, sgfs = v2.official_pool(T_ / f"player_identification_candidates2_{tag}.csv", T_ / f"player_identification_test2_{tag}.csv")
        players = sorted(cands)
        C = len(players)
        colour = {k: c for games in (*cands.values(), *real_q) for k, c in games}
        bad = {k for k in sgfs if not check_game(sgfs[k], colour[k])[0] or k not in store}
        moves = parse_all({k: s_ for k, s_ in sgfs.items() if k not in bad}, a.workers)
        # ---- part B: the real pool, no answers
        fr = cache / f"real_{pool}_{a.doc}.npz"
        if fr.exists():
            z = np.load(fr)
            F, n = z["F"], z["n"]
        else:
            ref = {p: [(k, c) for k, c in g if k not in bad] for p, g in cands.items()}
            qs = [[(k, c) for k, c in q if k not in bad] for q in real_q]
            qs = [q for q in qs if q]
            S = v2.components(ref, qs, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
            pick = 0 if pool == "pub2" else 1
            S["seq"] = seq_component([([f for f in key if f"official_{pool}_" in Path(f).name] or [key[min(pick, len(key) - 1)]], w) for key, w in weight.items()], players, ref, qs)
            F, n = fus.scores([standardise(S[c]) for c in comps]), np.array([len(q) for q in qs])
            np.savez(fr, F=F.astype(np.float32), n=n)
        big = n > 1
        Tm, fm = pair_stats(F[big])
        idx1 = assign_soft(F[big], 1.0, 5)
        real[pool] = {"multi_game_questions": int(big.sum()), "T": {f"{m:g}": v for m, v in Tm.items()}, "f": {f"{m:g}": v for m, v in fm.items()},
                      "by_size_distinct_top1": {str(m): int(len(set(F[n == m].argmax(1).tolist()))) for m in sorted(set(n.tolist()))},
                      "soft1_changes": int((idx1[:, 0] != F[big].argmax(1)).sum()),
                      "changes": {name: int((idx[:, 0] != idx1[:, 0]).sum()) for name, idx in methods(F[big]).items()}}
        print(f"REAL {pool}: {int(big.sum())} multi-game questions; T_m " + " ".join(f"m>{m:g}:{v}" for m, v in Tm.items()) + "; f_m " + " ".join(f"{v:.3f}" for v in fm.values())
              + f"; soft lam 1 changes {real[pool]['soft1_changes']} Top-1 answers; Top-1 answers that differ from soft lam 1: " + " ".join(f"{k}:{v}" for k, v in real[pool]["changes"].items()) + f" ({time.time() - t0:.0f} s)", flush=True)
        # ---- part A: emulation on the pseudo-pools of this pool
        ckeys = {k for g in cands.values() for k, _ in g}
        for s in a.seeds:
            fe = cache / f"offsim_{pool}_s{s}_{a.doc}.npz"
            if fe.exists():
                z = np.load(fe)
                F, owner, size = z["F"], z["owner"], z["size"]
            else:
                hold = v2.candidate_holdout(cands, 3, s)
                held = {k for g in hold.values() for k, _ in g}
                ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
                questions, owner, size = [], [], []
                for i, p in enumerate(players):
                    h = [(k, c) for k, c in hold[p] if k not in bad]
                    for m in (2, 3):
                        if len(h) >= m:
                            questions.append(h[:m]); owner.append(i); size.append(m)
                owner, size = np.array(owner), np.array(size)
                S = v2.components(ref, questions, {k: v_ for k, v_ in moves.items() if k in ckeys}, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])
                files = []
                for key, w in weight.items():
                    v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
                    files.append(([str(Path(a.ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")], w))
                S["seq"] = seq_component(files, players, ref, questions)
                F = fus.scores([standardise(S[c]) for c in comps])
                np.savez(fe, F=F.astype(np.float32), owner=owner, size=size)
            row = {(int(o), int(m)): j for j, (o, m) in enumerate(zip(owner, size))}
            have = [i for i in range(C) if (i, 2) in row and (i, 3) in row]
            rng = np.random.default_rng(9500 + 17 * s + (0 if pool == "pub2" else 1))
            acc = {}
            for _ in range(a.draws):
                perm = rng.permutation(have)
                base_q = [(int(c), 3) for c in perm[:a.per_size]] + [(int(c), 2) for c in perm[a.per_size:2 * a.per_size]]
                for p in a.contam:
                    q = list(base_q)
                    k = int(round(p * len(q)))
                    if k:
                        drop = set(rng.choice(len(q), k, replace=False).tolist())
                        keep = [x for j, x in enumerate(q) if j not in drop]
                        donors = rng.choice(len(keep), k, replace=False)
                        q = keep + [(keep[j][0], 5 - keep[j][1]) for j in donors]
                    Sq, truth = F[np.array([row[x] for x in q])], np.array([x[0] for x in q])
                    Tm, _f = pair_stats(Sq)
                    for m in MARGINS:
                        emu_T[p][m].append(Tm[m])
                    for name, idx in methods(Sq).items():
                        acc.setdefault((p, name), []).append(score(idx, truth))
            for key, vals in acc.items():
                res.setdefault(key, []).append(float(np.mean(vals)))
            print(f"{pool} split {s}: " + "  ".join(f"{nm} {np.mean(acc[(0.0, nm)]):.4f}" for nm in ("none", a.ref, "hard")) + f" ({time.time() - t0:.0f} s)", flush=True)
    names = sorted({nm for _p, nm in res})
    out = {"doc": a.doc, "seeds": a.seeds, "draws": a.draws, "ref": a.ref, "real": real, "gain": {}, "emulated_T": {}}
    for p in a.contam:
        refv = np.array(res[(p, a.ref)])
        print(f"contamination {p:g}: gain over {a.ref} (mean over {len(refv)} runs [min, max], runs up)")
        for nm in names:
            d_ = np.array(res[(p, nm)]) - refv
            out["gain"][f"{p:g}|{nm}"] = {"score": float(np.mean(res[(p, nm)])), "diff": float(d_.mean()), "min": float(d_.min()), "max": float(d_.max()), "runs_up": int((d_ > 0).sum())}
            print(f"  {nm:<14} {np.mean(res[(p, nm)]):.4f}  {d_.mean():+.4f} [{d_.min():+.4f}, {d_.max():+.4f}]  up {int((d_ > 0).sum())}/{len(d_)}", flush=True)
        out["emulated_T"][f"{p:g}"] = {f"{m:g}": {"mean": float(np.mean(v)), "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95))} for m, v in emu_T[p].items()}
        print(f"  emulated T_m (mean [5%, 95%]) of one draw of {2 * a.per_size} questions: " + "  ".join(f"m>{m:g}: {np.mean(v):.1f} [{np.quantile(v, 0.05):.0f}, {np.quantile(v, 0.95):.0f}]" for m, v in emu_T[p].items()), flush=True)
    # the bound on repeated-owner pairs from the real pools (both pools together; every confident same-Top-1 pair is counted as a repeat)
    out["bound"] = {}
    for m in (2.0, 3.0):
        t = sum(real[pool]["T"][f"{m:g}"] for pool in real)
        f2 = np.mean([real[pool]["f"][f"{m:g}"] for pool in real]) ** 2
        up = float(0.5 * chi2.ppf(0.95, 2 * (t + 1)))  # upper 95% bound of a Poisson mean given t observed (3.0 for t = 0)
        k = up / max(f2, 1e-9)
        q_total = sum(real[pool]["multi_game_questions"] for pool in real)
        out["bound"][f"{m:g}"] = {"T_both_pools": int(t), "f_squared": float(f2), "pairs_upper95": float(k), "share_upper95": float(2 * k / q_total)}
        print(f"bound from T_{m:g}: {t} confident same-Top-1 pairs in both pools, detection f^2 = {f2:.3f} -> at most {k:.1f} repeated-owner pairs (95%), "
              f"i.e. {200 * k / q_total:.1f}% of the {q_total} multi-game questions", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
