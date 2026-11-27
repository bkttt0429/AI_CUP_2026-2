#!/usr/bin/env python
"""Follow-ups to scripts/pid_probe_bottleneck.py (METHOD_NOTES 2.139): exploratory measurements on the official pseudo-pools'
1-game questions (true answers), one fusion document. Nothing is chosen here: a variant that looks good gets a registered
test of its own.
  K  headroom of a second stage: the score if the owner were always first whenever he is inside the fused Top-2 / Top-5 / Top-10
  L  the white / black gap by source: the sequence component, the other components, the fused score
  M  mutual look-alike pairs: how many candidates sit in one, how strong, and the share of the lost score they carry
  N  what a rarely recognised candidate looks like: his own strength (owner's standard score), how attractive he is to
     foreign questions (column mean), how close his nearest neighbour is
  O  score normalisations that need no answers: per-candidate (column) centring / standardising of every network's cosine
     matrix before the row standardisation; the same on the fused matrix; symmetric (row + column) normalisation
  P  nearest-game evidence: the question against the candidate's single reference games (max, mean of the best three)
     instead of their mean vector - as a further component for all candidates, and as a re-ranking of the fused Top-5
Every variant: score difference against the present fused score, bootstrap over candidates, for hold-out seeds 0 1 and 2 3
separately and together.
  python -I <this> --doc v2o19d_j --out results/final/v2_probe_bottleneck2_v2o19d_j.json
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
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import standardise  # noqa: E402
from aicup_pid.trunkstyle import load_store  # noqa: E402
from pid_probe_bottleneck import ranks_of, score_of  # noqa: E402
from pid_probe_qq import Pool, boot, unit  # noqa: E402

G = {}


def col_std(S):
    return standardise(S.T).T


def one_seed(task):
    s, ft_dir, pool = task
    P = G["P"]
    C = P.C
    hold = v2.candidate_holdout(P.cands, 3, s)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in P.ref[p] if k not in held] for p in P.players}
    one = [[g] for p in P.players for g in hold[p]]
    own = np.repeat(np.arange(C), 3)
    Q = len(one)
    ckeys = [k for p in P.players for k, _ in P.cands[p]]
    row = {k: i for i, k in enumerate(ckeys)}
    n_ref = np.array([len(ref[p]) for p in P.players])
    q_idx = np.array([row[q[0][0]] for q in one])
    g_idx = np.array([row[k] for p in P.players for k, _ in ref[p]])
    g_start = np.concatenate([[0], np.cumsum(n_ref)[:-1]])
    A = sparse.csr_matrix((np.ones(len(g_idx), dtype=np.float32), (np.repeat(np.arange(C), n_ref), g_idx)), shape=(C, len(row)))
    acc = {k: 0.0 for k in ("base", "colcentre", "colstd", "nnmax", "nntop3")}
    same7 = bool((n_ref == n_ref[0]).all())
    for key, w in zip(P.net_keys, P.wts):
        v, tg = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(key[0]).name).groups()
        z = np.load(Path(ft_dir) / f"offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")
        d = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        V = np.stack([d[k] for k in ckeys])
        qv = unit(V[q_idx].astype(np.float64))
        cos = qv @ unit(np.asarray(A @ V, dtype=np.float64)).T  # question against the candidates' mean vectors
        acc["base"] += w * standardise(cos)
        acc["colcentre"] += w * standardise(cos - cos.mean(0, keepdims=True))
        acc["colstd"] += w * standardise(col_std(cos))
        gg = qv @ unit(V[g_idx].astype(np.float64)).T  # question against every single reference game
        acc["nnmax"] += w * standardise(np.maximum.reduceat(gg, g_start, axis=1))
        if same7:
            top3 = np.sort(gg.reshape(Q, C, int(n_ref[0])), axis=2)[:, :, -3:].mean(2)
            acc["nntop3"] += w * standardise(top3)
    W = float(sum(P.wts))
    seq = {k: v / W for k, v in acc.items() if not isinstance(v, float)}
    S = v2.components(ref, one, P.moves, P.sgfs, P.cfg, P.store, P.emb, which=P.other)
    Zc = {c: standardise(np.nan_to_num(S[c], nan=0.0, posinf=0.0, neginf=0.0)) for c in P.other}
    wj = dict(zip(P.comps, P.w))
    rest = sum(wj[c] * Zc[c] for c in P.other)
    F = rest + wj["seq"] * seq["base"]
    var = {"base": F, "seq alone": seq["base"], "others alone": rest,
           "nets column-centred": rest + wj["seq"] * seq["colcentre"], "nets column-standardised": rest + wj["seq"] * seq["colstd"],
           "fused column-centred": F - F.mean(0, keepdims=True), "fused column-standardised": col_std(F), "fused symmetric (row + column)": 0.5 * (standardise(F) + col_std(F))}
    for b in (0.5, 1.0, 2.0):
        var[f"+ nearest game (max) x{b:g}"] = F + b * wj["seq"] * seq["nnmax"] / 4.0
        if "nntop3" in seq:
            var[f"+ best three games x{b:g}"] = F + b * wj["seq"] * seq["nntop3"] / 4.0
    top5 = np.argsort(-F, axis=1)[:, :5]
    for name in ("nnmax", "nntop3"):
        if name not in seq:
            continue
        for b in (0.25, 1.0):
            R = np.full_like(F, -1e9)
            rows = np.arange(Q)[:, None]
            R[rows, top5] = standardise(F)[rows, top5] + b * standardise(seq[name])[rows, top5]
            Fr = np.where(R > -1e8, R + 1e6, F)  # the re-ranked five stay in front, the rest keep their order
            var[f"Top-5 re-ranked with {'nearest game' if name == 'nnmax' else 'best three games'} x{b:g}"] = Fr
    rk = {k: ranks_of(M, own) for k, M in var.items()}
    r0 = rk["base"]
    Z = standardise(F)
    out = {"own": own, "rank": rk, "colour": np.array([q[0][1] for q in one]), "z_own": Z[np.arange(Q), own], "col_mean": Z.mean(0), "top1": F.argmax(1),
           "headroom": {str(k): float(np.where(r0 < k, 1.0, score_of(r0)).mean()) for k in (2, 5, 10)}}
    Zo = Z.copy()
    Zo[np.arange(Q), own] = -np.inf
    out["best_other"] = Zo.max(1)
    return s, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--doc", required=True)
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--procs", type=int, default=4)
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
    torch.set_num_threads(1)
    emb = PlayerEmbedding.from_state(torch.load(str(ROOT / raw["emb_model"]), map_location="cpu", weights_only=False), device="cpu")
    store = load_store(str(ROOT / raw["trunk_store"]))
    rng = np.random.default_rng(7411)
    out = {"doc": a.doc, "pools": {}}
    keep = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        P = Pool(a, doc, tag, pool, emb, store, t0)
        P.net_keys = list(nets)
        C = P.C
        G.update(P=P)
        with mp.get_context("fork").Pool(min(a.procs, len(a.seeds))) as ex:
            got = dict(ex.map(one_seed, [(s, a.ft_dir, pool) for s in a.seeds]))
        names = list(got[a.seeds[0]]["rank"])
        own = np.concatenate([got[s]["own"] for s in a.seeds])
        r0 = np.concatenate([got[s]["rank"]["base"] for s in a.seeds])
        s0 = score_of(r0)
        res = {"score": float(s0.mean()), "variants": {}}
        print(f"\n===== {pool}: {len(r0)} one-game questions; fused score {s0.mean():.4f} ({time.time() - t0:.0f} s)", flush=True)
        # K
        res["headroom"] = {k: float(np.mean([got[s]["headroom"][k] for s in a.seeds])) for k in ("2", "5", "10")}
        print("K  score if the owner were first whenever he is inside the fused Top-k: " + "  ".join(f"Top-{k}: {v:.3f}" for k, v in res["headroom"].items()) + f"  (now {s0.mean():.3f})", flush=True)
        # L
        col = np.concatenate([got[s]["colour"] for s in a.seeds])
        res["by_colour"] = {}
        for k in ("base", "seq alone", "others alone"):
            sk = score_of(np.concatenate([got[s]["rank"][k] for s in a.seeds]))
            res["by_colour"][k] = {str(c): float(sk[col == c].mean()) for c in np.unique(col)}
        print("L  score by subject colour: " + " | ".join(f"{k}: " + "  ".join(f"{c} {v:.3f}" for c, v in d_.items()) for k, d_ in res["by_colour"].items()), flush=True)
        # M
        top1 = np.concatenate([got[s]["top1"] for s in a.seeds])
        wrong = r0 > 0
        cnt = {}
        for o_, t_ in zip(own[wrong], top1[wrong]):
            cnt[(int(o_), int(t_))] = cnt.get((int(o_), int(t_)), 0) + 1
        lost = 1.0 - s0
        strong, in_pair, lost_pair = [], set(), 0.0
        for (o_, t_), v in cnt.items():
            if o_ < t_ and (t_, o_) in cnt:
                strong.append((v, cnt[(t_, o_)], o_, t_))
        for v, v2_, o_, t_ in strong:
            if v >= 2 and v2_ >= 2:
                in_pair.update((o_, t_))
        pair_of = {}
        for v, v2_, o_, t_ in strong:
            if v >= 2 and v2_ >= 2:
                pair_of[o_], pair_of[t_] = t_, o_
        for o_, t_, l_ in zip(own[wrong], top1[wrong], lost[wrong]):
            if pair_of.get(int(o_)) == int(t_):
                lost_pair += float(l_)
        strong.sort(reverse=True)
        res["mutual_pairs"] = {"pairs_with_errors_both_ways": len(strong), "pairs_two_or_more_each_way": int(sum(1 for v, v2_, *_ in strong if v >= 2 and v2_ >= 2)),
                               "candidates_in_such_a_pair": float(len(in_pair) / C), "share_of_lost_score_inside_them": float(lost_pair / lost.sum()),
                               "strongest": [[int(v), int(v2_)] for v, v2_, *_ in strong[:8]]}
        mp_ = res["mutual_pairs"]
        print(f"M  mutual pairs: {mp_['pairs_with_errors_both_ways']} with an error each way, {mp_['pairs_two_or_more_each_way']} with two or more each way ({mp_['candidates_in_such_a_pair']:.3f} of the candidates), "
              f"carrying {mp_['share_of_lost_score_inside_them']:.3f} of the lost score; strongest (errors a->b, b->a of 12 each): {mp_['strongest']}", flush=True)
        # N
        n_q = np.bincount(own, minlength=C)
        rate = np.bincount(own, weights=(r0 == 0).astype(float), minlength=C) / n_q
        zo = np.concatenate([got[s]["z_own"] for s in a.seeds])
        bo = np.concatenate([got[s]["best_other"] for s in a.seeds])
        strength = np.bincount(own, weights=zo, minlength=C) / n_q
        rival = np.bincount(own, weights=bo, minlength=C) / n_q
        attract = np.mean([got[s]["col_mean"] for s in a.seeds], axis=0)
        cc = lambda x, y: float(np.corrcoef(x, y)[0, 1])  # noqa: E731
        low, high = rate <= 0.25, rate >= 0.75
        res["rarely_recognised"] = {"corr_rate_own_strength": cc(rate, strength), "corr_rate_best_rival": cc(rate, rival), "corr_rate_attractiveness": cc(rate, attract),
                                    "own_strength_low_vs_high": [float(strength[low].mean()), float(strength[high].mean())], "best_rival_low_vs_high": [float(rival[low].mean()), float(rival[high].mean())],
                                    "attractiveness_low_vs_high": [float(attract[low].mean()), float(attract[high].mean())], "low_in_mutual_pair": float(np.mean([c_ in in_pair for c_ in np.flatnonzero(low)]))}
        rr = res["rarely_recognised"]
        print(f"N  a candidate's hit rate against: his own standard score r = {rr['corr_rate_own_strength']:+.2f}, his best rival's r = {rr['corr_rate_best_rival']:+.2f}, how attractive he is to foreign questions r = {rr['corr_rate_attractiveness']:+.2f}; "
              f"rarely (<= 1/4) vs mostly (>= 3/4) recognised: own {rr['own_strength_low_vs_high'][0]:.2f} vs {rr['own_strength_low_vs_high'][1]:.2f}, best rival {rr['best_rival_low_vs_high'][0]:.2f} vs {rr['best_rival_low_vs_high'][1]:.2f}, "
              f"attractiveness {rr['attractiveness_low_vs_high'][0]:+.3f} vs {rr['attractiveness_low_vs_high'][1]:+.3f}; rarely recognised and in a mutual pair {rr['low_in_mutual_pair']:.2f}", flush=True)
        # O, P
        half = {"0 1": [s for s in a.seeds if s < 2], "2 3": [s for s in a.seeds if s >= 2]}
        for k in names:
            if k == "base":
                continue
            d_all = score_of(np.concatenate([got[s]["rank"][k] for s in a.seeds])) - s0
            per = np.bincount(own, weights=d_all, minlength=C) / n_q
            r = {"all": boot(per, rng)}
            for hn, ss in half.items():
                if ss:
                    dd = np.concatenate([score_of(got[s]["rank"][k]) - score_of(got[s]["rank"]["base"]) for s in ss])
                    r[hn] = float(dd.mean())
            res["variants"][k] = r
            keep.setdefault(k, []).append(per)
            print(f"   {k}: {r['all'][0]:+.4f} [{r['all'][1]:+.4f}, {r['all'][2]:+.4f}]  seeds 0 1: {r.get('0 1', float('nan')):+.4f}  seeds 2 3: {r.get('2 3', float('nan')):+.4f}", flush=True)
        out["pools"][pool] = res
    out["both"] = {}
    print("\n===== both pools")
    for k, arrs in keep.items():
        both = boot(np.concatenate(arrs), rng)
        out["both"][k] = both
        print(f"   {k}: {both[0]:+.4f} [{both[1]:+.4f}, {both[2]:+.4f}]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"\nwrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
