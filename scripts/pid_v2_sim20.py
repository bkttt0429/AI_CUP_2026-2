#!/usr/bin/env python
"""A strength ("same rank class") component from a probe on the sequence vectors (METHOD_NOTES 2.49).
The error analysis shows that a third of the wrong first places go to a candidate two or more rank classes away from the
owner, and that a perfect filter "candidates within one class of the owner" would add +0.026 on the added pools. This tests
how much of that a REAL estimate recovers, with no new input: a 10-class linear probe on the sequence networks' game vectors.
  probe     per simulated pool of fold v: multinomial logistic regression on the game vectors (all networks of the fold
            concatenated, before the pool fine-tune) of fold v's players that are NOT in the pool; these players are as
            unseen by the networks as the pool's players, and their rank class is the training label of the rank competition
  posterior a set of games -> sum of the per-game log-probabilities (plus the prior once), softmax
  same      log sum_r P_question(r) * P_candidate(r) / prior(r)          (both have the same class)
  near      the same with the candidate's posterior spread to the adjacent classes (kernel 0.5 / 1 / 0.5)
  gap       - | E_question[r] - E_candidate[r] |                           (expected class apart)
Arms: reference (multi-network ft50) + one of the three, + all; weights cross-fitted on the other folds' pools.
Rule (fixed before the run): an arm is adopted when its gain has a player-bootstrap CI above 0 and at least 6 of 8 pools gain.
"""
import argparse
import json
import sys
import time
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
from aicup.metric import LEVELS  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
K = len(LEVELS)


def fit_probe(X, y, l2=1e-3, steps=400):
    """Multinomial logistic regression (standardised inputs) on the GPU; returns f(X) -> log-probabilities [n, K]."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    W = torch.zeros(X.shape[1], K, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    Wn, bn = W.detach().cpu().numpy(), b.detach().cpu().numpy()

    def f(Xn):
        L = ((Xn - mu) / sd) @ Wn + bn
        L = L - L.max(1, keepdims=True)
        return L - np.log(np.exp(L).sum(1, keepdims=True))
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    lvl = {r: i for i, r in enumerate(LEVELS)}
    rank_of = {p: lvl[r] for p, r in zip(games.player_id, games["rank"])}
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    ft = lambda v_, s, name: Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if name == "m88" else f"sim_f{v_}_s{s}{nets[name][1]}_t50.npz")  # noqa: E731
    runs, acc = [], []
    for v_ in a.folds:
        raw = [vecs(ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz") for name in nets]  # before the fine-tune, every network of the fold
        seq0 = raw[0]
        cat = lambda keys: np.hstack([np.stack([r[k] for k in keys]) for r in raw]).astype(np.float64)  # noqa: E731
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        allv = games[(games.fold == v_) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            Z = {c: standardise(x) for c, x in S.items()}
            Z["seq"] = standardise(np.mean([standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(ft(v_, s, name)))["seq"]) for name in nets], axis=0))
            players = sorted(cands)
            tr = allv[~allv.player_id.isin(set(players))]
            f = fit_probe(cat(tr.game_key.tolist()), np.array([rank_of[p] for p in tr.player_id]))
            prior = np.log(np.bincount([rank_of[p] for p in tr.player_id.unique()], minlength=K) + 1.0)
            prior -= np.log(np.exp(prior).sum())
            post = lambda gs: (lambda L: np.exp(L - L.max()) / np.exp(L - L.max()).sum())(f(cat([k for k, _ in gs])).sum(0) + prior)  # noqa: E731
            Pc = np.stack([post(cands[p]) for p in players])
            Pq = np.stack([post(q) for q in questions])
            ker = np.eye(K) + 0.5 * (np.eye(K, k=1) + np.eye(K, k=-1))
            ex = np.arange(K)
            Z["same"] = standardise(np.log((Pq / np.exp(prior)) @ Pc.T + 1e-12))
            Z["near"] = standardise(np.log((Pq / np.exp(prior)) @ (Pc @ ker).T + 1e-12))
            Z["gap"] = standardise(-np.abs((Pq @ ex)[:, None] - (Pc @ ex)[None]))
            truth = np.array([players.index(p) for p in owner])
            crank = np.array([rank_of[p] for p in players])
            n = np.array([len(q) for q in questions])
            acc.append({"train_players": int(tr.player_id.nunique()), "train_games": int(len(tr)), "cand_exact": float((Pc.argmax(1) == crank).mean()), "cand_within1": float((np.abs(Pc.argmax(1) - crank) <= 1).mean()),
                        **{f"q{m}_exact": float((Pq.argmax(1) == crank[truth])[n == m].mean()) for m in SIZES}, **{f"q{m}_within1": float((np.abs(Pq.argmax(1) - crank[truth]) <= 1)[n == m].mean()) for m in SIZES}})
            runs.append({"Z": Z, "truth": truth, "n": n, "owner": owner, "fold": v_})
            al = lambda M: float(score(topk(M, 5), truth).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: probe on {acc[-1]['train_games']} games of {acc[-1]['train_players']} players; class of a candidate (10 games) exact {acc[-1]['cand_exact']:.3f} / within one {acc[-1]['cand_within1']:.3f}; "
                  f"of a question by size exact " + " ".join(f"{m}:{acc[-1][f'q{m}_exact']:.2f}" for m in SIZES) + " within one " + " ".join(f"{m}:{acc[-1][f'q{m}_within1']:.2f}" for m in SIZES)
                  + f"; alone same {al(Z['same']):.3f} near {al(Z['near']):.3f} gap {al(Z['gap']):.3f} ({time.time() - t0:.0f} s)", flush=True)
    nn_ = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    arms = {"ref": COMPS, "+same": (*COMPS, "same"), "+near": (*COMPS, "near"), "+gap": (*COMPS, "gap"), "+all": (*COMPS, "same", "near", "gap")}
    out, per_q, per_pool = {"probe": acc, "arms": {}, "comparisons": {}}, {}, {}
    for arm, comps in arms.items():
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack([o["Z"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores([r["Z"][c] for c in comps]), 5), r["truth"]))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        fus = Fusion().fit([np.vstack([r["Z"][c] for r in runs]) for c in comps], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn_ == m].mean()) for m in SIZES}, "weights_all_pools": dict(zip(comps, fus.w.tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  weights {({c: round(w, 2) for c, w in zip(comps, fus.w.tolist()) if c in ('seq', 'emb', 'same', 'near', 'gap')})}", flush=True)
    for arm in arms:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 0.75 * len(runs))
        d["by_n"] = {str(m): float(per_q[arm][nn_ == m].mean() - per_q["ref"][nn_ == m].mean()) for m in SIZES}
        out["comparisons"][f"{arm} vs ref"] = d
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)} -> {'PASS' if d['pass'] else 'no'}; by n {({k: round(x, 4) for k, x in d['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
