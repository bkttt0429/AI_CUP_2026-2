#!/usr/bin/env python
"""Room for a second-stage reranker on the v2 added pools (METHOD_NOTES 2.43): what the linear fusion throws away.
The fusion adds nine [question x candidate] matrices with one global weight vector; before that, the 20 networks are averaged,
the games of a question are summed and a candidate's ten games become one prototype. This diagnostic (no training beyond the
usual cross-fitted fusion weights) measures, on the simulated pools of pid_v2_sim11.py's chosen arm (all networks after 50
fine-tune steps, emb2):
  1. where the owner stands in the fused ranking, by question size: share at rank 1, within 5, 10, 30, 100
     (the score counts the first five; a reranker of the top K can only use owners inside K)
  2. whether the discarded detail carries signal when added back as ONE more linear component:
       nn      mean over the question's games of the best cosine to any single game of the candidate (not the prototype)
       top3    the same with the mean of the three best games
       gmin    the weakest of the question's games against the candidate's prototype (all games must agree; = seq for n = 1)
       agree   minus the spread of the networks' standardised scores for the pair (the networks agree)
       hub     the fused sequence score minus the candidate's mean over the pool's questions (candidates chosen by many)
     and all five together
  3. whether weights fitted per question size beat the global weights
Each against the reference with a paired player bootstrap; pools up out of 8.
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
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
EXTRA = ("nn", "top3", "gmin", "agree", "hub")
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731


def detail(cands, questions, vec):
    """One network: prototype matrix (as aicup_pid.v2.components) and the game-level matrices nn / top3 / gmin, each [Q, C]."""
    players = sorted(cands)
    Ec = [unit(np.stack([vec[k] for k, _ in cands[p]]).astype(np.float64)) for p in players]
    Pc = unit(np.stack([np.sum([vec[k] for k, _ in cands[p]], axis=0) for p in players]).astype(np.float64))
    proto, nn, top3, gmin = [], [], [], []
    for q in questions:
        G = np.stack([vec[k] for k, _ in q]).astype(np.float64)
        Gu = unit(G)
        proto.append(unit(G.sum(0)) @ Pc.T)
        best = np.stack([np.sort(Gu @ e.T, axis=1)[:, ::-1][:, :3] for e in Ec], axis=1)  # [games, C, 3]
        nn.append(best[:, :, 0].mean(0))
        top3.append(best.mean(2).mean(0))
        gmin.append((Gu @ Pc.T).min(0))
    return {k: np.stack(x) for k, x in (("proto", proto), ("nn", nn), ("top3", top3), ("gmin", gmin))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--steps", type=int, default=50)
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
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731

    def ft_file(v_, s, name):
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
            D = [detail(cands, questions, vecs(ft_file(v_, s, name))) for name in nets]
            P = np.stack([standardise(d["proto"]) for d in D])  # [nets, Q, C]
            Z = {c: standardise(x) for c, x in S.items()}
            Z["seq"] = standardise(P.mean(0))
            for k in ("nn", "top3", "gmin"):
                Z[k] = standardise(np.mean([standardise(d[k]) for d in D], axis=0))
            Z["agree"] = standardise(-P.std(0))
            Z["hub"] = standardise(Z["seq"] - Z["seq"].mean(0, keepdims=True))
            players = sorted(cands)
            runs.append({"Z": Z, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_})
            al = lambda M: float(score(topk(M, 5), runs[-1]["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: alone seq {al(Z['seq']):.3f}  nn {al(Z['nn']):.3f}  top3 {al(Z['top3']):.3f}  gmin {al(Z['gmin']):.3f}  hub {al(Z['hub']):.3f} ({time.time() - t0:.0f} s)", flush=True)
    nn_ = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out = {"nets": {k: list(x) for k, x in nets.items()}, "steps": a.steps, "arms": {}, "comparisons": {}, "rank_of_owner": {}}

    def cross(comps, by_n=False):
        """Cross-fitted fused matrices of every pool (weights from the other folds' pools; by_n: one weight vector per question size)."""
        F = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            if not by_n:
                fus = Fusion().fit([np.vstack([o["Z"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
                F.append(fus.scores([r["Z"][c] for c in comps]))
            else:
                M = np.zeros_like(r["Z"]["seq"])
                for m in SIZES:
                    fus = Fusion().fit([np.vstack([o["Z"][c][o["n"] == m] for o in others]) for c in comps], np.concatenate([o["truth"][o["n"] == m] for o in others]))
                    M[r["n"] == m] = fus.scores([r["Z"][c][r["n"] == m] for c in comps])
                F.append(M)
        return F

    arms = {"ref": (COMPS, False), **{f"+{k}": ((*COMPS, k), False) for k in EXTRA}, "+all": ((*COMPS, *EXTRA), False), "by_n": (COMPS, True), "by_n+all": ((*COMPS, *EXTRA), True)}
    per_q, per_pool = {}, {}
    for arm, (comps, by_n) in arms.items():
        F = cross(comps, by_n)
        sc = [score(topk(f, 5), r["truth"]) for f, r in zip(F, runs)]
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn_ == m].mean()) for m in SIZES}}
        if not by_n:
            fus = Fusion().fit([np.vstack([r["Z"][c] for r in runs]) for c in comps], np.concatenate([r["truth"] for r in runs]))
            out["arms"][arm]["weights_all_pools"] = dict(zip(comps, fus.w.tolist()))
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}"
              + ("" if by_n else f"  weights {({c: round(w, 2) for c, w in out['arms'][arm]['weights_all_pools'].items() if c in ('seq', 'emb', *EXTRA)})}"), flush=True)
        if arm == "ref":
            rank = np.concatenate([(f > f[np.arange(len(r["truth"])), r["truth"]][:, None]).sum(1) for f, r in zip(F, runs)])
            for m in (*SIZES, "all"):
                sel = np.ones(len(rank), bool) if m == "all" else nn_ == m
                out["rank_of_owner"][str(m)] = {f"within_{k}": float((rank[sel] < k).mean()) for k in (1, 5, 10, 30, 100)}
                print(f"   owner's rank, n={m}: " + "  ".join(f"top-{k} {out['rank_of_owner'][str(m)][f'within_{k}']:.3f}" for k in (1, 5, 10, 30, 100)), flush=True)
    for arm in arms:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["by_n"] = {str(m): float(per_q[arm][nn_ == m].mean() - per_q["ref"][nn_ == m].mean()) for m in SIZES}
        out["comparisons"][f"{arm} vs ref"] = d
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)}; by n {({k: round(x, 4) for k, x in d['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
