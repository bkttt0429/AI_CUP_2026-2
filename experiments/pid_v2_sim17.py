#!/usr/bin/env python
"""Bayesian set scoring of the sequence vectors (PLDA) against the prototype cosine (METHOD_NOTES 2.44).
The sequence component reduces a candidate's ten games to one prototype and scores a question by the cosine of its summed
vector: the candidate's own spread, the directions in which one player's games vary, and the different certainty of 1-, 2-,
3- and 5-game questions are not modelled. PLDA (two-covariance model, the standard of speaker verification) does: a game
vector = player mean + within-player noise, player mean ~ N(mu, B), noise ~ N(0, W). After a transform with W = I and
B = diag(psi), the log-likelihood ratio "same player" against "another player" of a question (n_q games, mean xq) and a
candidate (n_c games, mean xc) is, per dimension,
    log N(xq; n_c psi / (n_c psi + 1) xc, psi / (n_c psi + 1) + 1 / n_q) - log N(xq; 0, psi + 1 / n_q).
Parameters are fitted per network on the POOL'S OWN CANDIDATE GAMES (their owners are given; no question game is used):
centre, whiten by the total covariance, length-normalise, moment estimates of W and B.
Pools, networks and reference as pid_v2_sim11.py's arm ft50; vectors before the fine-tune (t0) and after --steps steps.
Arms (weights cross-fitted on the other folds' pools; the sequence component is the mean of the networks' row-standardised
matrices in every arm):
  cos@0 / plda@0      cosine / PLDA on the vectors before the fine-tune
  cos@t (ref) / plda@t   the same after the fine-tune (candidate games were trained on: W is then estimated on seen games)
  cos@t+plda@0, cos@t+plda@t   PLDA as one more component next to the reference
Rule (fixed before the run): an arm replaces the reference when its gain has a player-bootstrap CI above 0 and at least
6 of the 8 pools gain.
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

unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731


def plda_fit(X, y):
    """Two-covariance PLDA from labelled vectors X [N, D], y [N] class index. Returns (transform f, psi [D]): in f's space the
    within-class covariance is the identity and the between-class covariance diag(psi)."""
    mu = X.mean(0)
    ev, U = np.linalg.eigh(np.cov((X - mu).T) + 1e-8 * np.eye(X.shape[1]))
    P1 = U / np.sqrt(np.maximum(ev, 1e-3 * ev.mean()))  # floor: unit vectors leave some directions almost empty
    Z = unit((X - mu) @ P1)
    mu2 = Z.mean(0)
    Z = Z - mu2
    C = int(y.max()) + 1
    cnt = np.bincount(y, minlength=C).astype(float)
    means = np.zeros((C, Z.shape[1]))
    np.add.at(means, y, Z)
    means /= cnt[:, None]
    R = Z - means[y]
    Sw = R.T @ R / (len(Z) - C)
    Sb = (means * cnt[:, None]).T @ means / cnt.sum() - Sw * (C / cnt.sum())  # covariance of the class means minus the noise in them
    ew, Uw = np.linalg.eigh(Sw)
    A = Uw / np.sqrt(np.maximum(ew, 1e-3 * ew.mean()))
    psi, V = np.linalg.eigh(A.T @ Sb @ A)
    psi = np.maximum(psi, 0.0)
    f = lambda Xn: ((unit((Xn - mu) @ P1) - mu2) @ A) @ V  # noqa: E731
    return f, psi


def plda_scores(f, psi, cand_games, q_games):
    """cand_games / q_games: lists of [n, D] arrays. Log-likelihood ratio [Q, C]."""
    xc = np.stack([f(g).mean(0) for g in cand_games])
    nc = np.array([len(g) for g in cand_games], float)[:, None]
    m = nc * psi / (nc * psi + 1.0) * xc  # [C, D] posterior mean of the candidate
    v = psi / (nc * psi + 1.0)  # [C, D] posterior variance
    out = np.zeros((len(q_games), len(cand_games)))
    nq = np.array([len(g) for g in q_games])
    xq = np.stack([f(g).mean(0) for g in q_games])
    for n in np.unique(nq):
        rows = np.flatnonzero(nq == n)
        vs, vd = v + 1.0 / n, psi + 1.0 / n  # [C, D], [D]
        for i in rows:  # [C, D] per question keeps the memory small
            d = xq[i][None] - m
            out[i] = (-0.5 * np.log(vs / vd) - 0.5 * d * d / vs + 0.5 * xq[i][None] ** 2 / vd).sum(1)
    return out


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
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float64))))(np.load(f))  # noqa: E731

    def vec_file(v_, s, name, t):
        if t == 0:
            return ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz"
        if name == "m88":
            return Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if t == 50 else f"sim_f{v_}_s{s}_t100.npz")
        return Path(a.ft_dir) / f"sim_f{v_}_s{s}{nets[name][1]}_t{t}.npz"

    T = a.steps
    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            players = sorted(cands)
            y = np.array([i for i, p in enumerate(players) for _ in cands[p]])
            Z = {c: standardise(x) for c, x in S.items()}
            psi_log = []
            for t in (0, T):
                cos, pl = [], []
                for name in nets:
                    vec = vecs(vec_file(v_, s, name, t))
                    cg = [np.stack([vec[k] for k, _ in cands[p]]) for p in players]
                    qg = [np.stack([vec[k] for k, _ in q]) for q in questions]
                    cos.append(standardise(unit(np.stack([x.sum(0) for x in qg])) @ unit(np.stack([x.sum(0) for x in cg])).T))
                    f, psi = plda_fit(np.concatenate(cg), y)
                    pl.append(standardise(plda_scores(f, psi, cg, qg)))
                    psi_log.append(np.sort(psi)[::-1][[0, 9, 31, 63, 127]].round(3).tolist())
                Z[f"cos@{t}"], Z[f"plda@{t}"] = standardise(np.mean(cos, axis=0)), standardise(np.mean(pl, axis=0))
            runs.append({"Z": Z, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_})
            r = runs[-1]
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            byn = lambda M: {int(m): round(float(score(topk(M, 5), r["truth"])[r["n"] == m].mean()), 3) for m in SIZES}  # noqa: E731
            print(f"pool fold {v_} seed {s}: sequence component alone  cos@0 {al(Z['cos@0']):.3f}  plda@0 {al(Z['plda@0']):.3f}  cos@{T} {al(Z[f'cos@{T}']):.3f}  plda@{T} {al(Z[f'plda@{T}']):.3f}; "
                  f"by n cos@{T} {byn(Z[f'cos@{T}'])} plda@{T} {byn(Z[f'plda@{T}'])}; psi (1st, 10th, 32nd, 64th, 128th) first net t0 {psi_log[0]} t{T} {psi_log[len(nets)]} ({time.time() - t0:.0f} s)", flush=True)
    nn_ = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    arms = {"ref": (*BASE, f"cos@{T}"), f"plda@{T}": (*BASE, f"plda@{T}"), "cos@0": (*BASE, "cos@0"), "plda@0": (*BASE, "plda@0"),
            f"cos@{T}+plda@0": (*BASE, f"cos@{T}", "plda@0"), f"cos@{T}+plda@{T}": (*BASE, f"cos@{T}", f"plda@{T}")}
    out, per_q, per_pool = {"nets": {k: list(x) for k, x in nets.items()}, "steps": T, "arms": {}, "comparisons": {}}, {}, {}
    for arm, comps in arms.items():
        sc, alone = [], []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack([o["Z"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores([r["Z"][c] for c in comps]), 5), r["truth"]))
            alone.append(float(score(topk(r["Z"][comps[-1]], 5), r["truth"]).mean()))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        fus = Fusion().fit([np.vstack([r["Z"][c] for r in runs]) for c in comps], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "last_component_alone": float(np.mean(alone)), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn_ == m].mean()) for m in SIZES},
                            "weights_all_pools": dict(zip(comps, fus.w.tolist()))}
        print(f"arm {arm}: fused {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  weights {({c: round(w, 2) for c, w in zip(comps, fus.w.tolist()) if c not in BASE or c == 'emb'})}", flush=True)
    for arm, base in ((f"plda@{T}", "ref"), ("plda@0", "cos@0"), (f"cos@{T}+plda@0", "ref"), (f"cos@{T}+plda@{T}", "ref")):
        d = paired_player_bootstrap(per_q[arm], per_q[base], pid, n=2000)
        d["pools_positive"] = int(sum(x > y_ for x, y_ in zip(per_pool[arm], per_pool[base])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 0.75 * len(runs))
        d["by_n"] = {str(m): float(per_q[arm][nn_ == m].mean() - per_q[base][nn_ == m].mean()) for m in SIZES}
        out["comparisons"][f"{arm} vs {base}"] = d
        print(f"{arm} vs {base}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)} -> {'PASS' if d['pass'] else 'no'}; by n {({k: round(x, 4) for k, x in d['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
