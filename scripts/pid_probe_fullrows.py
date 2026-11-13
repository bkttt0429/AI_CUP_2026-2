#!/usr/bin/env python
"""What does the random projection of the trunk vectors lose? (METHOD_NOTES 2.69, probe A: per-game rows)

Every stored feature of this project is the humanv0 trunk's 384-channel vector pushed through a fixed random projection to
64 dimensions (board-pooled and played-point). scripts/pid_trunk_extract.py --full stores, for a sample of games, the per-game
mean / std of the UNPROJECTED vectors next to the usual 256-dim row. The same learner is trained on each representation
  rp256     the usual row (mean / std of the two projected vectors)
  full      mean / std of the two unprojected vectors (1,536)
  pca256, pca512   principal components of `full` (fitted on the training sample)
  rp128m    the two projected means only            fullm    the two unprojected means only (768)
  rpm128, rpm256   the unprojected means through a fresh random projection to 128 / 256 dims each (the curve by dimension)
and scored on the simulated pools of folds 1 and 2 (scripts/pid_v2_sim2.py: 600 candidates x 10 games, both colours):
  learner emb   aicup_pid.embed.PlayerEmbedding trained on players of folds 3 and 4 (the emb component's episode loss)
  learner lda   the within-pool linear discriminant (aicup_pid.style.StyleModel), no training players
Reported: the representation's own score by question size, differences from rp256 with a bootstrap over candidates.

  --stage keys   writes the list of games to extract (the simulated pools of folds 1 and 2, up to --per-player games of every
                 player of folds 3 and 4); fold 0 is never read
  --stage eval   the comparison (one GPU)
Run from the AI_CUP_2026-1 checkout.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
SIZES = (1, 2, 3, 5)
C = 384


def pools_and_games(folds, seeds):
    """The simulated pools exactly as scripts/pid_v2_where.py builds them, and the table of training games with folds."""
    import aicup_pid
    from aicup_pid import v2
    from aicup_pid.paths import FOLDS_CSV
    from pid_v2_sim2 import build
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    out = []
    for v in folds:
        z = np.load(R2 / "data" / "seq" / f"fold{v}.npz")
        g = games[(games.fold == v) & games.game_key.isin(rows.keys()) & games.game_key.isin(set(z["key"].astype(str)))]
        for s in seeds:
            cands, questions, owner, _block, _sgfs, _info = build(g, s)
            out.append((v, s, cands, questions, owner))
    return out, games


def per_question(F, truth):
    rank = (F > F[np.arange(len(truth)), truth][:, None]).sum(1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


def mix(x, n):
    return float(np.mean([x[n == m].mean() for m in SIZES]))


def boot(d, n, cluster, B, rng):
    ids, inv = np.unique(cluster, return_inverse=True)
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])
    pick = rng.integers(0, len(ids), (B, len(ids)))
    bs = (num[:, pick].sum(2) / den[:, pick].sum(2)).mean(0)
    return {"diff": float((num.sum(1) / den.sum(1)).mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "by_n": {str(m): float(d[n == m].mean()) for m in SIZES}}


def lda_matrix(lda, q_means, q_sizes):
    """[Q, C] scores of aicup_pid.style.StyleModel.log_scores for many questions at once (whitened, one matrix product)."""
    L = np.linalg.cholesky(lda.prec)
    Mw, Qw = lda.means @ L, q_means @ L
    d2 = (Qw ** 2).sum(1)[:, None] + (Mw ** 2).sum(1)[None] - 2 * Qw @ Mw.T
    return -0.5 * d2 / (1.0 / q_sizes[:, None] + 1.0 / lda.n_games[None])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("keys", "eval"))
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--train-folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--per-player", type=int, default=16)
    ap.add_argument("--rows", default="/work/u2507115/pid_v2/fullrows")
    ap.add_argument("--emb-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    assert 0 not in a.folds and 0 not in a.train_folds, "fold 0 is never read here"
    pools, games = pools_and_games(a.folds, a.seeds)
    if a.stage == "keys":
        keys = {k for _v, _s, cands, questions, _o in pools for ks in list(cands.values()) + questions for k, _ in ks}
        n_pool = len(keys)
        rng = np.random.RandomState(0)
        tr = games[games.fold.isin(a.train_folds)]
        for _p, ks in tr.groupby("player_id")["game_key"].apply(sorted).items():
            if len(ks) >= 8:
                keys.update(rng.permutation(ks)[:a.per_player].tolist())
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(sorted(keys)) + "\n")
        print(f"{len(keys)} games: {n_pool} of the simulated pools, {len(keys) - n_pool} of the training players of folds {a.train_folds} ({time.time() - t0:.0f} s)")
        return 0

    import torch
    from aicup.ladder import to_json
    from aicup_pid.embed import PlayerEmbedding, score_embed
    from aicup_pid.style import StyleModel
    rp, full = {}, {}
    for f in sorted(glob.glob(f"{a.rows}/*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        for k, r, u in zip(z["keys"].astype(str), z["feats"], z["full"]):
            rp[k], full[k] = r.astype(np.float32), u.astype(np.float32)
    owner_of = dict(zip(games.game_key, games.player_id))
    tr_keys = sorted(k for k, f in zip(games.game_key, games.fold) if f in a.train_folds and k in full)
    by = {}
    for k in tr_keys:
        by.setdefault(owner_of[k], []).append(k)
    by = {p: ks for p, ks in by.items() if len(ks) >= 8}
    tr_keys = [k for ks in by.values() for k in ks]
    print(f"rows of {len(full)} games; training sample {len(tr_keys)} games of {len(by)} players (folds {a.train_folds}) ({time.time() - t0:.0f} s)", flush=True)

    # the representations: name -> function of a key list -> [n, d]
    F_tr = np.stack([full[k] for k in tr_keys])
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    w, V = np.linalg.eigh(np.cov(((F_tr - mu) / sd).T))
    V = V[:, ::-1]
    evr = (w[::-1] / w.sum()).cumsum()
    print("principal components of the standardised unprojected rows: variance explained by the first 64 / 128 / 256 / 512 = "
          + " / ".join(f"{evr[k - 1]:.3f}" for k in (64, 128, 256, 512)), flush=True)
    m_idx = np.r_[0:C, 2 * C:3 * C]  # the two means
    gen = np.random.default_rng(1)
    R = {k: (gen.standard_normal((C, k)) / C ** 0.5, gen.standard_normal((C, k)) / C ** 0.5) for k in (128, 256)}
    stack = lambda d, ks: np.stack([d[k] for k in ks])  # noqa: E731
    reps = {
        "rp256": lambda ks: stack(rp, ks),
        "full": lambda ks: stack(full, ks),
        "pca256": lambda ks: ((stack(full, ks) - mu) / sd) @ V[:, :256],
        "pca512": lambda ks: ((stack(full, ks) - mu) / sd) @ V[:, :512],
        "rp128m": lambda ks: stack(rp, ks)[:, np.r_[0:64, 128:192]],
        "fullm": lambda ks: stack(full, ks)[:, m_idx],
        "rpm128": lambda ks: (lambda X: np.hstack([X[:, :C] @ R[128][0], X[:, 2 * C:3 * C] @ R[128][1]]))(stack(full, ks)),
        "rpm256": lambda ks: (lambda X: np.hstack([X[:, :C] @ R[256][0], X[:, 2 * C:3 * C] @ R[256][1]]))(stack(full, ks)),
    }
    names = list(reps)
    sc = {}  # (learner, representation) -> per-question scores over all pools
    n_all, cl_all = [], []
    models = {}
    for name in names:
        rows_by = {p: reps[name](ks).astype(np.float32) for p, ks in by.items()}
        models[name] = [PlayerEmbedding(seed=s, p_aug=0.0).fit(rows_by) for s in a.emb_seeds]
        print(f"  {name}: {rows_by[next(iter(rows_by))].shape[1]} dims, episode top-1 at the end {np.mean([m.history[-1]['acc'] for m in models[name]]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    for v, s, cands, questions, owner in pools:
        players = sorted(cands)
        truth = np.array([players.index(p) for p in owner])
        n = np.array([len(q) for q in questions])
        n_all.append(n); cl_all.append(np.array([f"{v}|{s}|{t}" for t in truth]))
        ck = {p: [k for k, _ in cands[p]] for p in players}
        qk = [[k for k, _ in q] for q in questions]
        need = sorted({k for ks in list(ck.values()) + qk for k in ks})
        missing = [k for k in need if k not in full]
        if missing:
            raise KeyError(f"{len(missing)} pool games without rows, e.g. {missing[:3]}")
        line = []
        for name in names:
            X = dict(zip(need, reps[name](need).astype(np.float32)))
            e = np.mean([per_question(score_embed(m, ck, qk, X), truth) for m in models[name]], axis=0)
            lda = StyleModel(shrink=0.1).fit({p: np.stack([X[k] for k in ck[p]]).astype(np.float64) for p in players})
            l_ = per_question(lda_matrix(lda, np.stack([np.mean([X[k] for k in ks], axis=0) for ks in qk]).astype(np.float64), n.astype(float)), truth)
            sc.setdefault(("emb", name), []).append(e)
            sc.setdefault(("lda", name), []).append(l_)
            line.append(f"{name} {mix(e, n):.4f}/{mix(l_, n):.4f}")
        print(f"fold {v} split {s}: emb/lda  " + "  ".join(line) + f" ({time.time() - t0:.0f} s)", flush=True)
    n_all, cl_all = np.concatenate(n_all), np.concatenate(cl_all)
    rng = np.random.default_rng(0)
    out = {"games": len(full), "train_games": len(tr_keys), "train_players": len(by), "pca_variance": {str(k): float(evr[k - 1]) for k in (64, 128, 256, 512)}, "scores": {}, "diff_vs_rp256": {}}
    for learner in ("emb", "lda"):
        base = np.concatenate(sc[(learner, "rp256")])
        for name in names:
            x = np.concatenate(sc[(learner, name)])
            out["scores"][f"{learner}:{name}"] = {"mix": mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}}
            msg = f"{learner} {name:8s} mix {mix(x, n_all):.4f} (n=1 {x[n_all == 1].mean():.4f}, n=2 {x[n_all == 2].mean():.4f}, n=3 {x[n_all == 3].mean():.4f}, n=5 {x[n_all == 5].mean():.4f})"
            if name != "rp256":
                b = boot(x - base, n_all, cl_all, a.boot, rng)
                out["diff_vs_rp256"][f"{learner}:{name}"] = b
                msg += f"  vs rp256 {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n=1 {b['by_n']['1']:+.4f}"
            print(msg, flush=True)
    b = boot(np.concatenate(sc[("emb", "fullm")]) - np.concatenate(sc[("emb", "rp128m")]), n_all, cl_all, a.boot, rng)
    out["diff_fullm_vs_rp128m"] = b
    print(f"emb fullm vs rp128m (means only, unprojected against projected): {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n=1 {b['by_n']['1']:+.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
