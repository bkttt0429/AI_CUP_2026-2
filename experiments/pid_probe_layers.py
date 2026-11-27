#!/usr/bin/env python
"""METHOD_NOTES 2.172 (layer probe): how much identity information does each depth of the humanv0 trunk carry?

Rows: scripts/pid_layer_rows.py (per-game means of the pooled and the played-point vector of the residual stream at several
depths, and of trunkfinal). The instrument is the one of probe A / notes 2.88 (scripts/pid_probe_fullrows.py): for every
representation the 64 supervised directions of its per-game means (between-player against within-player scatter, fitted on
the training sample of folds 3 and 4 only), then
  learner emb   aicup_pid.embed.PlayerEmbedding trained on the players of folds 3 and 4
  learner lda   the within-pool linear discriminant (no training players)
on the simulated pools of folds 1 and 2. Representations:
  <layer>     the 768 means of one depth -> 64 supervised directions   (final = the existing features' source; notes 2.88 had 0.460)
  cat<k>      the means of ALL depths together -> k supervised directions (64, 128)
  mid<k>      the depths other than final together -> k directions
Differences from `final` with a bootstrap over candidates. Nothing is selected here.
Run from the AI_CUP_2026-1 checkout, one GPU.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pid_probe_fullrows as PF  # noqa: E402  (pools, scoring and bootstrap of probe A)

SIZES = PF.SIZES


def supervised(X, inv, n_players, k):
    """[d, k]: the k directions of standardised rows X that separate the players best (generalised eigenvectors of the
    between-player scatter against the within-player scatter; as pid_probe_fullrows.py)."""
    from scipy.linalg import eigh as geigh
    cnt = np.bincount(inv, minlength=n_players).astype(float)
    assert (cnt > 0).all()
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(cnt.astype(int))[:-1]]
    cm = np.add.reduceat(X[order], starts, axis=0) / cnt[:, None]
    resid = X - cm[inv]
    Sw = resid.T @ resid / max(len(X) - n_players, 1)
    Sb = np.cov(cm.T)
    lam, W = geigh(Sb, Sw + 1e-3 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))
    return W[:, ::-1][:, :k], lam[::-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--train-folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--emb-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    assert 0 not in a.folds and 0 not in a.train_folds, "fold 0 is never read here"
    from aicup.ladder import to_json
    from aicup_pid.embed import PlayerEmbedding, score_embed
    from aicup_pid.style import StyleModel
    pools, games = PF.pools_and_games(a.folds, a.seeds)
    rows, layers = {}, None
    for f in sorted(glob.glob(f"{a.rows}/layers_s*_p*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        layers = [str(x) for x in z["layers"]] if layers is None else layers
        for k, r in zip(z["keys"].astype(str), z["rows"]):
            rows[k] = r.astype(np.float32)
    L = len(layers)
    owner_of = dict(zip(games.game_key, games.player_id))
    by = {}
    for k, f in zip(games.game_key, games.fold):
        if f in a.train_folds and k in rows:
            by.setdefault(owner_of[k], []).append(k)
    by = {p: ks for p, ks in by.items() if len(ks) >= 8}
    tr_keys = [k for ks in by.values() for k in ks]
    pl, inv = np.unique(np.array([owner_of[k] for k in tr_keys]), return_inverse=True)
    F_tr = np.stack([rows[k] for k in tr_keys])  # [n, L, 768]
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    print(f"rows of {len(rows)} games, depths {' '.join(layers)}; training sample {len(tr_keys)} games of {len(by)} players (folds {a.train_folds}) ({time.time() - t0:.0f} s)", flush=True)
    std = lambda ks: (np.stack([rows[k] for k in ks]) - mu) / sd  # noqa: E731
    Z_tr = (F_tr - mu) / sd
    reps, ratio = {}, {}
    for li, name in enumerate(layers):
        W, lam = supervised(Z_tr[:, li].astype(np.float64), inv, len(pl), 64)
        ratio[name] = [float(lam[i]) for i in (0, 15, 31, 63)]
        reps[name] = (lambda li_, W_: lambda ks: std(ks)[:, li_] @ W_)(li, W)
        print(f"  {name}: between / within ratio of directions 1 / 16 / 32 / 64: " + " ".join(f"{x:.2f}" for x in ratio[name]) + f" ({time.time() - t0:.0f} s)", flush=True)
    for label, idx in (("cat", list(range(L))), ("mid", [i for i, n in enumerate(layers) if n != "final"])):
        if len(idx) < 2:
            continue
        W, lam = supervised(Z_tr[:, idx].reshape(len(Z_tr), -1).astype(np.float64), inv, len(pl), 128)
        for k in (64, 128):
            reps[f"{label}{k}"] = (lambda idx_, W_: lambda ks: std(ks)[:, idx_].reshape(len(ks), -1) @ W_)(idx, W[:, :k])
        print(f"  {label} ({len(idx)} depths, {len(idx) * F_tr.shape[2]} means): ratio of directions 1 / 16 / 32 / 64 / 128: "
              + " ".join(f"{lam[i]:.2f}" for i in (0, 15, 31, 63, 127)) + f" ({time.time() - t0:.0f} s)", flush=True)
    names = list(reps)
    models = {}
    for name in names:
        rows_by = {p: reps[name](ks).astype(np.float32) for p, ks in by.items()}
        models[name] = [PlayerEmbedding(seed=s, p_aug=0.0).fit(rows_by) for s in a.emb_seeds]
        print(f"  {name}: {rows_by[next(iter(rows_by))].shape[1]} dims, episode top-1 at the end {np.mean([m.history[-1]['acc'] for m in models[name]]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    sc, n_all, cl_all = {}, [], []
    for v, s, cands, questions, owner in pools:
        players = sorted(cands)
        truth = np.array([players.index(p) for p in owner])
        n = np.array([len(q) for q in questions])
        n_all.append(n); cl_all.append(np.array([f"{v}|{s}|{t}" for t in truth]))
        ck = {p: [k for k, _ in cands[p]] for p in players}
        qk = [[k for k, _ in q] for q in questions]
        need = sorted({k for ks in list(ck.values()) + qk for k in ks})
        missing = [k for k in need if k not in rows]
        if missing:
            raise KeyError(f"{len(missing)} pool games without rows, e.g. {missing[:3]}")
        line = []
        for name in names:
            X = dict(zip(need, reps[name](need).astype(np.float32)))
            e = np.mean([PF.per_question(score_embed(m, ck, qk, X), truth) for m in models[name]], axis=0)
            lda = StyleModel(shrink=0.1).fit({p: np.stack([X[k] for k in ck[p]]).astype(np.float64) for p in players})
            l_ = PF.per_question(PF.lda_matrix(lda, np.stack([np.mean([X[k] for k in ks], axis=0) for ks in qk]).astype(np.float64), n.astype(float)), truth)
            sc.setdefault(("emb", name), []).append(e)
            sc.setdefault(("lda", name), []).append(l_)
            line.append(f"{name} {PF.mix(e, n):.4f}/{PF.mix(l_, n):.4f}")
        print(f"fold {v} split {s}: emb/lda  " + "  ".join(line) + f" ({time.time() - t0:.0f} s)", flush=True)
    n_all, cl_all = np.concatenate(n_all), np.concatenate(cl_all)
    rng = np.random.default_rng(0)
    out = {"games": len(rows), "train_games": len(tr_keys), "train_players": len(by), "layers": layers, "ratio": ratio, "scores": {}, "diff_vs_final": {}}
    for learner in ("emb", "lda"):
        base = np.concatenate(sc[(learner, "final")]) if (learner, "final") in sc else None
        for name in names:
            x = np.concatenate(sc[(learner, name)])
            out["scores"][f"{learner}:{name}"] = {"mix": PF.mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}}
            msg = f"{learner} {name:7s} mix {PF.mix(x, n_all):.4f} (n=1 {x[n_all == 1].mean():.4f}, n=2 {x[n_all == 2].mean():.4f}, n=3 {x[n_all == 3].mean():.4f}, n=5 {x[n_all == 5].mean():.4f})"
            if base is not None and name != "final":
                b = PF.boot(x - base, n_all, cl_all, a.boot, rng)
                out["diff_vs_final"][f"{learner}:{name}"] = b
                msg += f"  vs final {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n=1 {b['by_n']['1']:+.4f}"
            print(msg, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
