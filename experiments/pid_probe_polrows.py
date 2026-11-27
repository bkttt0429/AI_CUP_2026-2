#!/usr/bin/env python
"""Probe P, row level (METHOD_NOTES 2.102): do per-game statistics RELATIVE TO THE NETWORK'S OWN POLICY identify players better
than the statistics of the position and of the played point alone?

Rows of scripts/pid_pol_extract.py. Blocks of a game's row (C = 384 each unless noted):
  Gm, Gs  mean / std of the board-pooled trunk vector          Xm, Xs  mean / std of the played-point vector
  Em, Es  mean / std of the EXPECTED played-point vector (policy-weighted mean over the board points)
  Ds      std of (played - expected)                           S       36 statistics of surprisal / rank / entropy
Every representation is standardised on the training sample (players of folds 3 and 4, never fold 0) and reduced to k
dimensions by the supervised projection fitted there (between-player against within-player scatter of the per-game rows):
  m2_64   [Gm, Xm]            the information the existing supervised per-ply features were built from = the reference
  m3_64   [Gm, Xm, Em]        the expected vector added          md_64  [Gm, Xm - Em]   the difference in place of Xm
  m2_128, m3_128              the same with 128 dimensions
  f2_128  [Gm, Gs, Xm, Xs]    f3_128  [Gm, Gs, Xm, Xs, Em, Es, Ds]
  s36     the 36 statistics alone (standardised, no projection)  m3s_64 [Gm, Xm, Em, S]
Learners and scoring as scripts/pid_probe_fullrows.py on the simulated pools of folds 1 and 2: emb = PlayerEmbedding trained on
the training sample, lda = the within-pool linear discriminant. Paired differences against m2_64, bootstrap over candidates.
Also checks that the `z` of the first 64 dimensions of <rows>/../ply_fc equals the stored features of --old-ply (the new
extraction path reproduces the old one). Run from the AI_CUP_2026-1 checkout (one GPU).
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
from pid_probe_fullrows import SIZES, boot, lda_matrix, mix, per_question, pools_and_games  # noqa: E402

C = 384


def sup_proj(X, owners, k, ridge=1e-3):
    """X [n, d] standardised rows of the training sample, owners [n] -> (W [d, k], between / within ratios [k])."""
    from scipy.linalg import eigh
    from scipy.sparse import csr_matrix
    _pl, inv = np.unique(owners, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    cm = np.asarray(csr_matrix((np.ones(len(X)), (inv, np.arange(len(X)))), shape=(len(cnt), len(X))) @ X) / cnt[:, None]
    resid = X - cm[inv]
    Sw = resid.T @ resid / max(len(X) - len(cnt), 1)
    lam, W = eigh(np.cov(cm.T), Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))
    return W[:, ::-1][:, :k], lam[::-1][:k]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="/work/u2507115/pid_v2/probep/rows")
    ap.add_argument("--old-ply", default="/work/u2507115/pid_v2/probeb/ply")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--train-folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--emb-seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    assert 0 not in a.folds and 0 not in a.train_folds, "fold 0 is never read here"
    from aicup.ladder import to_json
    from aicup_pid.embed import PlayerEmbedding, score_embed
    from aicup_pid.style import StyleModel

    # 0. the new extraction path against the stored features of the same games
    chk = {}
    new_parts = sorted(glob.glob(f"{Path(a.rows).parent}/ply_fc/*.npz"))[:3]
    if new_parts and a.old_ply:
        want = {}
        for f in new_parts:
            z = np.load(f)
            st = np.concatenate([[0], np.cumsum(z["n"])])
            for i, k in enumerate(z["keys"].astype(str)):
                want[k] = (z["t"][st[i]:st[i + 1]], z["z"][st[i]:st[i + 1], :64].astype(np.float32))
        diffs, cors, found = [], [], 0
        for f in sorted(glob.glob(f"{a.old_ply}/*.npz")):
            z = np.load(f)
            ks = z["keys"].astype(str)
            if not any(k in want for k in ks):
                continue
            st = np.concatenate([[0], np.cumsum(z["n"])])
            for i, k in enumerate(ks):
                if k in want and len(want[k][0]):
                    t_new, z_new = want[k]
                    z_old = z["z"][st[i]:st[i + 1]].astype(np.float32)
                    if len(z_old) == len(z_new) and (z["t"][st[i]:st[i + 1]] == t_new).all():
                        found += 1
                        diffs.append(float(np.abs(z_old - z_new).max()))
                        cors.append(float(np.corrcoef(z_old.ravel(), z_new.ravel())[0, 1]))
            if found >= 400:
                break
        chk = {"games": found, "max_abs_diff": float(np.max(diffs)) if diffs else None, "min_corr": float(np.min(cors)) if cors else None}
        print(f"new path against the stored supervised features: {found} games, largest absolute difference {chk['max_abs_diff']}, smallest correlation {chk['min_corr']} ({time.time() - t0:.0f} s)", flush=True)

    pools, games = pools_and_games(a.folds, a.seeds)
    keys, blocks = [], {"full": [], "pole": [], "pold": [], "pols": []}
    for f in sorted(glob.glob(f"{a.rows}/*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        keys.append(z["keys"].astype(str))
        for b in blocks:
            blocks[b].append(z[b].astype(np.float32))
    keys = np.concatenate(keys)
    blocks = {b: np.concatenate(v) for b, v in blocks.items()}
    row_of = {k: i for i, k in enumerate(keys.tolist())}
    full, pole = blocks["full"], blocks["pole"]
    raw = {"Gm": full[:, 0:C], "Gs": full[:, C:2 * C], "Xm": full[:, 2 * C:3 * C], "Xs": full[:, 3 * C:4 * C], "Em": pole[:, :C], "Es": pole[:, C:],
           "Ds": blocks["pold"], "S": blocks["pols"]}
    raw["Dm"] = raw["Xm"] - raw["Em"]
    owner_of, fold_of = dict(zip(games.game_key, games.player_id)), dict(zip(games.game_key, games.fold))
    by = {}
    for k in keys.tolist():
        if fold_of.get(k) in a.train_folds:
            by.setdefault(owner_of[k], []).append(k)
    by = {p: ks for p, ks in sorted(by.items()) if len(ks) >= 8}
    tr_keys = [k for ks in by.values() for k in ks]
    tr_idx = np.array([row_of[k] for k in tr_keys])
    own_tr = np.array([owner_of[k] for k in tr_keys])
    print(f"rows of {len(keys)} games; training sample {len(tr_keys)} games of {len(by)} players (folds {a.train_folds}) ({time.time() - t0:.0f} s)", flush=True)

    spec = {"m2_64": (["Gm", "Xm"], 64), "m3_64": (["Gm", "Xm", "Em"], 64), "md_64": (["Gm", "Dm"], 64), "m2_128": (["Gm", "Xm"], 128), "m3_128": (["Gm", "Xm", "Em"], 128),
            "f2_128": (["Gm", "Gs", "Xm", "Xs"], 128), "f3_128": (["Gm", "Gs", "Xm", "Xs", "Em", "Es", "Ds"], 128), "s36": (["S"], 0), "m3s_64": (["Gm", "Xm", "Em", "S"], 64)}
    if a.arms:
        spec = {k: v for k, v in spec.items() if k in a.arms or k == "m2_64"}
    rep, ratios = {}, {}
    for name, (cols, k) in spec.items():
        Xall = np.hstack([raw[c] for c in cols]).astype(np.float64)
        mu, sd = Xall[tr_idx].mean(0), Xall[tr_idx].std(0) + 1e-6
        Xall = (Xall - mu) / sd
        if k:
            W, lam = sup_proj(Xall[tr_idx], own_tr, k)
            ratios[name] = [float(lam[i - 1]) for i in (1, 8, 16, 32, 64) if i <= k]
            rep[name] = (Xall @ W).astype(np.float32)
            print(f"  {name}: {Xall.shape[1]} -> {k} dims; between / within ratio of directions 1, 8, 16, 32, 64: " + " ".join(f"{x:.2f}" for x in ratios[name]) + f" ({time.time() - t0:.0f} s)", flush=True)
        else:
            rep[name] = Xall.astype(np.float32)
        del Xall
    names = list(rep)
    models = {}
    for name in names:
        rows_by = {p: rep[name][[row_of[k] for k in ks]] for p, ks in by.items()}
        models[name] = [PlayerEmbedding(seed=s, p_aug=0.0).fit(rows_by) for s in a.emb_seeds]
        print(f"  {name}: {rep[name].shape[1]} dims, episode top-1 at the end {np.mean([m.history[-1]['acc'] for m in models[name]]):.3f} ({time.time() - t0:.0f} s)", flush=True)
    sc, n_all, cl_all = {}, [], []
    for v, s, cands, questions, owner in pools:
        players = sorted(cands)
        truth = np.array([players.index(p) for p in owner])
        n = np.array([len(q) for q in questions])
        n_all.append(n)
        cl_all.append(np.array([f"{v}|{s}|{t}" for t in truth]))
        ck = {p: [k for k, _ in cands[p]] for p in players}
        qk = [[k for k, _ in q] for q in questions]
        need = sorted({k for ks in list(ck.values()) + qk for k in ks})
        missing = [k for k in need if k not in row_of]
        if missing:
            raise KeyError(f"{len(missing)} pool games without rows, e.g. {missing[:3]}")
        idx = np.array([row_of[k] for k in need])
        line = []
        for name in names:
            X = dict(zip(need, rep[name][idx]))
            e = np.mean([per_question(score_embed(m, ck, qk, X), truth) for m in models[name]], axis=0)
            lda = StyleModel(shrink=0.1).fit({p: np.stack([X[k] for k in ck[p]]).astype(np.float64) for p in players})
            l_ = per_question(lda_matrix(lda, np.stack([np.mean([X[k] for k in ks], axis=0) for ks in qk]).astype(np.float64), n.astype(float)), truth)
            sc.setdefault(("emb", name), []).append(e)
            sc.setdefault(("lda", name), []).append(l_)
            line.append(f"{name} {mix(e, n):.4f}/{mix(l_, n):.4f}")
        print(f"fold {v} split {s}: emb/lda  " + "  ".join(line) + f" ({time.time() - t0:.0f} s)", flush=True)
    n_all, cl_all = np.concatenate(n_all), np.concatenate(cl_all)
    rng = np.random.default_rng(0)
    out = {"games": int(len(keys)), "train_games": len(tr_keys), "train_players": len(by), "check_old_path": chk, "ratios": ratios, "scores": {}, "diff_vs_m2_64": {}}
    for learner in ("emb", "lda"):
        base = np.concatenate(sc[(learner, "m2_64")])
        for name in names:
            x = np.concatenate(sc[(learner, name)])
            out["scores"][f"{learner}:{name}"] = {"mix": mix(x, n_all), **{str(m): float(x[n_all == m].mean()) for m in SIZES}}
            msg = f"{learner} {name:8s} mix {mix(x, n_all):.4f} (n=1 {x[n_all == 1].mean():.4f}, n=2 {x[n_all == 2].mean():.4f}, n=3 {x[n_all == 3].mean():.4f}, n=5 {x[n_all == 5].mean():.4f})"
            if name != "m2_64":
                b = boot(x - base, n_all, cl_all, a.boot, rng)
                out["diff_vs_m2_64"][f"{learner}:{name}"] = b
                msg += f"  vs m2_64 {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n=1 {b['by_n']['1']:+.4f}"
            print(msg, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
