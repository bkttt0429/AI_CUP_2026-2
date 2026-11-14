#!/usr/bin/env python
"""Supervised per-ply features (METHOD_NOTES 2.88, probe B).

The stored per-ply input of the sequence networks is the humanv0 trunk vector pushed through a RANDOM projection. A
supervised projection (the directions of the per-game means that separate training players best) keeps the identity
information in 32-64 dimensions. Here:
  fit(...)      the projection from per-game rows of unprojected statistics (scripts/pid_trunk_extract.py --full)
  load_sup(...) the per-ply parts of scripts/pid_sup_ply_extract.py, standardised per dimension over all their plies
  dense(...)    a padded [games, 256, K] array aligned with a list of game keys (zeros = the mean where a ply is not the
                subject's, beyond the game, or the game has no part)
  put(...)      writes the features into chosen input dimensions of a batch, in the units the network standardises
"""
import glob

import numpy as np

C = 384


def fit(rows, owners, k=64, ridge=1e-3):
    """rows [G, 4C] (`full`: mean / std of the pooled vector, mean / std of the played-point vector), owners [G].
    Returns mu [2C], sd [2C], W [2C, k]: feature = ((x - mu) / sd) @ W for x = [pooled, played-point] (per game mean or per ply)."""
    from scipy.linalg import eigh
    X = rows[:, np.r_[0:C, 2 * C:3 * C]].astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    _pl, inv = np.unique(owners, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    cm = np.stack([np.bincount(inv, weights=X[:, j]) for j in range(X.shape[1])], axis=1) / cnt[:, None]
    resid = X - cm[inv]
    Sw = resid.T @ resid / max(len(X) - len(cnt), 1)
    Sb = np.cov(cm.T)
    lam, W = eigh(Sb, Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))
    return mu.astype(np.float32), sd.astype(np.float32), W[:, ::-1][:, :k].astype(np.float32), lam[::-1][:k]


def load_sup(dirs):
    """(index {key: (start, n)}, t [N] int16 ply numbers, Z [N, K] float16 standardised) over every part of the directories."""
    keys, ns, ts, zs = [], [], [], []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            keys.append(z["keys"].astype(str)); ns.append(z["n"].astype(np.int64)); ts.append(z["t"]); zs.append(z["z"])
    if not keys:
        raise FileNotFoundError(f"no supervised per-ply parts in {dirs}")
    keys, n, t, Z = np.concatenate(keys), np.concatenate(ns), np.concatenate(ts), np.concatenate(zs).astype(np.float32)
    Z = ((Z - Z.mean(0)) / (Z.std(0) + 1e-6)).astype(np.float16)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    return {k: (int(s), int(m)) for k, s, m in zip(keys.tolist(), start.tolist(), n.tolist())}, t, Z


def dense(sup, keys, T=256, alloc=None):
    """idx [len(keys)] int32 (row of D; -1: the game has no supervised features), D [n_have, T, K] float16."""
    index, t, Z = sup
    have = [i for i, k in enumerate(keys) if k in index]
    idx = np.full(len(keys), -1, np.int32)
    idx[have] = np.arange(len(have), dtype=np.int32)
    shape = (max(len(have), 1), T, Z.shape[1])
    D = alloc(shape, np.float16) if alloc is not None else np.zeros(shape, np.float16)
    for j, i in enumerate(have):
        s, m = index[keys[i]]
        tt = t[s:s + m].astype(np.int64)
        ok = tt < T
        D[j, tt[ok]] = Z[s:s + m][ok]
    return idx, D


def put(gl, z, mu, sd, dims, only=False):
    """gl [B, T, 128] and z [B, T, K] (torch tensors or numpy arrays of the same kind): the supervised features in the input
    dimensions dims = (d0, d1), scaled so that the network's own standardisation ((x - mu) / sd) gives z back; with
    only=True every other dimension is set to its mean (the network then sees nothing of the random projection)."""
    d0, d1 = dims
    assert d1 - d0 == z.shape[-1], (dims, z.shape)
    if only:
        gl = gl * 0 + mu
    gl[..., d0:d1] = z * sd[d0:d1] + mu[d0:d1]
    return gl
