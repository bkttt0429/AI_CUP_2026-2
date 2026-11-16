"""Conditional choice evidence, with reference-only ridge shrinkage.

The score is a SECOND ORDER log-likelihood-ratio approximation. It is not
a calibrated identity probability or a full posterior predictive model.
"""
import hashlib

import numpy as np
import torch


def _inputs(logp, phi, played):
    if logp.ndim != 2 or phi.ndim != 3 or phi.shape[:2] != logp.shape or played.shape != logp.shape[:1]:
        raise ValueError("incompatible choice shapes")
    if len(logp) == 0 or not torch.isfinite(phi).all() or torch.isnan(logp).any() or torch.isposinf(logp).any():
        raise ValueError("empty/nonfinite choices")
    if played.dtype not in (torch.int32, torch.int64) or (played < 0).any() or (played >= logp.shape[1]).any():
        raise ValueError("invalid played index")
    if not torch.isfinite(logp.gather(1, played.long()[:, None])).all():
        raise ValueError("actual move masked or nonfinite")
    if not torch.allclose(torch.logsumexp(logp, 1), torch.zeros(len(logp), device=logp.device, dtype=logp.dtype), atol=1e-5, rtol=0):
        raise ValueError("log probabilities must be normalised")


def choice_moments(logp, phi, played):
    _inputs(logp, phi, played)
    p = logp.exp()
    mean = torch.einsum("ta,tad->td", p, phi)
    centred = phi - mean[:, None]
    U = (phi[torch.arange(len(played), device=played.device), played.long()] - mean).sum(0)
    weighted = centred * p.sqrt()[:, :, None]
    flat = weighted.flatten(0, 1)
    F = flat.T @ flat
    return U, (F + F.T) * .5, len(played)


def context_features(psi, value):
    v = torch.as_tensor(value, dtype=psi.dtype, device=psi.device)
    if v.shape != psi.shape[:1] or not torch.isfinite(v).all() or (v.abs() > 1 + 1e-6).any():
        raise ValueError("value must be finite, side-to-move and in [-1,1]")
    return torch.cat((psi, psi * v[:, None, None]), dim=-1)


def permute_context(value, plies, *, pool, key):
    v, t = np.asarray(value), np.asarray(plies)
    if v.ndim != 1 or t.shape != v.shape or np.any(t < 0) or np.any(t >= 256):
        raise ValueError("context/ply mismatch")
    out = v.copy()
    phases = np.digitize(t, [60, 150])
    for phase in range(3):
        ix = np.flatnonzero(phases == phase)
        salt = f"choice-vperm-v1|{pool}|{key}|{phase}".encode()
        seed = int.from_bytes(hashlib.sha256(salt).digest()[:8], "little")
        out[ix] = v[np.random.default_rng(seed).permutation(ix)]
    return out


def normalise_moments(U, F, scale):
    U, F, scale = np.asarray(U, dtype=np.float64), np.asarray(F, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if scale.shape != U.shape[-1:] or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("invalid moment scales")
    return U / scale, F / (scale[:, None] * scale[None, :])


def validate_moments(U, F):
    U, F = np.asarray(U, dtype=np.float64), np.asarray(F, dtype=np.float64)
    if U.ndim != 1 or F.shape != (len(U), len(U)) or not np.isfinite(U).all() or not np.isfinite(F).all():
        raise ValueError("invalid moments")
    tol = 1e-5 * max(float(np.trace(F)), 1.)
    if np.max(np.abs(F - F.T)) > tol or np.linalg.eigvalsh((F + F.T) / 2).min() < -tol:
        raise ValueError("Fisher matrix is not positive semidefinite/symmetric")
    return U, (F + F.T) / 2


def fit_candidate(U, F, n, ridge):
    if n <= 0 or ridge <= 0 or not np.isfinite(n) or not np.isfinite(ridge):
        raise ValueError("positive reference count and ridge required")
    U, F = validate_moments(U, F)
    z = np.linalg.solve(F + ridge * n * np.eye(len(U)), U)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite fitted preference")
    return z


def score_candidates(Uq, Fq, Z, block=128):
    U, F, Z = (np.asarray(x, dtype=np.float64) for x in (Uq, Fq, Z))
    if U.ndim != 2 or Z.ndim != 2 or F.shape != (len(U), U.shape[1], U.shape[1]) or Z.shape[1] != U.shape[1]:
        raise ValueError("invalid score shapes")
    if block < 1 or not all(np.isfinite(x).all() for x in (U, F, Z)):
        raise ValueError("nonfinite scoring input/invalid block")
    out = np.empty((len(U), len(Z)), dtype=np.float64)
    for i in range(0, len(U), block):
        for j in range(0, len(Z), block):
            z = Z[j:j + block]
            quad = np.einsum("cd,qde,ce->qc", z, F[i:i + block], z, optimize=True)
            out[i:i + block, j:j + block] = U[i:i + block] @ z.T - .5 * quad
    return out


def exact_delta(logp, phi, played, z):
    _inputs(logp, phi, played)
    z = torch.as_tensor(z, dtype=phi.dtype, device=phi.device)
    if z.shape != phi.shape[-1:] or not torch.isfinite(z).all():
        raise ValueError("invalid preference")
    tilt = phi @ z
    return tilt.gather(1, played.long()[:, None])[:, 0] - torch.logsumexp(logp + tilt, dim=1)
