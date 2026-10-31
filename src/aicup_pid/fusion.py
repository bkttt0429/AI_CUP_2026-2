"""Fuse per-candidate log scores of several components into Top-5 lists (docs/ARCHITECTURE.md ladder (5)).

Each component gives a [Q, C] matrix of scores (questions x candidates). standardise centres and scales every row, so
components live on one scale; Fusion learns one weight per component by maximising the multinomial log-likelihood of
the true candidate under softmax(sum_j w_j z_j) (L-BFGS in float64, tiny L2); topk returns the indices of the best k.
"""
import numpy as np
import torch


def standardise(S):
    S = np.asarray(S, dtype=float)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (S - mu) / sd


def topk(S, k=5):
    """[Q, k] candidate indices with the highest scores, best first."""
    S = np.asarray(S, dtype=float)
    k = min(k, S.shape[1])
    idx = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return idx


class Fusion:
    """weights over components; fit(components [J x (Q, C)], truth [Q]) ; scores(components) -> [Q, C]."""

    def __init__(self, l2=1e-3, max_iter=200):
        self.l2, self.max_iter = l2, max_iter
        self.w = None

    def fit(self, components, truth):
        Z = torch.tensor(np.stack([standardise(S) for S in components]), dtype=torch.float64)  # [J, Q, C]
        y = torch.tensor(np.asarray(truth, dtype=np.int64))
        w = torch.ones(Z.shape[0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            logits = torch.einsum("j,jqc->qc", w, Z)
            loss = torch.nn.functional.cross_entropy(logits, y) + self.l2 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach().numpy().copy()
        return self

    def scores(self, components):
        if self.w is None:
            raise ValueError("fit first")
        return np.einsum("j,jqc->qc", self.w, np.stack([standardise(S) for S in components]))

    def state(self):
        return {"weights": self.w.tolist(), "l2": self.l2}
