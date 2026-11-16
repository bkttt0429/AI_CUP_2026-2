"""Reference-conditioned full-action policy, with no query-action input to retrieval."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReferencePolicy(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=64, mode="attention"):
        super().__init__()
        if mode not in ("attention", "uniform"):
            raise ValueError("unknown memory mode")
        self.mode, self.hidden = mode, hidden
        self.key = nn.Sequential(nn.Linear(context_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.value = nn.Linear(action_dim, hidden, bias=False)
        self.action = nn.Linear(action_dim, hidden, bias=False)

    def forward(self, qcontext, actions, base_logp, reference_context, reference_value):
        # qcontext[Q,D], actions[Q,A,D], references[C,G,T,D].
        c, g, t = reference_value.shape[:3]
        if min(c, g, t) < 1:
            raise ValueError("empty memory")
        values = self.value(reference_value)
        if self.mode == "attention":
            keys = self.key(reference_context)
            query = self.key(qcontext)
            logits = torch.einsum("qh,cgth->qcgt", query, keys) / math.sqrt(self.hidden)
            weights = F.softmax(torch.cat([logits, torch.zeros_like(logits[..., :1])], -1), -1)
            z = torch.einsum("qcgt,cgth->qcgh", weights[..., :-1], values).mean(2)
            null = weights[..., -1].mean(2)
        else:
            z = (values.sum(2) / (t + 1)).mean(1)[None].expand(len(qcontext), -1, -1)
            null = qcontext.new_full((len(qcontext), c), 1 / (t + 1))
        action = self.action(actions)
        delta = 2 * torch.tanh(torch.einsum("qah,qch->qca", action, z) / math.sqrt(self.hidden))
        logits = base_logp.float()[:, None] + delta.float()
        return F.log_softmax(logits, -1), null.float()


def played_scores(logp, played):
    return logp.gather(-1, played[:, None, None].expand(-1, logp.shape[1], 1))[..., 0]


def episodic_loss(logp, played, owners):
    s = played_scores(logp, played)
    n = logp.shape[1]
    counts = torch.bincount(owners, minlength=n)
    if (counts == 0).any():
        raise ValueError("each episode candidate needs query positions")
    score = s.new_zeros(n, n).index_add(0, owners, s) / counts[:, None]
    identity = F.cross_entropy(score * 10, torch.arange(n, device=s.device))
    move_nll = -s[torch.arange(len(s), device=s.device), owners].mean()
    return identity + .1 * move_nll


def owner_metrics(scores):
    """[owner,3 single query games,candidate] -> equal-weight query-size metrics."""
    p = len(scores)
    if scores.shape != (p, 3, p):
        raise ValueError("need three query games per owner against every candidate")
    out, bysize = [], {}
    for size in (1, 2, 3):
        s = scores if size == 1 else scores[:, None, :size].sum(2)
        own = np.take_along_axis(s, np.arange(p)[:, None, None], 2)
        rank = ((s > own) | ((s == own) & (np.arange(p)[None, None] < np.arange(p)[:, None, None]))).sum(2)
        rs = np.where(rank < 5, np.exp(-rank.astype(float)), 0).mean(1)
        top1 = (rank == 0).mean(1)
        out.append(np.stack([rs, top1], 1))
        bysize[str(size)] = {"rank_score": float(rs.mean()), "top1": float(top1.mean())}
    per = np.mean(out, axis=0)
    return per[:, 0], {"rank_score": float(per[:, 0].mean()), "top1": float(per[:, 1].mean()), "by_size": bysize}


def interval(values, seed=20261116, draws=5000):
    values = np.asarray(values, dtype=float)
    ix = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    return {"mean": float(values.mean()), "ci95": np.quantile(values[ix].mean(1), [.025, .975]).tolist()}
