"""Local response descriptors and a reference-conditioned, legal action policy."""
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=4)
def orbit_kernels(device):
    """A 9×9 patch has 15 offset classes under board rotations and reflections."""
    offsets = torch.arange(-4, 5)
    y, x = torch.meshgrid(offsets, offsets, indexing="ij")
    major = torch.maximum(x.abs(), y.abs())
    minor = torch.minimum(x.abs(), y.abs())
    masks = torch.stack([(major == i) & (minor == j) for i in range(5) for j in range(i + 1)]).float()
    return (masks / masks.sum((1, 2), keepdim=True))[:, None].repeat(3, 1, 1, 1).to(device)


def local_geometry(planes, base_prob):
    """Input is a pre-move KataGo board. Plane 9 marks the previous opponent stone."""
    n = len(planes)
    if planes.shape[2:] != (19, 19) or base_prob.shape != (n, 362):
        raise ValueError("expected a 19×19 state and all 362 policy probabilities")
    features = F.conv2d(planes[:, [1, 2, 0]].float(), orbit_kernels(planes.device), padding=4, groups=3)
    features[:, 30:] = 1 - features[:, 30:]
    shape = F.pad(features.flatten(2).transpose(1, 2), (0, 0, 0, 1))
    mark = planes[:, 9].reshape(n, 361)
    exists = mark.any(-1)
    idx = mark.argmax(-1)
    previous = shape[torch.arange(n, device=planes.device), idx] * exists[:, None]
    pos = torch.arange(361, device=planes.device)
    dx = (pos[None] % 19 - (idx % 19)[:, None]).abs().float()
    dy = (pos[None] // 19 - (idx // 19)[:, None]).abs().float()
    distance = torch.maximum(dx, dy)
    relative = torch.stack([torch.exp(-distance / 3), distance / 18, (dx + dy) / 36,
                            torch.minimum(dx, dy) / 18, (distance <= 4).float()], -1)
    relative = F.pad(relative * exists[:, None, None], (0, 0, 0, 1))
    summary = torch.cat([previous, (base_prob[..., None] * shape).sum(1),
                         (base_prob[..., None] * relative).sum(1), exists[:, None].float()], -1)
    return shape, previous, relative, summary


def action_basis(trunk, shape, previous, relative, *, products):
    """Every legal proposal, including pass, has its own row before labels are scored."""
    prior = previous[:, None].expand(-1, shape.size(1), -1)
    if products:
        pair = shape * prior
        response = relative[..., :1] * prior
    else:
        pair = shape.square()
        response = shape.clamp_min(0).sqrt()
    return torch.cat([trunk, shape, prior, relative, pair, response], -1)


class ConditionalChoice(nn.Module):
    def __init__(self, context_dim, action_dim, hidden=64, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.hidden = hidden
        self.embed = nn.Linear(action_dim, hidden, bias=False)
        if not uniform:
            self.key = nn.Sequential(nn.Linear(context_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, context, actions, baseline, reference_context, reference_residual):
        # Reference games are independent units. Within each game there is one null key.
        n, games, positions = reference_residual.shape[:3]
        if min(n, games, positions) < 1:
            raise ValueError("empty reference data")
        values = self.embed(reference_residual)
        if self.uniform:
            memory = (values.sum(2) / (positions + 1)).mean(1).expand(len(context), -1, -1)
            null = context.new_full((len(context), n), 1 / (positions + 1))
        else:
            logits = torch.einsum("qd,cgtd->qcgt", self.key(context), self.key(reference_context)) / math.sqrt(self.hidden)
            weights = F.softmax(F.pad(logits.float(), (0, 1)), -1)
            memory = torch.einsum("qcgt,cgtd->qcd", weights[..., :-1].to(values.dtype), values) / games
            null = weights[..., -1].mean(2)
        correction = torch.einsum("qad,qcd->qca", self.embed(actions), memory) / math.sqrt(self.hidden)
        return F.log_softmax(baseline.float()[:, None] + 2 * torch.tanh(correction.float()), -1), null


def phase_permutation(plies, seed):
    """Permute residuals within one player, one reference game and a 32-ply band."""
    plies = np.asarray(plies)
    phase = plies // 32
    order = np.broadcast_to(np.arange(plies.shape[-1]), plies.shape).copy()
    rng = np.random.default_rng(seed)
    for bands, out in zip(phase.reshape(-1, phase.shape[-1]), order.reshape(-1, phase.shape[-1])):
        for band in np.unique(bands):
            idx = rng.permutation(np.flatnonzero(bands == band))
            if len(idx) > 1:
                out[idx] = np.roll(idx, 1)
    return order
