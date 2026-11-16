"""Shared conditional action models; current query action is never an input."""
import math
import torch
from torch import nn


def mlp(d, out):
    return nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, out))


class BehaviorPolicy(nn.Module):
    def __init__(self, mode='contextual'):
        super().__init__()
        if mode not in ('background', 'global', 'contextual'):
            raise ValueError('unknown model mode')
        self.mode = mode
        self.state = nn.Sequential(nn.LayerNorm(388), mlp(388, 64))
        self.history = nn.GRU(6, 32, batch_first=True)
        self.context = nn.Sequential(nn.Linear(96, 64), nn.LayerNorm(64))
        self.value = nn.Sequential(nn.LayerNorm(386), mlp(386, 64))
        self.character = mlp(128, 64)
        self.key = nn.Linear(64, 64, bias=False)
        self.query = nn.Linear(64, 64, bias=False)
        self.action = nn.Sequential(nn.LayerNorm(385), nn.Linear(385, 64))
        self.background = mlp(64, 64)
        self.personal = mlp(192, 64)
        for head in (self.background, self.personal):
            nn.init.normal_(head[-1].weight, std=.005)
            nn.init.zeros_(head[-1].bias)

    def encode_context(self, state, history):
        shape = state.shape[:-1]
        _, h = self.history(history.reshape(-1, 8, 6))
        return self.context(torch.cat([self.state(state), h[-1].reshape(*shape, 32)], -1))

    def reference(self, state, history, value, mask, ablation=None):
        if mask.dtype != torch.bool or not mask.any(-1).all():
            raise ValueError('every candidate needs a valid reference observation')
        if ablation not in (None, 'no_actions', 'permute'):
            raise ValueError('unknown ablation')
        # Mask before nonlinear operations, so padded extreme inputs cannot leak.
        k = self.encode_context(state.masked_fill(~mask[..., None], 0),
                                history.masked_fill(~mask[..., None, None], 0))
        v = self.value(torch.zeros_like(value) if ablation == 'no_actions' else value.masked_fill(~mask[..., None], 0))
        g = self.character(torch.cat([k, v], -1)).masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]
        if ablation == 'permute':
            # Deterministic nonidentity permutation within each candidate. Global
            # character above remains EXACTLY fixed: only context/value binding changes.
            v = v.clone()
            for i in range(len(v)):
                ix = torch.where(mask[i])[0]
                v[i, ix] = v[i, ix.roll(1)].clone()
        return g, k, v

    def forward(self, phi, logp, state, history, ref_state, ref_history, ref_value, ref_mask, ablation=None):
        if not torch.isfinite(logp).any(-1).all() or torch.isnan(logp).any() or torch.isposinf(logp).any():
            raise ValueError('invalid base policy')
        ctx = self.encode_context(state, history)
        act = self.action(phi)
        base = logp + torch.einsum('qd,qad->qa', self.background(ctx), act) / math.sqrt(64)
        c = len(ref_state)
        if self.mode == 'background':
            return base.log_softmax(-1)[:, None].expand(-1, c, -1)
        g, k, v = self.reference(ref_state, ref_history, ref_value, ref_mask, ablation)
        if self.mode == 'contextual':
            attn = torch.einsum('qd,cmd->qcm', self.query(ctx), self.key(k)) / math.sqrt(64)
            attn = attn.masked_fill(~ref_mask[None], -torch.inf).softmax(-1)
            r = torch.einsum('qcm,cmd->qcd', attn, v)
        else:
            r = torch.zeros((len(ctx), c, 64), device=ctx.device, dtype=ctx.dtype)
        z = self.personal(torch.cat([ctx[:, None].expand(-1, c, -1), g[None].expand(len(ctx), -1, -1), r], -1))
        return (base[:, None] + torch.einsum('qcd,qad->qca', z, act) / math.sqrt(64)).log_softmax(-1)
