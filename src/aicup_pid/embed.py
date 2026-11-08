"""Line B8: a player embedding learnt on per-game trunk rows (METHOD_NOTES 2.23).

The trunk component (aicup_pid.trunkstyle) is a linear discriminant on the 256-dimensional per-game rows. Here a small
network maps every row to a unit vector so that the mean vector of a few games of one player is close to the mean vector
of his other games and far from other players' (the identification task itself, trained episode by episode):
  episode: P players; per player a query set of n games (n drawn from `query_sizes`) and a disjoint support set of up
  to `support` other games; set vector = L2-normalised mean of the game vectors; logits = scale * cos(query_i,
  support_j); loss = cross-entropy both ways (query -> support and support -> query).
Augmentation: with probability `p_aug` a game's row is replaced by its row under a random board symmetry when one is
given (`aug_rows`, from pid_trunk_extract.py --sym-seed), so the embedding learns to ignore orientation.
Scoring: score_embed gives the [Q, C] matrix scale * cos(question set vector, candidate set vector), the same shape and
row order as every other component. Players enter training only with >= min_games rows; fold 0 never enters (callers).
"""
import numpy as np
import torch
from torch import nn

DIM = 256


class EmbedNet(nn.Module):
    def __init__(self, d_in=DIM, hidden=512, d_out=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, d_out))
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


class PlayerEmbedding:
    """fit({player: [G, 256] rows}, aug={player: [G, 256] rows aligned or None}); embed(rows) -> [G, d]; set_vector(rows)."""

    def __init__(self, hidden=512, d_out=128, dropout=0.1, steps=3000, players_per_episode=256, support=40, query_sizes=(1, 2, 3, 5, 10, 15, 20),
                 lr=1e-3, weight_decay=1e-4, p_aug=0.5, min_games=8, seed=0, device=None):
        self.cfg = dict(hidden=hidden, d_out=d_out, dropout=dropout, steps=steps, players_per_episode=players_per_episode, support=support,
                        query_sizes=tuple(query_sizes), lr=lr, weight_decay=weight_decay, p_aug=p_aug, min_games=min_games, seed=seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.mu = self.sd = None
        self.history = []

    def _x(self, rows):
        return torch.as_tensor((np.asarray(rows, np.float32) - self.mu) / self.sd, device=self.device)

    def fit(self, rows_by_player, aug_by_player=None, log=None):
        c = self.cfg
        rng = np.random.RandomState(c["seed"])
        torch.manual_seed(c["seed"])
        players = sorted(p for p, r in rows_by_player.items() if len(r) >= c["min_games"])
        if len(players) < 2:
            raise ValueError("need at least two players with min_games rows")
        allrows = np.concatenate([rows_by_player[p] for p in players]).astype(np.float32)
        self.mu, self.sd = allrows.mean(axis=0), allrows.std(axis=0) + 1e-6
        X = {p: self._x(rows_by_player[p]) for p in players}
        A = {p: self._x(aug_by_player[p]) for p in players if aug_by_player and aug_by_player.get(p) is not None} if aug_by_player else {}
        self.net = EmbedNet(DIM, c["hidden"], c["d_out"], c["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["steps"], pct_start=0.1)
        P = min(c["players_per_episode"], len(players))
        self.net.train()
        for step in range(c["steps"]):
            batch = rng.choice(len(players), P, replace=False)
            q_vecs, s_vecs = [], []
            rows, owners, sides = [], [], []
            for b, i in enumerate(batch):
                p = players[i]
                G = len(X[p])
                n = int(min(rng.choice(c["query_sizes"]), max(1, G // 2)))
                perm = rng.permutation(G)
                q_idx, s_idx = perm[:n], perm[n:n + c["support"]]
                for idx, side in ((q_idx, 0), (s_idx, 1)):
                    xs = X[p][torch.as_tensor(idx, device=self.device)]
                    if p in A and c["p_aug"] > 0:
                        m = torch.as_tensor(rng.rand(len(idx)) < c["p_aug"], device=self.device)
                        xs = torch.where(m[:, None], A[p][torch.as_tensor(idx, device=self.device)], xs)
                    rows.append(xs)
                    owners.append(torch.full((len(idx),), b, device=self.device))
                    sides.append(torch.full((len(idx),), side, device=self.device))
            x, own, side = torch.cat(rows), torch.cat(owners), torch.cat(sides)
            z = self.net(x)
            seg = own * 2 + side  # 2 sets per player
            sums = torch.zeros(2 * P, z.shape[1], device=self.device).index_add_(0, seg, z)
            sets = nn.functional.normalize(sums, dim=-1)
            q, s = sets[0::2], sets[1::2]
            logits = self.net.log_scale.exp() * q @ s.T
            target = torch.arange(P, device=self.device)
            loss = 0.5 * (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == c["steps"] - 1:
                acc = float((logits.argmax(dim=1) == target).float().mean())
                rec = {"step": step, "loss": float(loss.detach()), "acc": acc, "scale": float(self.net.log_scale.detach().exp())}
                self.history.append(rec)
                if log:
                    log(f"  step {step}: loss {rec['loss']:.4f}, episode top-1 {acc:.3f} of {P}, scale {rec['scale']:.2f}", flush=True)
        self.net.eval()
        self.players_trained = len(players)
        return self

    @torch.no_grad()
    def embed(self, rows, batch=8192):
        out = []
        rows = np.asarray(rows, np.float32)
        for i in range(0, len(rows), batch):
            out.append(self.net(self._x(rows[i:i + batch])).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.cfg["d_out"]), np.float32)

    def set_vector(self, rows):
        v = self.embed(rows).sum(axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    @property
    def scale(self):
        return float(self.net.log_scale.detach().exp())

    def state_dict(self):
        return {"cfg": self.cfg, "mu": self.mu, "sd": self.sd, "net": {k: v.cpu() for k, v in self.net.state_dict().items()}, "history": self.history,
                "players_trained": self.players_trained}

    @classmethod
    def from_state(cls, st, device=None):
        m = cls(**st["cfg"], device=device)
        m.mu, m.sd, m.history, m.players_trained = st["mu"], st["sd"], st["history"], st["players_trained"]
        m.net = EmbedNet(DIM, st["cfg"]["hidden"], st["cfg"]["d_out"], st["cfg"]["dropout"]).to(m.device)
        m.net.load_state_dict(st["net"])
        m.net.eval()
        return m


def score_embed(model, cand_keys_by_player, question_keys, store):
    """[Q, C] scale * cosine of the question and candidate set vectors (candidates in sorted id order)."""
    players = sorted(cand_keys_by_player)
    C = np.stack([model.set_vector(np.stack([store[k] for k in cand_keys_by_player[p]])) for p in players])
    if not question_keys:
        return np.zeros((0, len(players)))
    Q = np.stack([model.set_vector(np.stack([store[k] for k in keys])) for keys in question_keys])
    return model.scale * (Q @ C.T)
