"""METHOD_NOTES 2.192: two variants of the window input of pid_win.py (probe W2), for the short-run filter.

pid_win.py shows the network, per ply, the size x size neighbourhood of the played point with six cell states (empty, own
stone, other stone, off the board, the previous ply's stone, the ply before that).  The merged retrain (notes 2.190 / 2.191)
found this input worth more than everything else tried since the by-games sampling, so two cheap extensions of the SAME
information source (the stones played so far; still no feature extraction, no file):

  age    every stone in the window also carries HOW LONG AGO it was played, in eight buckets (1, 2, 3-4, 5-8, 9-16, 17-32,
         33-64, more plies ago) - whether a move joins a fight that is going on or returns to an old position
  prev   a second window of the same size around the PREVIOUS ply's point (the opponent's last move), read by its own
         encoder - what the mover was answering, or left unanswered when the move is elsewhere on the board

cfg: {"size": odd, "ch": channels, "age": 0|1, "prev": 0|1}.  pid_win.wrap() sends a cfg with "age" or "prev" here, so the
loaders (fine-tune, embedding) need no change.  Every new part's contribution starts at zero: with age = prev = 0 the module
is pid_win's, and at the start of training a wrapped network equals the plain one.
"""


def windows2(pos, mov, size, age=False, prev=False):
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours ->
    cells [B, T, size * size] (the six states of pid_win.windows), ok [B, T];
    ages [B, T, size * size] or None (0 no stone / off the board, 1..8 the age bucket of the stone);
    pcells, pages (the same around the previous ply's point, states still relative to the mover) and pok [B, T], or None."""
    import torch
    import torch.nn.functional as F
    B, T = pos.shape
    dev = pos.device
    ok = pos < 361
    t = torch.arange(1, T + 1, device=dev, dtype=torch.int16)[None].expand(B, T)
    stamp = torch.zeros(B, T, 363, dtype=torch.int16, device=dev)  # (ply number) * 2 + colour where a stone lands
    stamp.scatter_(2, pos[..., None], torch.where(ok, t * 2 + mov.to(torch.int16), torch.zeros_like(t))[..., None])
    last = torch.cummax(stamp[..., :361], dim=1).values  # the latest stone on every point after each ply
    before = torch.zeros_like(last)
    before[:, 1:] = last[:, :-1]
    del stamp, last
    ply = (before >> 1).long()  # 0 = empty, else the ply number (1-based) of the stone
    col = (before & 1).long()
    now = torch.arange(1, T + 1, device=dev)[None, :, None]  # the 1-based number of the current ply
    state = torch.where(ply == 0, 0, torch.where(col == mov[..., None], 1, 2))
    state = torch.where((ply > 0) & (ply == now - 1), 4, state)
    state = torch.where((ply > 0) & (ply == now - 2), 5, state)
    del col, before
    p = size // 2
    W = 19 + 2 * p
    d = torch.arange(size, device=dev)
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # the window's cells from its corner (y - p, x - p) in padded coordinates
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)
    del state
    aboard = None
    if age:
        bounds = torch.tensor([1, 2, 4, 8, 16, 32, 64], device=dev)
        bucket = torch.where(ply > 0, torch.bucketize(now - ply, bounds) + 1, torch.zeros_like(ply)).to(torch.uint8)
        aboard = F.pad(bucket.reshape(B, T, 19, 19), (p, p, p, p), value=0).reshape(B, T, W * W)
        del bucket
    del ply

    def cut(cen):
        idx = ((cen // 19) * W + cen % 19)[..., None] + off[None, None]
        return board.gather(2, idx), (aboard.gather(2, idx).long() if aboard is not None else None)

    cells, ages = cut(torch.where(ok, pos, torch.zeros_like(pos)))
    pcells = pages = pok = None
    if prev:
        ppos = torch.full_like(pos, 362)
        ppos[:, 1:] = pos[:, :-1]
        pok = ok & (ppos < 361)
        pcells, pages = cut(torch.where(pok, ppos, torch.zeros_like(ppos)))
    return cells, ok, ages, pcells, pages, pok


def wrap(net, cfg):
    """Adds the window encoder(s) to a built m88_seqid.SeqID (in place) and returns it."""
    import torch
    from torch import nn
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))
    age, prev = bool(cfg.get("age")), bool(cfg.get("prev"))
    assert size % 2 == 1 and 3 <= size <= 19 and (age or prev)
    D = net.gl.out_features
    base = net.__class__

    class Enc(nn.Module):  # pid_win's encoder, with the optional age embedding (starts at zero)
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(6, 16)
            if age:
                self.age = nn.Embedding(9, 16)
                nn.init.zeros_(self.age.weight)
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
            self.out = nn.Linear(2 * ch, D)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, cells, ages, flat, B, T):
            x = self.emb(cells.reshape(B * T, -1)[flat])
            if ages is not None:
                x = x + self.age(ages.reshape(B * T, -1)[flat])
            h = self.conv(x.reshape(-1, size, size, 16).permute(0, 3, 1, 2))
            c = h.shape[-1] // 2
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)
            res[flat] = f
            return res.reshape(B, T, -1)

    class Win2(nn.Module):
        def __init__(self):
            super().__init__()
            self.own = Enc()
            if prev:
                self.prev = Enc()
            self.size = size

        def forward(self, pos, mov):
            B, T = pos.shape
            with torch.no_grad():
                cells, ok, ages, pcells, pages, pok = windows2(pos, mov, size, age, prev)
            res = self.own(cells, ages, ok.reshape(-1), B, T)
            if prev:
                res = res + self.prev(pcells, pages, pok.reshape(-1), B, T)
            return res

    class SeqIDW2(base):
        def forward(self, gl, pos, mov, length, subj):
            B, T = pos.shape
            x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) \
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = x + self.win(pos, mov).to(x.dtype)
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)

    net.win = Win2().to(net.gl.weight.device)
    net.__class__ = SeqIDW2
    net.win_cfg = {"size": size, "ch": ch, "age": int(age), "prev": int(prev)}
    return net


def selftest():
    """windows2 against a plain replay (no captures) on random games, and its first two outputs against pid_win.windows."""
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pid_win
    rng = np.random.default_rng(0)
    B, T, size = 6, 90, 9
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 60:] = 362; pos[2, 7] = pos[2, 3]; pos[3, 10] = 362  # a pass, padding, a point played twice, a hidden ply
    mov = np.tile(np.arange(T) % 2, (B, 1))
    tp, tm = torch.as_tensor(pos), torch.as_tensor(mov)
    cells, ok, ages, pcells, pages, pok = [x.numpy() for x in windows2(tp, tm, size, True, True)]
    c0, ok0 = [x.numpy() for x in pid_win.windows(tp, tm, size)]
    same = int((cells != c0).sum() + (ok != ok0).sum())
    c1, ok1, a1, p1, q1, r1 = windows2(tp, tm, size, False, False)
    assert a1 is None and p1 is None and q1 is None and r1 is None and int((c1.numpy() != c0).sum()) == 0
    p = size // 2

    def bucket(d):
        return 1 + int(np.searchsorted([1, 2, 4, 8, 16, 32, 64], d, side="left"))

    bad = {"cells": 0, "ages": 0, "pcells": 0, "pages": 0, "pok": 0}
    n = 0
    for b in range(B):
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)
        for t in range(T):
            has = pos[b, t] < 361
            hasp = has and t > 0 and pos[b, t - 1] < 361
            bad["pok"] += int(pok[b, t] != hasp)
            for which, on, cy_cx in (("", has, pos[b, t]), ("p", hasp, pos[b, t - 1] if t > 0 else 0)):
                if not on:
                    continue
                y, x = divmod(int(cy_cx), 19)
                got_c, got_a = (cells, ages) if which == "" else (pcells, pages)
                for i in range(size):
                    for j in range(size):
                        yy, xx = y - p + i, x - p + j
                        if not (0 <= yy < 19 and 0 <= xx < 19):
                            want, wa = 3, 0
                        elif board[yy, xx] == 0:
                            want, wa = 0, 0
                        else:
                            dd = (t + 1) - when[yy, xx]
                            want = 4 if dd == 1 else 5 if dd == 2 else (1 if board[yy, xx] - 1 == mov[b, t] else 2)
                            wa = bucket(dd)
                        bad[which + "cells"] += int(got_c[b, t, i * size + j] != want)
                        bad[which + "ages"] += int(got_a[b, t, i * size + j] != wa)
                        n += 1
            if has:
                y, x = divmod(int(pos[b, t]), 19)
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1
    print(f"windows2 against a plain replay: differing {bad} of {n} cells; against pid_win.windows: {same} differences; "
          f"age buckets seen {sorted(set(ages.reshape(-1).tolist()))}")
    return 0 if same == 0 and not any(bad.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
