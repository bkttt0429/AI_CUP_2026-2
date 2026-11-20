"""METHOD_NOTES 2.170 (probe W): the stones around the played point as an explicit input of the sequence network.

The network of m88_seqid reads, per ply, two projected trunk vectors and the played point; what the board looks like around
the move reaches it only through the 64 projected dimensions of the trunk at that point and through attention over the
earlier plies' point embeddings.  Here every ply also gets the size x size neighbourhood of its point, read by a small
convolutional encoder that is trained with the network:

  cell states   0 empty, 1 a stone of the mover, 2 a stone of the other side, 3 off the board,
                4 the stone of the previous ply, 5 the stone of the ply before that (the mover's own last move)
  the board     the stones PLAYED so far, the latest one on a point (captures are not replayed: a captured stone stays
                until something is played there again) - built on the GPU from the plies' points and colours, so no feature
                extraction and no file is needed, and the trainer's board symmetries / hidden plies apply to it as to the points.

wrap(net, cfg) turns a built m88_seqid network into this variant in place (its class gets a forward with one more term; the
new term's last layer starts at zero, so the wrapped network starts as the plain one).  A checkpoint of such a network
carries cfg under "win"; the loaders call wrap() when they find it.
"""


def windows(pos, mov, size):
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours -> [B, T, size * size] cell states
    of the board before each ply around its point, and [B, T] whether the ply has a point."""
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
    del ply, col, before
    p = size // 2
    W = 19 + 2 * p
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)
    cen = torch.where(ok, pos, torch.zeros_like(pos))
    d = torch.arange(size, device=dev)
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # the window's cells from its corner (y - p, x - p) in padded coordinates
    idx = ((cen // 19) * W + cen % 19)[..., None] + off[None, None]
    return board.gather(2, idx), ok


def wrap(net, cfg):
    """Adds the window encoder to a built m88_seqid.SeqID (in place) and returns it. cfg: {"size": odd, "ch": channels}."""
    if cfg.get("exact"):  # METHOD_NOTES 2.180: the exact board (captured stones removed) lives in pid_winx.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pid_winx
        return pid_winx.wrap(net, cfg)
    if cfg.get("age") or cfg.get("prev"):  # METHOD_NOTES 2.192: the window variants (stone ages, a second window) live in pid_win2.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pid_win2
        return pid_win2.wrap(net, cfg)
    import torch
    from torch import nn
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))
    assert size % 2 == 1 and 3 <= size <= 19
    D = net.gl.out_features
    base = net.__class__

    class Win(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(6, 16)
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
            self.out = nn.Linear(2 * ch, D)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
            self.size = size

        def forward(self, pos, mov):
            B, T = pos.shape
            with torch.no_grad():
                cells, ok = windows(pos, mov, self.size)
            flat = ok.reshape(-1)
            x = self.emb(cells.reshape(B * T, -1)[flat]).reshape(-1, self.size, self.size, 16).permute(0, 3, 1, 2)
            h = self.conv(x)
            c = h.shape[-1] // 2
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)
            res[flat] = f
            return res.reshape(B, T, -1)

    class SeqIDW(base):
        def forward(self, gl, pos, mov, length, subj):
            B, T = pos.shape
            x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) \
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = x + self.win(pos, mov).to(x.dtype)
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)

    net.win = Win().to(net.gl.weight.device)
    net.__class__ = SeqIDW
    net.win_cfg = {"size": size, "ch": ch}
    return net


def selftest():
    """The window builder against a plain replay (no captures) on random games; the wrapped network against the plain one at start."""
    import numpy as np
    import torch
    rng = np.random.default_rng(0)
    B, T, size = 6, 40, 9
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 30:] = 362; pos[2, 7] = pos[2, 3]  # a pass, padding, a point played twice
    mov = np.tile(np.arange(T) % 2, (B, 1))
    cells, ok = windows(torch.as_tensor(pos), torch.as_tensor(mov), size)
    cells, ok = cells.numpy(), ok.numpy()
    p = size // 2
    bad = 0
    for b in range(B):
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)
        for t in range(T):
            if pos[b, t] < 361:
                y, x = divmod(pos[b, t], 19)
                for i in range(size):
                    for j in range(size):
                        yy, xx = y - p + i, x - p + j
                        if not (0 <= yy < 19 and 0 <= xx < 19):
                            want = 3
                        elif board[yy, xx] == 0:
                            want = 0
                        elif when[yy, xx] == t:  # 1-based number of the previous ply
                            want = 4
                        elif when[yy, xx] == t - 1:
                            want = 5
                        else:
                            want = 1 if board[yy, xx] - 1 == mov[b, t] else 2
                        bad += int(cells[b, t, i * size + j] != want)
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1
            assert ok[b, t] == (pos[b, t] < 361)
    print(f"window builder against a plain replay: {bad} differing cells of {B * T * size * size}")
    return bad == 0


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
