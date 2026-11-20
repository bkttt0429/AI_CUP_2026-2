"""METHOD_NOTES 2.202: further variants of the window input (probe W3), for the next short runs.

pid_win.py: the size x size neighbourhood of the played point, six cell states relative to the mover. pid_win2.py added a
second window around the PREVIOUS ply's point (the opponent's last move; "prev"), which met the short-run rule and became
the families o40-o43. Three more variants of the same information source (the stones played so far, no file):

  own2    a third window around the point of the ply BEFORE the previous one - the mover's OWN last move: what became of
          it (answered or ignored by the opponent, and whether the mover now continues there or leaves it)
  ch      channels of every window encoder (64 in all families so far)
  depth   further residual blocks (two 3x3 convolutions each, the second one starting at zero) after the three
          convolutions of every window encoder

cfg: {"size": odd, "ch": channels, "prev": 0|1, "own2": 0|1, "depth": n}. The stone ages of pid_win2 are not offered here
(closed, notes 2.193). The encoders' outputs start at zero, so a wrapped network equals the plain one at the start.

The shared modules are NOT changed for the short runs: the trainer scripts/pid_seq_train_w3.py calls wrap() here, and
the loaders are started through scripts/pid_w3_run.py, which makes pid_win.wrap() send a cfg with "own2" or "depth"
here (install()). A family that is expanded later gets the permanent dispatch in pid_win.py after a regression check.
"""


def windows3(pos, mov, size, prev=False, own2=False):
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours ->
    cells [B, T, size * size] (the six states of pid_win.windows), ok [B, T];
    pcells, pok: the same around the previous ply's point (states still relative to the mover), or None;
    ocells, ook: the same around the point of the ply before that (the mover's own last move), or None.
    The board part is pid_win2.windows2's (checked against it by selftest)."""
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
    del col, before, ply
    p = size // 2
    W = 19 + 2 * p
    d = torch.arange(size, device=dev)
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # the window's cells from its corner (y - p, x - p) in padded coordinates
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)
    del state

    def cut(cen):
        return board.gather(2, ((cen // 19) * W + cen % 19)[..., None] + off[None, None])

    def around(back):  # the window around the point played `back` plies earlier
        q = torch.full_like(pos, 362)
        q[:, back:] = pos[:, :-back]
        qok = ok & (q < 361)
        return cut(torch.where(qok, q, torch.zeros_like(q))), qok

    cells = cut(torch.where(ok, pos, torch.zeros_like(pos)))
    pcells, pok = around(1) if prev else (None, None)
    ocells, ook = around(2) if own2 else (None, None)
    return cells, ok, pcells, pok, ocells, ook


def wrap(net, cfg):
    """Adds the window encoder(s) to a built m88_seqid.SeqID (in place) and returns it."""
    import torch
    from torch import nn
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))
    prev, own2, depth = bool(cfg.get("prev")), bool(cfg.get("own2")), int(cfg.get("depth", 0))
    assert size % 2 == 1 and 3 <= size <= 19 and 0 <= depth <= 8 and not cfg.get("age") and not cfg.get("exact")
    D = net.gl.out_features
    base = net.__class__

    class Enc(nn.Module):  # pid_win's encoder (the same parameter names), with optional residual blocks
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(6, 16)
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())
            self.more = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1)) for _ in range(depth)])
            for m in self.more:
                nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)
            self.out = nn.Linear(2 * ch, D)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        def forward(self, cells, flat, B, T):
            x = self.emb(cells.reshape(B * T, -1)[flat])
            h = self.conv(x.reshape(-1, size, size, 16).permute(0, 3, 1, 2))
            for m in self.more:
                h = h + m(h)
            c = h.shape[-1] // 2
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)
            res[flat] = f
            return res.reshape(B, T, -1)

    class Win3(nn.Module):
        def __init__(self):
            super().__init__()
            self.own = Enc()
            if prev:
                self.prev = Enc()
            if own2:
                self.own2 = Enc()
            self.size = size

        def forward(self, pos, mov):
            B, T = pos.shape
            with torch.no_grad():
                cells, ok, pcells, pok, ocells, ook = windows3(pos, mov, size, prev, own2)
            res = self.own(cells, ok.reshape(-1), B, T)
            if prev:
                res = res + self.prev(pcells, pok.reshape(-1), B, T)
            if own2:
                res = res + self.own2(ocells, ook.reshape(-1), B, T)
            return res

    class SeqIDW3(base):
        def forward(self, gl, pos, mov, length, subj):
            B, T = pos.shape
            x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) \
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = x + self.win(pos, mov).to(x.dtype)
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)

    net.win = Win3().to(net.gl.weight.device)
    net.__class__ = SeqIDW3
    net.win_cfg = {"size": size, "ch": ch, "prev": int(prev), "own2": int(own2), "depth": depth}
    return net


def install():
    """In THIS process only: pid_win.wrap() sends a cfg with "own2" or "depth" to wrap() above (the loaders call pid_win.wrap
    with the cfg stored in the checkpoint). Every other cfg goes where it went before."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pid_win
    if getattr(pid_win.wrap, "_w3", False):
        return
    plain = pid_win.wrap

    def dispatch(net, cfg):
        return wrap(net, cfg) if (cfg.get("own2") or cfg.get("depth")) else plain(net, cfg)

    dispatch._w3 = True
    pid_win.wrap = dispatch


def selftest():
    """windows3 against pid_win2.windows2 (own and previous-ply windows) and against a plain replay (the third window);
    a wrapped toy module: zero output at the start, gradients reach every encoder, the state loads back."""
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pid_win2
    rng = np.random.default_rng(0)
    B, T, size = 6, 90, 9
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 60:] = 362; pos[2, 7] = pos[2, 3]; pos[3, 10] = 362  # a pass, padding, a point played twice, a hidden ply
    mov = np.tile(np.arange(T) % 2, (B, 1))
    tp, tm = torch.as_tensor(pos), torch.as_tensor(mov)
    cells, ok, pcells, pok, ocells, ook = [x.numpy() for x in windows3(tp, tm, size, True, True)]
    c2, ok2, _a, p2, _q, pok2 = pid_win2.windows2(tp, tm, size, False, True)
    same = int((cells != c2.numpy()).sum() + (ok != ok2.numpy()).sum() + (pcells != p2.numpy()).sum() + (pok != pok2.numpy()).sum())
    c1, ok1, a1, b1, d1, e1 = windows3(tp, tm, size)
    assert a1 is None and b1 is None and d1 is None and e1 is None and int((c1.numpy() != cells).sum()) == 0
    p = size // 2
    bad = {"ocells": 0, "ook": 0}
    n = 0
    for b in range(B):
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)
        for t in range(T):
            has = pos[b, t] < 361
            haso = bool(has and t > 1 and pos[b, t - 2] < 361)
            bad["ook"] += int(bool(ook[b, t]) != haso)
            if haso:
                y, x = divmod(int(pos[b, t - 2]), 19)
                for i in range(size):
                    for j in range(size):
                        yy, xx = y - p + i, x - p + j
                        if not (0 <= yy < 19 and 0 <= xx < 19):
                            want = 3
                        elif board[yy, xx] == 0:
                            want = 0
                        else:
                            dd = (t + 1) - when[yy, xx]
                            want = 4 if dd == 1 else 5 if dd == 2 else (1 if board[yy, xx] - 1 == mov[b, t] else 2)
                        bad["ocells"] += int(ocells[b, t, i * size + j] != want)
                        n += 1
            if has:
                y, x = divmod(int(pos[b, t]), 19)
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1
    print(f"windows3 against pid_win2.windows2: {same} differences; the third window against a plain replay: differing {bad} of {n} cells")

    class Toy(torch.nn.Module):  # the parts wrap() touches
        def __init__(self):
            super().__init__()
            self.gl = torch.nn.Linear(4, 32)

    rc = 0 if same == 0 and not any(bad.values()) else 1
    for cfg in ({"size": 9, "ch": 64, "prev": 1, "own2": 1}, {"size": 9, "ch": 128, "prev": 1}, {"size": 9, "ch": 64, "prev": 1, "depth": 2}):
        torch.manual_seed(0)
        net = wrap(Toy(), dict(cfg))
        out = net.win(tp, tm)
        zero = float(out.abs().max())
        for e in net.win.children():  # move every layer that starts at zero off zero, then every parameter must get a gradient
            torch.nn.init.normal_(e.out.weight, std=0.02)
            for m in e.more:
                torch.nn.init.normal_(m[2].weight, std=0.02)
        net.win(tp, tm).square().sum().backward()
        nograd = [k for k, v in net.win.named_parameters() if v.grad is None or float(v.grad.abs().sum()) == 0]
        net2 = wrap(Toy(), dict(cfg))
        net2.win.load_state_dict(net.win.state_dict())
        back = float((net2.win(tp, tm) - net.win(tp, tm)).abs().max())
        print(f"cfg {net.win_cfg}: {sum(v.numel() for v in net.win.parameters())} parameters, output at the start {zero:g} (want 0), "
              f"parameters without gradient {nograd or 'none'}, reload difference {back:g}")
        rc |= int(zero != 0 or bool(nograd) or back != 0)
    install()  # the loaders' path: pid_win.wrap() with the cfg stored in a checkpoint
    import pid_win
    names = [type(pid_win.wrap(Toy(), dict(c)).win).__name__ for c in ({"size": 9, "ch": 64, "prev": 1, "own2": 1}, {"size": 9, "ch": 64, "prev": 1, "depth": 2},
                                                                    {"size": 9, "ch": 128, "age": 0, "prev": 1}, {"size": 9, "ch": 64})]
    print(f"dispatch after install(): {names} (want ['Win3', 'Win3', 'Win2', 'Win'])")
    rc |= int(names != ["Win3", "Win3", "Win2", "Win"])
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
