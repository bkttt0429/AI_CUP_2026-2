"""METHOD_NOTES 2.180: the window input of pid_win.py with the EXACT board (captured stones are removed).

pid_win.py builds the board of every ply from the moves alone, so a captured stone stays until something is played on its
point again: on 3,000 competition games 25.7 % of the plies have a wrong cell in their 9 x 9 window (55 % after move 180;
scripts/pid_probe_captures.py). Here every game carries, per ply, the number of the ply that CAPTURES the stone placed at it
(0 = it stays; a replay of the moves on the CPU, once per game), and the window builder drops the stones whose capture lies
before the current ply. Setup (handicap) stones are still unknown to the board: they are not in the plies' points.

How the removal plies reach the network without another argument in every caller: they travel inside the colour tensor,
    packed = colour + 4 * removal ply + FLAG          (pack_mov; FLAG marks a packed tensor)
and the wrapped network's forward unpacks them first. A network of this kind refuses colours that are not packed, so a
loader that forgot to pack cannot silently fall back to the inexact board.

wrap(net, cfg) is called through pid_win.wrap when cfg has "exact": the state-dict names are the ones of pid_win.py.
"""
import numpy as np

FLAG = 1 << 13
_NB = None
_G = None


def _neigh():
    global _NB
    if _NB is None:
        nb = []
        for p in range(361):
            y, x = divmod(p, 19)
            nb.append(tuple(q for q, inside in (((y - 1) * 19 + x, y > 0), ((y + 1) * 19 + x, y < 18), (y * 19 + x - 1, x > 0), (y * 19 + x + 1, x < 18)) if inside))
        _NB = nb
    return _NB


def _dead_group(board, q, colour, NB):
    """The stones of the group at q when it has no liberty, else None."""
    seen = {q}
    stack = [q]
    while stack:
        s = stack.pop()
        for r in NB[s]:
            b = board[r]
            if b == 0:
                return None
            if b == colour and r not in seen:
                seen.add(r)
                stack.append(r)
    return seen


def removal_plies(pos, mov, n):
    """One game: pos [T] points (y * 19 + x; 361 pass, 362 padding), mov [T] colours (0 / 1), n plies -> [T] int16, for every
    ply the 1-based number of the ply at which its stone is captured (0 = never, also for passes and padding)."""
    NB = _neigh()
    rem = np.zeros(len(pos), np.int16)
    board = [0] * 361  # 0 empty, colour + 1
    placed = [0] * 361  # the 1-based ply number of the stone on a point
    pl, ml = pos[:n].tolist(), mov[:n].tolist()
    for t in range(n):
        p = pl[t]
        if p >= 361:
            continue
        c = ml[t] + 1
        board[p] = c
        placed[p] = t + 1
        o = 3 - c
        for q in NB[p]:
            if board[q] == o:
                dead = _dead_group(board, q, o, NB)
                if dead:
                    for s in dead:
                        rem[placed[s] - 1] = t + 1
                        board[s] = 0
                        placed[s] = 0
        dead = _dead_group(board, p, c, NB)  # a move without liberties after the captures (no legal game has one)
        if dead:
            for s in dead:
                rem[placed[s] - 1] = t + 1
                board[s] = 0
                placed[s] = 0
    return rem


def _range(task):
    a, b = task
    POS, MOV, LEN, REM = _G
    for i in range(a, b):
        REM[i] = removal_plies(POS[i], MOV[i], int(LEN[i]))
    return b - a


def fill_removals(POS, MOV, LEN, REM, workers=1, log=None):
    """REM [N, T] int16 (zeros) <- the removal plies of every game. With workers > 1 REM must be a shared mapping created
    before this call (the forked workers write into it)."""
    global _G
    import time
    t0 = time.time()
    _G = (POS, MOV, LEN, REM)
    n = len(LEN)
    if workers <= 1 or n < 4000:
        _range((0, n))
    else:
        import multiprocessing as mp
        step = 2000
        done = 0
        with mp.get_context("fork").Pool(workers) as pool:
            for k in pool.imap_unordered(_range, [(a, min(n, a + step)) for a in range(0, n, step)]):
                done += k
                if log is not None and done % 400000 < step:
                    log(f"  removal plies: {done} of {n} games ({done / max(time.time() - t0, 1e-9):.0f} games/s)")
    _G = None
    if log is not None:
        log(f"removal plies of {n} games in {time.time() - t0:.0f} s ({workers} workers); games with a capture {float((np.asarray(REM[:min(n, 200000)]) > 0).any(1).mean()):.3f} (first 200,000)")
    return REM


def pack_mov(POS, MOV, LEN, REM=None):
    """Colours with the removal plies packed in ([N, T] int16; see the module text). REM is computed here when not given."""
    if REM is None:
        REM = fill_removals(POS, MOV, LEN, np.zeros(MOV.shape, np.int16))
    return (np.asarray(MOV).astype(np.int16) + 4 * np.asarray(REM).astype(np.int16) + FLAG).astype(np.int16)


def windows(pos, mov, rem, size):
    """pid_win.windows with the captured stones removed: rem [B, T] the removal ply of every ply's stone (0 = stays)."""
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
    gone = rem.to(torch.int16)[:, None, :].expand(B, T, T).gather(2, (ply - 1).clamp(min=0))  # when the stone on the point is captured
    ply = torch.where((gone > 0) & (gone < now), torch.zeros_like(ply), ply)  # captured before the current ply: the point is empty
    del gone
    state = torch.where(ply == 0, 0, torch.where(col == mov[..., None], 1, 2))
    state = torch.where((ply > 0) & (ply == now - 1), 4, state)
    state = torch.where((ply > 0) & (ply == now - 2), 5, state)
    del ply, col, before
    p = size // 2
    W = 19 + 2 * p
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)
    cen = torch.where(ok, pos, torch.zeros_like(pos))
    d = torch.arange(size, device=dev)
    off = (d[:, None] * W + d[None, :]).reshape(-1)
    idx = ((cen // 19) * W + cen % 19)[..., None] + off[None, None]
    return board.gather(2, idx), ok


def wrap(net, cfg):
    """As pid_win.wrap (same parameter names), with the exact board. cfg: {"size": odd, "ch": channels, "exact": True}."""
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

        def forward(self, pos, mov, rem):
            B, T = pos.shape
            with torch.no_grad():
                cells, ok = windows(pos, mov, rem, self.size)
            flat = ok.reshape(-1)
            x = self.emb(cells.reshape(B * T, -1)[flat]).reshape(-1, self.size, self.size, 16).permute(0, 3, 1, 2)
            h = self.conv(x)
            c = h.shape[-1] // 2
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)
            res[flat] = f
            return res.reshape(B, T, -1)

    class SeqIDWX(base):
        def forward(self, gl, pos, mov, length, subj):
            if not bool((mov[:, 0] >= FLAG).all()):
                raise ValueError("a network with the exact window got colours without the packed removal plies (pid_winx.pack_mov)")
            rem = (mov - FLAG) >> 2
            mov = mov & 1
            B, T = pos.shape
            x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) \
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = x + self.win(pos, mov, rem).to(x.dtype)
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)

    net.win = Win().to(net.gl.weight.device)
    net.__class__ = SeqIDWX
    net.win_cfg = {"size": size, "ch": ch, "exact": True}
    net.win_exact = True
    return net


def selftest(games=40, seed=0):
    """(1) removal_plies against an independent brute-force replay on random games with many captures; (2) the window builder
    against that replay's boards; (3) without captures the builder equals pid_win.windows."""
    import sys
    from pathlib import Path
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not put the script's folder on the path
    import pid_win
    rng = np.random.default_rng(seed)
    T, size = 220, 9
    p_ = size // 2
    POS = np.full((games, T), 362, np.int16); MOV = np.zeros((games, T), np.int8); LEN = np.zeros(games, np.int32)
    boards = []  # per game: list over plies of the exact board BEFORE the ply (0 empty, colour + 1) and the ply numbers on it

    def group(bd, y, x):
        c = bd[y, x]; seen = {(y, x)}; st = [(y, x)]; lib = False
        while st:
            a, b = st.pop()
            for aa, bb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= aa < 19 and 0 <= bb < 19:
                    if bd[aa, bb] == 0:
                        lib = True
                    elif bd[aa, bb] == c and (aa, bb) not in seen:
                        seen.add((aa, bb)); st.append((aa, bb))
        return seen, lib

    want_rem = np.zeros((games, T), np.int16)
    for g in range(games):
        bd = np.zeros((19, 19), int); when = np.zeros((19, 19), int)
        seq = []
        dense = 6 + g % 6  # a small region: many captures
        t = 0
        while t < T - 3:
            if rng.random() < 0.03:
                POS[g, t] = 361; MOV[g, t] = t % 2; seq.append((bd.copy(), when.copy())); t += 1
                continue
            y, x = rng.integers(0, dense, 2)
            if bd[y, x]:
                continue
            c = t % 2 + 1
            trial = bd.copy(); trial[y, x] = c
            dead = []
            for aa, bb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= aa < 19 and 0 <= bb < 19 and trial[aa, bb] == 3 - c:
                    s, lib = group(trial, aa, bb)
                    if not lib:
                        dead += list(s)
            for a, b in dead:
                trial[a, b] = 0
            if not group(trial, y, x)[1]:
                continue  # suicide: not played
            seq.append((bd.copy(), when.copy()))
            POS[g, t] = y * 19 + x; MOV[g, t] = t % 2
            for a, b in set(dead):
                want_rem[g, when[a, b] - 1] = t + 1; when[a, b] = 0
            bd = trial; when[y, x] = t + 1
            t += 1
        LEN[g] = t
        boards.append(seq)
    REM = fill_removals(POS, MOV, LEN, np.zeros((games, T), np.int16))
    bad_rem = int((REM != want_rem).sum())
    caps = int((want_rem > 0).sum())
    packed = pack_mov(POS, MOV, LEN, REM)
    mv = torch.as_tensor(packed.astype(np.int64))
    rem_t, mov_t, pos_t = (mv - FLAG) >> 2, mv & 1, torch.as_tensor(POS.astype(np.int64))
    cells, ok = windows(pos_t, mov_t, rem_t, size)
    cells = cells.numpy()
    bad = total = 0
    for g in range(games):
        for t in range(int(LEN[g])):
            if POS[g, t] >= 361:
                continue
            bd, when = boards[g][t]
            y, x = divmod(int(POS[g, t]), 19)
            for i in range(size):
                for j in range(size):
                    yy, xx = y - p_ + i, x - p_ + j
                    if not (0 <= yy < 19 and 0 <= xx < 19):
                        want = 3
                    elif bd[yy, xx] == 0:
                        want = 0
                    elif when[yy, xx] == t:
                        want = 4
                    elif when[yy, xx] == t - 1:
                        want = 5
                    else:
                        want = 1 if bd[yy, xx] - 1 == MOV[g, t] else 2
                    bad += int(cells[g, t, i * size + j] != want); total += 1
    plain, _ = pid_win.windows(pos_t, mov_t, size)
    same = bool((windows(pos_t, mov_t, torch.zeros_like(rem_t), size)[0] == plain).all())
    differs = float((torch.as_tensor(cells) != plain).any(-1)[torch.as_tensor(POS.astype(np.int64)) < 361].float().mean())
    print(f"removal plies against a brute-force replay: {bad_rem} differing entries ({caps} captured stones in {games} games)")
    print(f"exact window builder against the replay's boards: {bad} differing cells of {total}")
    print(f"with no removals the builder equals pid_win.windows: {same}; windows that differ from the inexact ones in these games: {differs:.3f}")
    return bad_rem == 0 and bad == 0 and same and caps > 100


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
