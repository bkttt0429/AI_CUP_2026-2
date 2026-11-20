#!/usr/bin/env python
"""A network built for the near misses (METHOD_NOTES 2.144): the PAIR network.

The sequence networks turn every game into one vector on its own and compare vectors; the remaining errors are 1-game
questions whose owner sits at rank 2-5 behind the same few look-alikes, and every network group makes the same ones
(notes 2.139). This network never sees a game alone: it reads the QUESTION game and ONE reference game of a candidate
together (the first tq plies of each, a segment mark on every ply) and answers "the same player?" with one number; a
candidate's score is the mean over his reference games. It is trained on exactly the cases that go wrong:
  - an anchor player, one of his games as the question, `refs` other games of his as the right candidate;
  - k - 1 wrong candidates: for a competition player drawn from his `nn_top` nearest competition players by a teacher
    network's prototypes (look-alikes; nn_top=0: random players, the control), for an external account random external
    accounts (two accounts may be one person, so no look-alikes there);
  - the loss ranks the right candidate first among the k (cross-entropy over the candidates' mean pair scores).
It starts from a trained sequence network (init), whose weights already read a game; the segment marks and the scoring
head are new. At test time it only re-ranks the first few candidates of the fused score (scripts/pid_pair_eval.py).
Competition folds 1-4 (never fold 0) and the external accounts given; no official game is read. `val` competition players
are kept out of the pair training (as question and as candidates) and scored every 2000 steps: right candidate first
among k look-alikes, next to the teacher's own answer on the same episodes (the teacher was trained on these players, so
its number is flattering).
  python -I <this> --ext-ply ... --ext-select ... --run "gpu=0 tag=pa" --run "gpu=1 tag=pb seed=2 init=ckpt_-1_o22Ss1e.pt"
From the AI_CUP_2026-1 checkout. Checkpoints <out>/pair_<tag>.pt (EMA weights), logs logs/nano4/pair_<tag>-<job>.log.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pid_seq_train as ST  # noqa: E402  (the game loader and the rank session's model module, read-only)

M = ST.M
DEFAULTS = dict(gpu="0", tag="pa", init="ckpt_-1_o22Ss0e.pt", steps=16000, anchors=48, k=8, refs=4, tq=128, lr=1e-4, warm=300, mask=0.3, nn_top=64, own_share=0.5, seed=1, ema=0.999, val=300, min_games=8, ext_nn=0, ext_skip=2)


def sym_table():
    """[8, 363]: where a point goes under each board symmetry (361 = pass and 362 = padding stay)."""
    yy, xx = np.divmod(np.arange(361), 19)
    maps = [(xx, yy), (18 - xx, yy), (xx, 18 - yy), (18 - xx, 18 - yy), (yy, xx), (18 - yy, xx), (yy, 18 - xx), (18 - yy, 18 - xx)]
    return np.stack([np.concatenate([b_ * 19 + a_, [361, 362]]) for a_, b_ in maps]).astype(np.int64)


def build_pair(base):
    import torch

    class PairID(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
            d = b.cls.shape[-1]
            self.seg = torch.nn.Parameter(torch.zeros(2, d))
            self.score = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1))
            self.pair_scale = torch.nn.Parameter(torch.tensor(math.log(5.0)))

        def tok(self, gl, pos, mov, subj):
            b = self.b
            t = pos.shape[1]
            return b.norm(b.gl((gl.float() - b.mu) / b.sd) + b.pos(pos) + b.side((mov == subj[:, None]).long()) + b.col(mov)
                          + b.tpos(torch.arange(t, device=pos.device))[None] + b.dprev(b.disp(pos, 1)) + b.down(b.disp(pos, 2)))

        def forward(self, q, r, rep=1):
            """q, r: (gl, pos, mov, length, subj) of the question games and of the reference games; every question game is
            paired with `rep` consecutive reference games. One score per reference game."""
            b = self.b
            xa = self.tok(q[0], q[1], q[2], q[4]) + self.seg[0]
            xb = self.tok(r[0], r[1], r[2], r[4]) + self.seg[1]
            la = q[3]
            if rep != 1:
                xa, la = xa.repeat_interleave(rep, dim=0), la.repeat_interleave(rep, dim=0)
            n, dev = xb.shape[0], xb.device
            pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=dev), torch.arange(xa.shape[1], device=dev)[None] >= la[:, None],
                             torch.arange(xb.shape[1], device=dev)[None] >= r[3][:, None]], dim=1)
            h = b.enc(torch.cat([b.cls.expand(n, -1, -1), xa, xb], dim=1), src_key_padding_mask=pad)[:, 0]
            return self.score(h).squeeze(-1).float()
    return PairID(base)


def _seq_model(arch):
    arch0 = (M.D, M.LAYERS, M.HEADS, M.FFN)
    M.D, M.LAYERS, M.HEADS, M.FFN = arch
    net = M.build_model()
    M.D, M.LAYERS, M.HEADS, M.FFN = arch0
    return net


def load_base(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    assert not st.get("sup"), "a plain-input network is expected as the start (no supervised features are read here)"
    arch = list(st.get("arch", (M.D, M.LAYERS, M.HEADS, M.FFN)))
    net = _seq_model(arch)
    net.load_state_dict(st["best_state"] if "best_state" in st else st["net"])
    return net, arch


def load_pair(path):
    """(pair network on the CPU, its training settings) of a checkpoint written by this script."""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    net = build_pair(_seq_model(st["arch"]))
    net.load_state_dict(st["state"])
    return net, st["cfg"]


def pack(A, rows, tq, dev, sym=None, sy=None, mask=0.0, mu=None):
    """(gl, pos, mov, length, subj) of the first tq plies of the games `rows` of the arrays A, on the device."""
    import torch
    rows = np.asarray(rows, np.int64)
    gl = torch.as_tensor(A["GL"][rows, :tq]).to(dev)
    if mask > 0:
        hit = torch.rand(len(rows), tq, device=dev) < torch.rand(len(rows), 1, device=dev) * mask
        gl = torch.where(hit[..., None], mu.to(gl.dtype)[None, None], gl)
    pos = A["POS"][rows, :tq].astype(np.int64)
    if sym is not None:
        pos = sym[np.asarray(sy)[:, None], pos]
    t = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    return gl, t(pos), t(A["MOV"][rows, :tq].astype(np.int64)), t(np.minimum(A["LEN"][rows].astype(np.int64), tq)), t(A["SUBJ"][rows].astype(np.int64))


def episode_logits(net, A, q_rows, r_rows, tq, dev, sym=None, sy=None, mask=0.0):
    """q_rows [n], r_rows [n, k, refs] -> the candidates' logits [n, k] (scaled mean pair scores; the right one is column 0)."""
    import torch
    n, k, refs = r_rows.shape
    mu = net.b.mu
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s = net(pack(A, q_rows, tq, dev, sym, sy, mask, mu), pack(A, r_rows.reshape(-1), tq, dev, sym, None if sy is None else np.repeat(sy, k * refs), mask, mu), rep=k * refs)
    return net.pair_scale.exp() * s.reshape(n, k, refs).mean(2)


def train(G, c, out, ckpt_dir, log):
    import torch
    import torch.nn.functional as F
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c["gpu"])
    dev = "cuda"
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(5000 + c["seed"])
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    held = set(json.load(open(M.M82 / "held_keys.json")))
    players, owner = np.unique(G["pid"], return_inverse=True)
    C, fold = len(players), G["fold"]
    is_ext = np.char.startswith(players.astype(str), "x:")
    use = ~np.array([k in held for k in G["key"]]) & (fold != 0)
    cnt = np.bincount(owner[use], minlength=C)
    A_, K, R, tq = c["anchors"], c["k"], c["refs"], c["tq"]
    need = max(R + 1, c["min_games"])
    own_all = np.flatnonzero(~is_ext & (cnt >= need))
    tp_ext = np.flatnonzero(is_ext & (cnt >= need))
    W = int(cnt.max())
    idx = np.flatnonzero(use)
    o = owner[idx]
    order = np.argsort(o, kind="stable")
    idx, o = idx[order], o[order]
    tab = np.full((C, W), -1, np.int64)
    tab[o, np.arange(len(o)) - np.searchsorted(o, o, side="left")] = idx
    base, arch = load_base(Path(ckpt_dir) / c["init"])
    base = base.to(dev)
    # the players kept out of the pair training: the same for every network of every job
    is_val = np.zeros(len(own_all), bool)
    is_val[np.random.default_rng(20261116).permutation(len(own_all))[:c["val"]]] = True
    tp_own = own_all[~is_val]
    log(f"pair network {c['tag']}: start {c['init']} (arch {arch}); {len(tp_own)} competition players (folds {sorted(set(fold[use & ~is_ext[owner]].tolist()))}) + {int(is_val.sum())} kept out, {len(tp_ext)} external accounts; "
        f"{A_} questions x {K} candidates x {R} reference games per step, first {tq} plies of a game; GPU {torch.cuda.get_device_name(0)}")

    def draw(p, m, r_=rng):
        return tab[p, r_.permutation(cnt[p])[:m]]

    # the teacher's prototypes of the competition players (at most 48 games each): who looks like whom
    base.eval()
    rp = np.random.default_rng(99)
    rows_own = np.concatenate([draw(p, 48, rp) for p in own_all])
    slot = np.repeat(np.arange(len(own_all)), np.minimum(cnt[own_all], 48))
    od = np.argsort(rows_own)
    rows_own, slot = rows_own[od], slot[od]

    def embed(rows):
        out_ = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 2048):
                r = rows[i:i + 2048]
                out_.append(base(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float())
        return torch.cat(out_)

    proto = torch.zeros(len(own_all), M.D_OUT, device=dev)
    for i in range(0, len(rows_own), 65536):
        proto.index_add_(0, tb(slot[i:i + 65536]), embed(rows_own[i:i + 65536]))
    sim = F.normalize(proto, dim=-1) @ F.normalize(proto, dim=-1).T
    sim.fill_diagonal_(-2.0)
    sim[:, tb(np.flatnonzero(is_val))] = -2.0  # a kept-out player is nobody's wrong candidate either
    top = min(max(c["nn_top"], K - 1), len(own_all) - int(is_val.sum()) - 1)
    near_v, near = sim.topk(top, dim=1)
    near = own_all[near.cpu().numpy()]  # [slot, top] player indices, nearest first
    log(f"teacher prototypes of {len(own_all)} players from {len(rows_own)} games; cosine to the nearest player: median {float(near_v[:, 0].median()):.3f}, to the {top}th {float(near_v[:, -1].median()):.3f}")
    slot_of = np.full(C, -1, np.int64)
    slot_of[own_all] = np.arange(len(own_all))

    def wrong(p, r_=rng):
        if c["nn_top"] > 0:
            return r_.choice(near[slot_of[p]], K - 1, replace=False)
        w = r_.choice(tp_own, K, replace=False)
        return w[w != p][:K - 1]

    # the kept-out players' episodes, fixed (always against their K - 1 NEAREST players); the teacher's answer on them
    rv = np.random.default_rng(7)
    val_p = own_all[is_val]
    vq, vr = np.zeros(len(val_p), np.int64), np.zeros((len(val_p), K, R), np.int64)
    for i, p in enumerate(val_p):
        g = draw(p, R + 1, rv)
        vq[i], vr[i, 0] = g[0], g[1:]
        for j, w in enumerate(near[slot_of[p]][:K - 1]):
            vr[i, j + 1] = draw(w, R, rv)
    if len(val_p):
        zq, zr = embed(vq), embed(vr.reshape(-1)).reshape(len(val_p), K, R, -1)
        t_hit = ((zq[:, None] * F.normalize(zr.sum(2), dim=-1)).sum(-1).argmax(1) == 0).float().mean()
        log(f"kept-out players, {len(val_p)} fixed episodes (the {K - 1} nearest players as wrong candidates): the teacher ranks the right candidate first in {float(t_hit):.3f} (whole games; it was trained on these players)")
        del zq, zr
    del sim, proto
    # ext_nn > 0: the external accounts get look-alikes too (116,000 identities instead of 5,900): the nearest accounts by
    # the teacher's prototypes of at most 16 games each, without the first ext_skip (two accounts may be one person)
    near_ext = slot_x_of = None
    if c["ext_nn"] > 0 and len(tp_ext) > 4 * c["ext_nn"]:
        assert c["ext_nn"] - c["ext_skip"] >= K - 1
        rows_x = np.concatenate([draw(p, 16, rp) for p in tp_ext])
        slot_x = np.repeat(np.arange(len(tp_ext)), np.minimum(cnt[tp_ext], 16))
        od = np.argsort(rows_x)
        rows_x, slot_x = rows_x[od], slot_x[od]
        px = torch.zeros(len(tp_ext), M.D_OUT, device=dev)
        for i in range(0, len(rows_x), 65536):
            px.index_add_(0, tb(slot_x[i:i + 65536]), embed(rows_x[i:i + 65536]))
        px = F.normalize(px, dim=-1)
        near_ext = np.zeros((len(tp_ext), c["ext_nn"]), np.int64)
        v_a, v_b, v_c = [], [], []
        for i in range(0, len(tp_ext), 4096):
            s_ = px[i:i + 4096] @ px.T
            s_[torch.arange(s_.shape[0], device=dev), torch.arange(i, i + s_.shape[0], device=dev)] = -2.0
            v_, j_ = s_.topk(c["ext_nn"], dim=1)
            near_ext[i:i + 4096] = j_.cpu().numpy()
            v_a.append(v_[:, 0]); v_b.append(v_[:, c["ext_skip"]]); v_c.append(v_[:, -1])
        log(f"external look-alikes: prototypes of {len(tp_ext)} accounts from {len(rows_x)} games; cosine to the nearest account: median {float(torch.cat(v_a).median()):.3f} (above 0.95: {float((torch.cat(v_a) > 0.95).float().mean()):.4f}), "
            f"to the first one used (after {c['ext_skip']} left out) {float(torch.cat(v_b).median()):.3f}, to the {c['ext_nn']}th {float(torch.cat(v_c).median()):.3f}")
        slot_x_of = np.full(C, -1, np.int64)
        slot_x_of[tp_ext] = np.arange(len(tp_ext))
        del px, s_

    def validate(model):
        model.eval()
        hit = []
        with torch.no_grad():
            for i in range(0, len(val_p), 16):
                hit.append(episode_logits(model, G, vq[i:i + 16], vr[i:i + 16], tq, dev).argmax(1) == 0)
        return float(torch.cat(hit).float().mean())

    net = build_pair(base).to(dev)
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=0.05)
    steps = c["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / c["warm"]) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    sym = sym_table()
    p_own = cnt[tp_own] / cnt[tp_own].sum()  # competition players drawn by games (the standing recipe)
    n_own = int(round(A_ * c["own_share"]))
    if n_own < A_ and len(tp_ext) < 2 * K + A_:  # a start-up check on a few parts of the games may hold no external account
        log(f"only {len(tp_ext)} external accounts: every question is a competition player's")
        n_own = A_
    net.train()
    tl = ta = to = 0.0
    t_last, every = time.time(), 250
    tgt = torch.zeros(A_, dtype=torch.long, device=dev)
    hist = []
    for step in range(steps):
        anchors = np.concatenate([rng.choice(tp_own, n_own, replace=False, p=p_own), rng.choice(tp_ext, A_ - n_own, replace=False)])
        q_rows, r_rows = np.zeros(A_, np.int64), np.zeros((A_, K, R), np.int64)
        for i, p in enumerate(anchors):
            g = draw(p, R + 1)
            q_rows[i], r_rows[i, 0] = g[0], g[1:]
            if i < n_own:
                others = wrong(p)
            elif near_ext is not None:
                others = tp_ext[rng.choice(near_ext[slot_x_of[p]][c["ext_skip"]:], K - 1, replace=False)]
            else:
                others = rng.choice(tp_ext, K, replace=False)
                others = others[others != p][:K - 1]
            for j, w in enumerate(others):
                r_rows[i, j + 1] = draw(w, R)
        sy = rng.integers(8, size=A_)  # one board symmetry for a question and everything it is compared with
        logits = episode_logits(net, G, q_rows, r_rows, tq, dev, sym, sy, c["mask"])
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(c["ema"], (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
            hit_ = (logits.argmax(1) == 0).float()
        tl += float(loss.detach())
        ta += float(hit_.mean())
        to += float(hit_[:n_own].mean()) if n_own else 0.0
        if (step + 1) % every == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, right candidate first {ta / every:.3f} of {K} (competition questions {to / every:.3f}), scale {float(net.pair_scale.exp()):.1f}, {(time.time() - t_last) / every:.3f} s/step")
            tl = ta = to = 0.0
            t_last = time.time()
        if len(val_p) and ((step + 1) % 2000 == 0 or step + 1 == steps):
            v = validate(ema)
            net.train()
            hist.append([step + 1, v])
            log(f"  kept-out players at step {step + 1}: right candidate first {v:.3f} of {K}")
            t_last = time.time()
    dest = Path(out) / f"pair_{c['tag']}.pt"
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save({"state": {k: v.detach().cpu() for k, v in ema.state_dict().items()}, "arch": arch, "cfg": c, "val": hist}, tmp)
    os.replace(tmp, dest)
    log(f"wrote {dest}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ext-ply", nargs="*", default=None)
    ap.add_argument("--ext-select", nargs="*", default=None)
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--smoke", action="store_true", help="a few parts of the games, for a start-up check")
    ap.add_argument("--run", action="append", required=True, help='"gpu=0 tag=pa key=value ..." - one network; keys: ' + " ".join(DEFAULTS))
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    runs = []
    for spec in a.run:
        c = dict(DEFAULTS)
        for kv in spec.split():
            k, v = kv.split("=", 1)
            assert k in DEFAULTS, f"unknown key {k}"
            c[k] = type(DEFAULTS[k])(v)
        runs.append(c)
    assert len({c["tag"] for c in runs}) == len(runs), "two networks with one tag"
    for c in runs:
        assert (Path(a.ckpt_dir) / c["init"]).exists(), f"no start network {c['init']}"
    ST.SHARED_ANON = True  # the game arrays live in anonymous shared memory: the forked trainers read the same pages
    G = ST.load_games(a.smoke, log, False, a.ext_ply, a.ext_select)
    for k in ST.SHARED:
        G[k].flags.writeable = False
    pids = {}
    for c in runs:
        sys.stdout.flush()
        pid = os.fork()
        if pid == 0:
            rc = 1
            try:
                fd = os.open(f"logs/nano4/pair_{c['tag']}-{os.environ.get('SLURM_JOB_ID', 'local')}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
                os.dup2(fd, 1); os.dup2(fd, 2)
                sys.stdout = os.fdopen(1, "w", buffering=1); sys.stderr = os.fdopen(2, "w", buffering=1)
                t1 = time.time()
                rc = train(G, c, a.out, a.ckpt_dir, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True))
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(int(rc or 0))
        pids[pid] = c["tag"]
        log(f"trainer {pid} on GPU {c['gpu']}: {c}")
    failed = 0
    while pids:
        pid, status = os.wait()
        ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        log(f"trainer {pid} ({pids.pop(pid)}) {'done' if ok else 'FAILED'}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
