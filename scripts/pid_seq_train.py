#!/usr/bin/env python
"""A sequence identity network trained for players it has NOT seen (METHOD_NOTES 2.39). Architecture, data and episode
objective are the rank session's M90 (AI_CUP_2026-1 logs/nano4/m88_seqid.py --full; that module is imported read-only for
the model and the part readers). M90's accuracy on unseen players plateaus at 10-30k steps and falls afterwards while the
training players keep being learnt; the official players are all unseen. Changes offered here:
  --steps / --horizon   steps trained / the step where the cosine reaches zero (M90: 60,000; a 20k snapshot = --steps 20000
                        --horizon 60000)
  --drop                dropout (M90 0.1)
  --mask                ply masking in training: per game a share drawn from U(0, mask) of its plies has the trunk vector
                        replaced by the mean vector (zero after the model's standardisation)
  --ema                 decay of an exponential moving average of the weights, saved next to the raw state
Folds: net v trains on the players of folds 1-4 other than v; net -1 on folds 1-4. Fold 0 is not read into training or
evaluation (this repository never uses it); players without a fold train in every net, as in M90. The rank competition's
test games are not read.
  --rank-ep / --rank-ep-share   episodes of players of one rank class or band: hard negatives (METHOD_NOTES 2.55)
  --snap                     further checkpoints at given steps of one long run (METHOD_NOTES 2.51)
  --ext-ply / --ext-select   external players (METHOD_NOTES 2.46): per-ply parts of scripts/pid_ext_ply_extract.py and the
                        select.csv of scripts/ext_fox_select.py (game_key -> account). They train in every net (no fold),
                        are never in the held-out log, and fill --ext-share of every episode's players (default: drawn
                        like any other player).
  --sym-aug             every episode is seen under one random board symmetry (the played points are remapped; the trunk
                        vectors are kept): the network cannot tie a player to absolute corners, only to what is consistent
                        between the games of one episode (METHOD_NOTES 2.48)
  --p-ep                players per episode (M90: 256)
  --ext-share-end       the external share falls linearly from --ext-share to this value over the run (finish on the
                        competition's own players)
  --no-state            no resume file (saves 0.5 GB per network while it trains)
  --all-folds           fold 0 is read and trained on as well: net -1 then sees every training player, as the rank session's
                        net -1 (METHOD_NOTES 2.40). ONLY WITH THE USER'S CONSENT for fold 0; for official files only (no
                        simulation can score such a network). Held-out own games (results/m82/held_keys.json) are never trained on; for net v the ones of fold v
(unseen players) and of the training folds (seen players) are scored every --eval-every steps, as a log only: nothing is
selected on them (fixed step).
Outputs in --out: ckpt_<v>_<tag>.pt (raw state at the last step) and ckpt_<v>_<tag>e.pt (EMA state), both with the state
under "best_state" so that pid_v2_seq_embed.py / pid_v2_seq_finetune.py load them with --ckpt-suffix _<tag> / _<tag>e;
train_<v>_<tag>.json (history); state_<v>_<tag>.pt (resume point, removed at the end).
Run from the AI_CUP_2026-1 checkout: python -I <this> --net 1 --tag r1 ...
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AICUP1 = Path.cwd()
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4")]
import m88_seqid as M  # noqa: E402

T = M.T_MAX
PROTO_GAMES = 16  # evaluation prototypes: at most this many training games per player (fixed choice)


def _windows(pool, fn, files, w):
    """The parts in order, at most w of them waiting in this process at a time. With pool.imap the workers ran ahead of the copy
    into the big arrays and the waiting parts piled up: the processes of 2026-11-13 held 2.3 times the size of their arrays."""
    for i in range(0, len(files), w):
        yield from pool.map(fn, files[i:i + w], chunksize=1)


def _trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the freed part buffers back to the system
    except OSError:
        pass


def _rss_gb():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def load_games(smoke, log, all_folds=False, ext_ply=None, ext_select=None):
    """M90's training games as padded arrays (m88_seqid.load_games without the test part). The worker pool starts before any
    large array exists and every part is written straight into preallocated arrays: the compute nodes run with strict
    overcommit, and a pool forked from a large parent exhausted the node's commit limit in the rank session's first run."""
    files, folds = [], []
    for d in M.DIRS:
        for f in range(0 if all_folds else 1, 5):  # the stored parts of fold 0 are not opened unless --all-folds
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv"))
            fs = fs[:12] if smoke else fs
            files += fs
            folds += [f] * len(fs)
    files2 = [f for k in (0, 1, 2) for f in sorted((AICUP1 / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]
    files2 = files2[:20] if smoke else files2
    ext_own = {}
    if ext_ply:  # the same part layout as M90's (key, LEN, SUBJ, POS, MOV, GLQ; the same quantisation statistics)
        fx = sorted(f for d in ext_ply for f in Path(d).glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz"))
        files2 += fx[:5] if smoke else fx
        import csv
        csv.field_size_limit(10 ** 9)
        for sel in ext_select:
            with open(sel, newline="") as fh:
                ext_own.update((r["game_key"], "x:" + r["account"]) for r in csv.DictReader(fh))
    with mp.get_context("fork").Pool(12) as pool:
        n1 = pool.map(M.count_part, files, chunksize=8)
        n2 = pool.map(M.count_m90, files2, chunksize=8)
        n_old, N = int(sum(n1)), int(sum(n1)) + int(sum(n2))
        GL, POS, MOV = np.zeros((N, T, 128), np.float16), np.full((N, T), 362, np.int16), np.zeros((N, T), np.int8)
        LEN, SUBJ, keys, pids, a = np.zeros(N, np.int16), np.zeros(N, np.int8), [], [], 0
        for x in _windows(pool, M.read_part, files, 12):
            b = a + len(x[0])
            keys.append(x[0]); pids.append(x[1])
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[2], x[3], x[4], x[5], x[6]
            a = b
        assert a == n_old, (a, n_old)
        for x in _windows(pool, M.read_m90, files2, 48):
            b = a + len(x[0])
            keys.append(x[0])
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[1], x[2], x[3], x[4], x[5]
            a = b
        assert a == N, (a, N)
    _trim()
    key, pid = np.concatenate(keys), np.concatenate(pids)
    fold = np.concatenate([np.full(n, f, np.int8) for n, f in zip(n1, folds)])
    tab = pd.read_csv(AICUP1 / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
    own = {}
    for f_ in sorted((AICUP1 / "data/training").glob("train_*.csv")):
        c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str)
        own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))
    own.update(ext_own)
    p2 = np.array([own[k] for k in key[n_old:]])
    pid = np.concatenate([pid, p2])
    fold = np.concatenate([fold, np.array([int(tab.get(p, -9)) for p in p2], np.int8)])
    log(f"games loaded: {N} ({n_old} stored of folds {'0-4' if all_folds else '1-4'} + {N - n_old} extracted; {GL.nbytes / 1e9:.1f} GB; process memory {_rss_gb():.1f} GB); "
        f"games of fold 0: {int((fold == 0).sum())}, {'TRAINED ON (--all-folds)' if all_folds else 'left out everywhere'}")
    return dict(key=key, pid=pid, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, fold=fold)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", type=int, required=True, help="1-4: the fold left out (never sees its players); -1: folds 1-4")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--horizon", type=int, default=None, help="cosine horizon (default: --steps)")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--mask", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ext-ply", nargs="*", default=None, help="one or more directories of external per-ply parts")
    ap.add_argument("--ext-select", nargs="*", default=None, help="the select.csv files that name their accounts")
    ap.add_argument("--ext-share", type=float, default=None, help="share of an episode's players that are external (default: no quota)")
    ap.add_argument("--arch", default=None, help="D,LAYERS,HEADS,FFN of the transformer (M90: 384,8,8,1536); stored in the checkpoint, the loaders rebuild the same network")
    ap.add_argument("--sym-aug", action="store_true")
    ap.add_argument("--p-ep", type=int, default=None)
    ap.add_argument("--ext-share-end", type=float, default=None)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--rank-ep", choices=("class", "band"), default=None,
                    help="hard negatives (METHOD_NOTES 2.55): an episode's competition players come from ONE rank class (class) or from a class and its two neighbours (band), "
                         "its external accounts from one rank folder; classes and folders drawn in proportion to their players")
    ap.add_argument("--rank-ep-share", type=float, default=1.0, help="share of the steps drawn that way; the others are drawn as before")
    ap.add_argument("--snap", nargs="*", type=int, default=[], help="also write the checkpoints at these steps, as ckpt_<net>_<tag>k<step/1000>[e].pt (snapshots of one long run)")
    ap.add_argument("--all-folds", action="store_true", help="read and train on fold 0 as well (needs the user's consent; METHOD_NOTES 2.40)")
    a = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    v, steps = a.net, (300 if a.smoke else a.steps)
    horizon = a.horizon or steps
    eval_every = 100 if a.smoke else a.eval_every
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{v}_{a.tag}{'_smoke' if a.smoke else ''}"
    dev = "cuda"
    log(f"net {name}: steps {steps} (cosine horizon {horizon}), dropout {a.drop}, ply mask up to {a.mask}, EMA {a.ema}, seed {a.seed}; GPU {torch.cuda.get_device_name(0)}")
    G = load_games(a.smoke, log, a.all_folds, a.ext_ply, a.ext_select)
    held = set(json.load(open(M.M82 / "held_keys.json")))
    players, owner = np.unique(G["pid"], return_inverse=True)
    C, fold = len(players), G["fold"]
    is_held = np.array([k in held for k in G["key"]])
    allowed = np.ones(len(fold), bool) if a.all_folds else fold != 0
    is_ext = np.char.startswith(players.astype(str), "x:")  # per player
    use = ~is_held & allowed & (fold != v)
    cnt = np.bincount(owner[use], minlength=C)
    tp = np.flatnonzero(cnt >= M.MIN_GAMES)
    W = int(cnt.max())
    idx = np.flatnonzero(use)
    o = owner[idx]
    order = np.argsort(o, kind="stable")
    idx, o = idx[order], o[order]
    tab = np.full((C, W), -1, np.int64)
    tab[o, np.arange(len(o)) - np.searchsorted(o, o, side="left")] = idx
    P = (a.p_ep or M.P_EP) if not a.smoke else min(M.P_EP, len(tp))
    tp_ext, tp_own = tp[is_ext[tp]], tp[~is_ext[tp]]
    n_ext = 0 if a.ext_share is None or not len(tp_ext) else min(int(round(P * a.ext_share)), len(tp_ext))
    log(f"training on {int(use.sum())} games of {len(tp)} players (folds {sorted(set(fold[use].tolist()))}; external players {len(tp_ext)} with {int((use & is_ext[owner]).sum())} games"
        f"{'' if a.ext_share is None else f', {n_ext} of every episode'}); episodes of {P} players x {M.M_EP} games")
    strata = None
    if a.rank_ep:
        grp = np.array([k.rsplit(":", 1)[0] for k in G["key"]])  # the rank file of a competition game ('5D'), 'x:<folder>' of an external one
        names, gi = np.unique(grp, return_inverse=True)
        tally = np.zeros((C, len(names)), np.int32)
        np.add.at(tally, (owner[use], gi[use]), 1)
        pg = tally.argmax(1)  # a player's group: where most of its training games are filed

        def level(nm):  # strength order inside each population
            if nm.startswith("x:"):
                return (-1 if nm.endswith("k") else 1) * int(nm[2:-1])
            return ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"].index(nm)

        strata = {}
        for pop, ids in (("own", tp_own if a.ext_share is not None else tp), ("ext", tp_ext)):
            if len(ids):
                lv = np.array([level(names[g]) for g in pg[ids]])
                lv = np.searchsorted(np.unique(lv), lv)  # 0 .. L-1, neighbours are adjacent levels that exist
                strata[pop] = (ids, lv, np.bincount(lv) / len(lv))
                log(f"rank episodes ({a.rank_ep}, share {a.rank_ep_share}): {pop} players by level " + ", ".join(f"{names[g]} {int((pg[ids] == g).sum())}" for g in sorted(set(pg[ids].tolist()), key=lambda g: level(names[g]))))

    def rank_draw(pop, k):
        """k players of one level (or band) of a population; widened to the nearest levels when the level is short."""
        ids, lv, share = strata[pop]
        c = rng.choice(len(share), p=share)
        w = 0 if a.rank_ep == "class" else 1
        while (np.abs(lv - c) <= w).sum() < k:
            w += 1
        return rng.choice(ids[np.abs(lv - c) <= w], k, replace=False)

    rng = np.random.default_rng(990 + v + 1000 * a.seed)
    torch.manual_seed(990 + v + 1000 * a.seed)
    M.DROP = a.drop
    if a.arch:
        M.D, M.LAYERS, M.HEADS, M.FFN = (int(x) for x in a.arch.split(","))
    arch = [int(M.D), int(M.LAYERS), int(M.HEADS), int(M.FFN)]
    net = M.build_model().to(dev)
    sample = G["GL"][rng.choice(idx, min(4000, len(idx)), replace=False)].astype(np.float32).reshape(-1, 128)
    sample = sample[np.abs(sample).sum(1) > 0]
    net.mu.copy_(torch.as_tensor(sample.mean(0)))
    net.sd.copy_(torch.as_tensor(sample.std(0) + 1e-6))
    del sample
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=M.LR, weight_decay=M.WD)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / M.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, horizon) / horizon)))
    state_file, start, hist = out / f"state_{name}.pt", 0, []
    if state_file.exists():
        st = torch.load(state_file, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, hist = st["step"], st["hist"]
        rng = np.random.default_rng(990 + v + 1000 * a.seed + 7 * start)
        log(f"resumed at step {start}")
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def embed(model, rows):
        model.eval()
        res = np.zeros((len(rows), M.D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]
                res[i:i + 1024] = model(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
        return res

    # evaluation set: held-out own games of fold v (unseen) and of the training folds (seen); prototypes from non-held games, folds != 0
    ev_ok = ~is_held & allowed & ~is_ext[owner]  # the held-out log keeps the competition's players as its gallery
    e_idx = np.flatnonzero(ev_ok)
    e_o = owner[e_idx]
    e_order = np.argsort(e_o, kind="stable")
    e_idx, e_o = e_idx[e_order], e_o[e_order]
    proto_rows = e_idx[(np.arange(len(e_o)) - np.searchsorted(e_o, e_o, side="left")) < PROTO_GAMES]
    hk = np.flatnonzero(is_held & allowed & (np.bincount(owner[ev_ok], minlength=C)[owner] > 0))
    unseen = fold[hk] == v

    def evaluate(model, step, label):
        E = embed(model, proto_rows)
        Pm = np.zeros((C, M.D_OUT), np.float32)
        np.add.at(Pm, owner[proto_rows], E)
        has = np.linalg.norm(Pm, axis=1) > 0
        Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        S = embed(model, hk) @ Pm.T
        S[:, ~has] = -9.0
        r = (S > S[np.arange(len(hk)), owner[hk]][:, None]).sum(1)
        res = {"step": step, "state": label}
        for nm, s in (("seen", ~unseen), ("unseen", unseen)):
            if s.any():
                res[nm] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}
        log(f"  step {step} [{label}]: held-out own games among {int(has.sum())} players: " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if isinstance(x, dict)))
        return res

    sym_tab = None
    if a.sym_aug:  # [8, 363]: where a point goes under each board symmetry (361 = pass and 362 = padding stay)
        yy, xx = np.divmod(np.arange(361), 19)
        maps = [(xx, yy), (18 - xx, yy), (xx, 18 - yy), (18 - xx, 18 - yy), (yy, xx), (18 - yy, xx), (yy, 18 - xx), (18 - yy, 18 - xx)]
        sym_tab = np.stack([np.concatenate([b_ * 19 + a_, [361, 362]]) for a_, b_ in maps]).astype(np.int64)
    cpu = lambda m: {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}  # noqa: E731
    meta = {"step": steps, "horizon": horizon, "drop": a.drop, "mask": a.mask, "ema": a.ema, "seed": a.seed, "net": v, "players_trained": int(len(tp)), "games": int(use.sum()), "all_folds": bool(a.all_folds), "ext_players": int(len(tp_ext)), "ext_share": a.ext_share, "ext_share_end": a.ext_share_end, "sym_aug": bool(a.sym_aug), "p_ep": int(P), "rank_ep": a.rank_ep, "rank_ep_share": a.rank_ep_share}

    def write_ckpt(tag, at):
        for sfx, model, label in (("", net, "raw"), ("e", ema, "ema")):
            torch.save({"best_state": cpu(model), "net": cpu(model), "best": {**meta, "step": at, "state": label}, "step": at, "arch": arch}, out / f"ckpt_{v}_{tag}{sfx}{'_smoke' if a.smoke else ''}.pt")

    net.train()
    ar, tl, ta, t_last, every = np.arange(P), 0.0, 0.0, time.time(), (50 if a.smoke else 500)
    for step in range(start, steps):
        if a.ext_share is not None and a.ext_share_end is not None:
            n_ext = min(int(round(P * (a.ext_share + (a.ext_share_end - a.ext_share) * step / max(steps - 1, 1)))), len(tp_ext))
        if strata is not None and rng.random() < a.rank_ep_share:
            pl = rank_draw("own", P) if a.ext_share is None or n_ext == 0 else np.concatenate([rank_draw("ext", n_ext), rank_draw("own", P - n_ext)])
        else:
            pl = rng.choice(tp, P, replace=False) if a.ext_share is None else np.concatenate([rng.choice(tp_ext, n_ext, replace=False), rng.choice(tp_own, P - n_ext, replace=False)])
        u = rng.random((P, W))
        u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0
        rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M.M_EP]].reshape(-1)
        gl = tb(G["GL"][rows])
        if a.mask > 0:
            hit = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pos_np = G["POS"][rows].astype(np.int64)
            if sym_tab is not None:
                pos_np = sym_tab[rng.integers(8)][pos_np]
            z = net(gl, tb(pos_np), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float().reshape(P, M.M_EP, M.D_OUT)
        tot, sc = z.sum(1), net.log_scale.exp()
        logits = sc * z.reshape(-1, M.D_OUT) @ F.normalize(tot, dim=-1).T
        own_ = sc * (z * F.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)  # a game against its own player's prototype without itself
        tgt = tb(np.repeat(ar, M.M_EP))
        logits = logits.scatter(1, tgt[:, None], own_[:, None])
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        tl += float(loss.detach())
        ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % every == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, episode top-1 {ta / every:.3f} of {P}, scale {float(sc.detach()):.1f}, {(time.time() - t_last) / every:.2f} s/step")
            tl, ta, t_last = 0.0, 0.0, time.time()
        if step + 1 in a.snap and step + 1 < steps:
            write_ckpt(f"{a.tag}k{(step + 1) // 1000}", step + 1)
            log(f"  snapshot at step {step + 1}: ckpt_{v}_{a.tag}k{(step + 1) // 1000}[e].pt")
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            hist += [evaluate(net, step + 1, "raw"), evaluate(ema, step + 1, "ema")]
            net.train()
            if not a.no_state:
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "hist": hist}, state_file)
            t_last = time.time()
    write_ckpt(a.tag, steps)
    (out / f"train_{name}.json").write_text(json.dumps({**meta, "history": hist}, indent=1) + "\n")
    state_file.unlink(missing_ok=True)
    log(f"net {name}: wrote ckpt_{v}_{a.tag}.pt and ckpt_{v}_{a.tag}e.pt in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
