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


def load_games(smoke, log, all_folds=False):
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
    with mp.get_context("fork").Pool(12) as pool:
        n1 = pool.map(M.count_part, files, chunksize=8)
        n2 = pool.map(M.count_m90, files2, chunksize=8)
        n_old, N = int(sum(n1)), int(sum(n1)) + int(sum(n2))
        GL, POS, MOV = np.zeros((N, T, 128), np.float16), np.full((N, T), 362, np.int16), np.zeros((N, T), np.int8)
        LEN, SUBJ, keys, pids, a = np.zeros(N, np.int16), np.zeros(N, np.int8), [], [], 0
        for x in pool.imap(M.read_part, files, chunksize=2):
            b = a + len(x[0])
            keys.append(x[0]); pids.append(x[1])
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[2], x[3], x[4], x[5], x[6]
            a = b
        assert a == n_old, (a, n_old)
        for x in pool.imap(M.read_m90, files2, chunksize=2):
            b = a + len(x[0])
            keys.append(x[0])
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[1], x[2], x[3], x[4], x[5]
            a = b
        assert a == N, (a, N)
    key, pid = np.concatenate(keys), np.concatenate(pids)
    fold = np.concatenate([np.full(n, f, np.int8) for n, f in zip(n1, folds)])
    tab = pd.read_csv(AICUP1 / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
    own = {}
    for f_ in sorted((AICUP1 / "data/training").glob("train_*.csv")):
        c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str)
        own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))
    p2 = np.array([own[k] for k in key[n_old:]])
    pid = np.concatenate([pid, p2])
    fold = np.concatenate([fold, np.array([int(tab.get(p, -9)) for p in p2], np.int8)])
    log(f"games loaded: {N} ({n_old} stored of folds {'0-4' if all_folds else '1-4'} + {N - n_old} extracted; {GL.nbytes / 1e9:.1f} GB); games of fold 0: {int((fold == 0).sum())}, {'TRAINED ON (--all-folds)' if all_folds else 'left out everywhere'}")
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
    G = load_games(a.smoke, log, a.all_folds)
    held = set(json.load(open(M.M82 / "held_keys.json")))
    players, owner = np.unique(G["pid"], return_inverse=True)
    C, fold = len(players), G["fold"]
    is_held = np.array([k in held for k in G["key"]])
    allowed = np.ones(len(fold), bool) if a.all_folds else fold != 0
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
    P = M.P_EP if not a.smoke else min(M.P_EP, len(tp))
    log(f"training on {int(use.sum())} games of {len(tp)} players (folds {sorted(set(fold[use].tolist()))}); episodes of {P} players x {M.M_EP} games")
    rng = np.random.default_rng(990 + v + 1000 * a.seed)
    torch.manual_seed(990 + v + 1000 * a.seed)
    M.DROP = a.drop
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
    ev_ok = ~is_held & allowed
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

    net.train()
    ar, tl, ta, t_last, every = np.arange(P), 0.0, 0.0, time.time(), (50 if a.smoke else 500)
    for step in range(start, steps):
        pl = rng.choice(tp, P, replace=False)
        u = rng.random((P, W))
        u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0
        rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M.M_EP]].reshape(-1)
        gl = tb(G["GL"][rows])
        if a.mask > 0:
            hit = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, tb(G["POS"][rows].astype(np.int64)), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
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
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            hist += [evaluate(net, step + 1, "raw"), evaluate(ema, step + 1, "ema")]
            net.train()
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "hist": hist}, state_file)
            t_last = time.time()
    cpu = lambda m: {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}  # noqa: E731
    meta = {"step": steps, "horizon": horizon, "drop": a.drop, "mask": a.mask, "ema": a.ema, "seed": a.seed, "net": v, "players_trained": int(len(tp)), "games": int(use.sum()), "all_folds": bool(a.all_folds)}
    for sfx, model, label in (("", net, "raw"), ("e", ema, "ema")):
        torch.save({"best_state": cpu(model), "net": cpu(model), "best": {**meta, "state": label}, "step": steps}, out / f"ckpt_{v}_{a.tag}{sfx}{'_smoke' if a.smoke else ''}.pt")
    (out / f"train_{name}.json").write_text(json.dumps({**meta, "history": hist}, indent=1) + "\n")
    state_file.unlink(missing_ok=True)
    log(f"net {name}: wrote ckpt_{v}_{a.tag}.pt and ckpt_{v}_{a.tag}e.pt in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
