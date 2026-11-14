#!/usr/bin/env python
"""Pool adaptation of the sequence identity model (METHOD_NOTES 2.31): a short fine-tune of the rank session's network
(AI_CUP_2026-1 logs/nano4/m88_seqid.py) on ONE pool's own candidates (600 players x 10 games, ids given by the organisers), with
the model's own objective: every game is a query against the prototypes of the players in the episode, its own game left out.
No question game enters the fine-tune; a pool is adapted on its own candidates only (Public and Private never mixed).
Why: the network identifies players it was trained on far better than unseen ones (held-out games: 0.21-0.23 against 0.15-0.17
top-1 in the rank session's evaluation); official players are unseen.
  --mode sim --fold v --seed s        the simulated pool of pid_v2_sim2.py (fold v, net v) -> <out>/sim_f<v>_s<s>_t<step>.npz
  --mode official --pool pub2|priv2 --net v   an added official pool under net v -> <out>/official_<pool>_net<v>_t<step>.npz
  --mode official ... --holdout h [--holdout-seed s]   the labelled pseudo-pool of METHOD_NOTES 2.41: h games of every candidate
                                      (aicup_pid.v2.candidate_holdout) are kept out of the fine-tune, like question games
                                      -> <out>/offsim_<pool>_h<h>s<s>_net<v>_t<step>.npz
Each output holds the vectors of every game of the pool (candidates and questions) after <step> fine-tune steps (0 = as is).
Run from the AI_CUP_2026-1 checkout."""
import argparse
import glob
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
import m88_seqid as m88  # noqa: E402


def embed(net, A, idx, batch=1024):
    net.eval()
    out = np.zeros((len(idx), m88.D_OUT), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), batch):
            r = idx[i:i + batch]
            out[i:i + batch] = net(torch.as_tensor(A["GL"][r]).cuda(), torch.as_tensor(A["POS"][r].astype(np.int64)).cuda(), torch.as_tensor(A["MOV"][r].astype(np.int64)).cuda(),
                                   torch.as_tensor(A["LEN"][r].astype(np.int64)).cuda(), torch.as_tensor(A["SUBJ"][r].astype(np.int64)).cuda()).float().cpu().numpy()
    net.train()
    return out


def sym_table():
    """[8, 363]: where a point goes under each board symmetry (361 = pass and 362 = padding stay); as scripts/pid_seq_train.py --sym-aug."""
    yy, xx = np.divmod(np.arange(361), 19)
    maps = [(xx, yy), (18 - xx, yy), (xx, 18 - yy), (18 - xx, 18 - yy), (yy, xx), (18 - yy, xx), (yy, 18 - xx), (18 - yy, 18 - xx)]
    return np.stack([np.concatenate([b_ * 19 + a_, [361, 362]]) for a_, b_ in maps]).astype(np.int16)


def finetune(net, A, owner, steps_out, save, lr, p_ep, m_ep, seed, log, horizon=None, mix=None, warm=50, views=1, sym_aug=False):
    """A: arrays of every pool game; owner [n] candidate index or -1 (question game). Calls save(step, vectors) at steps_out."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    cand = np.flatnonzero(owner >= 0)
    C = int(owner.max()) + 1
    by = [cand[owner[cand] == c] for c in range(C)]
    m_ep = min(m_ep, min(len(b) for b in by))
    p_ep = min(p_ep, C)
    everything = np.arange(len(owner))
    save0 = save
    SYM = sym_table() if views > 1 or sym_aug else None
    rng_sym = np.random.default_rng(seed + 7919)  # its own stream: the episodes are the same with and without the augmentation

    def save(step, vec):  # with --views k: also the vectors of the pool seen under the board symmetries 1 .. k-1 (suffix v<i>)
        save0(step, vec, "")
        for i in range(1, views):
            B = dict(A)
            B["POS"] = SYM[i][A["POS"].astype(np.int64)]
            save0(step, embed(net, B, everything), f"v{i}")
    mix_by = None
    if mix is not None:  # training players in every episode (half of it): keeps the general discrimination while the pool is learnt
        m_own = mix["owner"]
        ids = [c for c in np.unique(m_own) if (m_own == c).sum() >= m_ep]
        mix_by = [np.flatnonzero(m_own == c) for c in ids]
        log(f"  mixing in {len(mix_by)} training players ({len(m_own)} games)")
    if 0 in steps_out:
        save(0, embed(net, A, everything))
    last = max(steps_out)
    if last == 0:
        return
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    hz = horizon or last  # the cosine runs over `horizon` steps even when the fine-tune stops earlier (11-12: the first run's t = 100 came from a 1,000-step schedule)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, hz) / hz)))
    net.train()
    ar, tl, ta, t0 = np.arange(p_ep), 0.0, 0.0, time.time()
    for step in range(last):
        n_pool = p_ep if mix_by is None else p_ep // 2
        pl = rng.choice(C, n_pool, replace=False)
        rows = np.concatenate([rng.choice(by[c], m_ep, replace=False) for c in pl])
        batch = {k: A[k][rows] for k in ("GL", "POS", "MOV", "LEN", "SUBJ")}
        if mix_by is not None:
            mrows = np.concatenate([rng.choice(mix_by[c], m_ep, replace=False) for c in rng.choice(len(mix_by), p_ep - n_pool, replace=False)])
            batch = {k: np.concatenate([batch[k], mix[k][mrows]]) for k in batch}
        if sym_aug:  # one random board symmetry per step, on the played points (as scripts/pid_seq_train.py --sym-aug)
            batch["POS"] = SYM[rng_sym.integers(8)][batch["POS"].astype(np.int64)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(torch.as_tensor(batch["GL"]).cuda(), torch.as_tensor(batch["POS"].astype(np.int64)).cuda(), torch.as_tensor(batch["MOV"].astype(np.int64)).cuda(),
                    torch.as_tensor(batch["LEN"].astype(np.int64)).cuda(), torch.as_tensor(batch["SUBJ"].astype(np.int64)).cuda())
        z = z.float().reshape(p_ep, m_ep, m88.D_OUT)
        tot = z.sum(1)
        sc = net.log_scale.exp()
        logits = sc * z.reshape(-1, m88.D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)  # its own game left out of its player's prototype
        tgt = torch.as_tensor(np.repeat(ar, m_ep)).cuda()
        logits = logits.scatter(1, tgt[:, None], own[:, None])
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % 50 == 0:
            log(f"  step {step + 1}/{last}: loss {tl / 50:.4f}, episode top-1 {ta / 50:.3f} of {p_ep}, {(time.time() - t0) / (step + 1):.2f} s/step")
            tl = ta = 0.0
        if step + 1 in steps_out:
            save(step + 1, embed(net, A, everything))


ARCH0 = (m88.D, m88.LAYERS, m88.HEADS, m88.FFN)
SUP_POOLS = ["/work/u2507115/pid_v2/sup_pools/ply"]  # official pools (scripts/pid_sup_ply_extract.py --pools)
SUP_TRAIN = ["/work/u2507115/pid_v2/probeb/ply", "/work/u2507115/pid_v2/probec/ply"]  # competition training games
SUP_STATS = "/work/u2507115/pid_v2/probeb/stats.npz"


def load_net(ckpt, state="best"):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = st.get("arch", ARCH0)  # a network trained with pid_seq_train.py --arch is rebuilt at its own size
    net = m88.build_model()
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = ARCH0
    net.load_state_dict(st["best_state" if state == "best" else "net"])
    net.sup_cfg = st.get("sup")  # pid_seq_train.py --sup-dims: the input dimensions that carry the supervised features (METHOD_NOTES 2.98)
    return net.cuda()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=("sim", "official", "oldsim"))
    ap.add_argument("--side", choices=("pub", "priv"), default="pub", help="oldsim: which ORIGINAL pool")
    ap.add_argument("--ply-original", default="/work/u2507115/pid_v2/ply_original")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=("pub2", "priv2"), default="pub2")
    ap.add_argument("--net", type=int, default=-1)
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 100, 300, 600, 1000])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--horizon", type=int, default=None, help="length of the cosine schedule (default: the last step asked for)")
    ap.add_argument("--tag", default="", help="suffix of the output names (a recipe label)")
    ap.add_argument("--mix-fold", type=int, default=0, help="mix training players of this fold into every episode (0 = none; the network must not be scored on that fold)")
    ap.add_argument("--p-ep", type=int, default=256)
    ap.add_argument("--m-ep", type=int, default=8)
    ap.add_argument("--ckpt-dir", default=str(AICUP1 / "results/m88"))
    ap.add_argument("--ckpt-suffix", default="", help="checkpoint file ckpt_<v><suffix>.pt")
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--tests-dir", default=str(R2 / "data" / "tests_1111"))
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--state", choices=("best", "last"), default="best", help="the checkpoint's state with the best held-out top-1 on seen players, or the one after its final step")
    ap.add_argument("--holdout", type=int, default=0, help="official mode: games per candidate kept out of the fine-tune (pseudo-pool; 0 = the real pool)")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=1, help="1-8: also write the vectors of the whole pool under that many board symmetries (test-time views of a network "
                                                         "trained with --sym-aug; the same symmetry for every game of the pool; METHOD_NOTES 2.59)")
    ap.add_argument("--sym-aug", action="store_true", help="one random board symmetry per fine-tune step (for networks trained with --sym-aug; METHOD_NOTES 2.69)")
    ap.add_argument("--warm", type=int, default=50, help="steps of the linear learning-rate warm-up (50: a 50-step fine-tune never leaves the warm-up; METHOD_NOTES 2.55)")
    ap.add_argument("--mix-original", default=None, help="official mode: directory of scripts/pid_v2_ply_extract.py --original; the candidates of the ORIGINAL pool of the same side "
                                                           "(pub for pub2, priv for priv2; never the other side) fill half of every episode (METHOD_NOTES 2.54)")
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "sim":
        import aicup_pid
        from aicup_pid import v2
        from aicup_pid.paths import FOLDS_CSV
        from pid_v2_sim2 import build
        games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
        rows = v2.load_side_rows(AICUP1)
        z = np.load(R2 / "data" / "seq" / f"fold{a.fold}.npz")
        g = games[(games.fold == a.fold) & games.game_key.isin(rows.keys()) & games.game_key.isin(set(z["key"].astype(str)))]
        cands, questions, _owner, _block, _sgfs, _info = build(g, a.seed)
        players = sorted(cands)
        own = {k: i for i, p in enumerate(players) for k, _ in cands[p]}
        need = set(own) | {k for q in questions for k, _ in q}
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{a.fold}").glob("games_s*_p*.csv"))]
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
        with mp.get_context("fork").Pool(12) as pool:
            for i in range(0, len(files), 48):
                for key, _pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                    m = np.array([k in need for k in key])
                    if m.any():
                        for name, arr in (("key", key), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):
                            parts[name].append(arr[m])
        A = {k: np.concatenate(x) for k, x in parts.items()}
        if set(A["key"].tolist()) != need:
            raise ValueError(f"{len(need) - len(set(A['key'].tolist()))} pool games without per-ply vectors")
        tag, ckpt = f"sim_f{a.fold}_s{a.seed}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.fold}{a.ckpt_suffix}.pt"
    elif a.mode == "oldsim":  # the simulated pool of aicup_pid.v2.original_sim: fine-tuned on its 10-game candidate sets only -> oldsim_<side>_s<seed>_net<v><tag>_t<step>.npz
        from aicup_pid import v2
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
        for f in sorted(glob.glob(f"{a.ply_original}/ply_*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            key = z["key"].astype(str)
            m = np.array([k.split(":")[2].startswith(a.side + "_g_") for k in key])
            if m.any():
                for name in parts:
                    parts[name].append(key[m] if name == "key" else z[name][m])
        A = {k: np.concatenate(x) for k, x in parts.items()}
        cands, _q, _t, _s = v2.original_sim(Path(a.tests_dir) / f"player_identification_candidates_{'public' if a.side == 'pub' else 'private'}.csv", set(A["key"].tolist()), a.holdout_seed)
        players = sorted(cands)
        own = {k: i for i, p in enumerate(players) for k, _ in cands[p]}
        tag, ckpt = f"oldsim_{a.side}_s{a.holdout_seed}_net{a.net}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.net}{a.ckpt_suffix}.pt"
    else:
        tagname = "public" if a.pool == "pub2" else "private"
        c = pd.read_csv(Path(a.tests_dir) / f"player_identification_candidates2_{tagname}.csv", dtype=str, keep_default_na=False)
        players = sorted(c.player_id.unique())
        pidx = {p: i for i, p in enumerate(players)}
        own = {f"c:{i}:{r.game_id}": pidx[r.player_id] for i, r in enumerate(c.itertuples())}
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
        for f in sorted(glob.glob(f"{a.ply_dir}/ply_*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            key = z["key"].astype(str)
            m = np.array([(k in own) or k.startswith(f"q:{a.pool}_") for k in key])
            if m.any():
                for name in parts:
                    parts[name].append(key[m] if name == "key" else z[name][m])
        A = {k: np.concatenate(x) for k, x in parts.items()}
        tag, ckpt = f"official_{a.pool}_net{a.net}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.net}{a.ckpt_suffix}.pt"
        if a.holdout:
            from aicup_pid import v2
            cands, _q, _ids, _sgfs = v2.official_pool(Path(a.tests_dir) / f"player_identification_candidates2_{tagname}.csv", Path(a.tests_dir) / f"player_identification_test2_{tagname}.csv")
            assert {k for g in cands.values() for k, _ in g} == set(own), "candidate keys differ from aicup_pid.v2.official_pool"
            held = {k for g in v2.candidate_holdout(cands, a.holdout, a.holdout_seed).values() for k, _ in g}
            own = {k: i for k, i in own.items() if k not in held}
            tag = f"offsim_{a.pool}_h{a.holdout}s{a.holdout_seed}_net{a.net}{a.tag}"
    owner = np.array([own.get(k, -1) for k in A["key"]])
    log(f"{tag}: {len(owner)} games ({int((owner >= 0).sum())} candidate games of {len(players)} players, {int((owner < 0).sum())} question games); network {ckpt.name}")
    net = load_net(ckpt, a.state)
    if net.sup_cfg:  # the pool's games get the same supervised features the network was trained on
        assert not a.mix_fold and not a.mix_original, "the mixing options have no supervised features"
        sys.path.insert(0, str(R2 / "scripts"))
        import pid_sup
        stores = SUP_TRAIN if a.mode == "sim" else SUP_POOLS
        idx_s, D_s = pid_sup.dense(pid_sup.load_sup(stores, SUP_STATS), A["key"].tolist(), A["GL"].shape[1])
        if (idx_s < 0).any():
            raise KeyError(f"{int((idx_s < 0).sum())} pool games without supervised per-ply features, e.g. {A['key'][idx_s < 0][:3].tolist()}")
        z_s = D_s[idx_s].astype(np.float32)
        A["GL"] = pid_sup.put(A["GL"].astype(np.float32), z_s, net.mu.float().cpu().numpy(), net.sd.float().cpu().numpy(), tuple(net.sup_cfg["dims"]), bool(net.sup_cfg["only"]))
        log(f"supervised features in input dimensions {net.sup_cfg['dims']}{' (alone)' if net.sup_cfg['only'] else ''} for all {len(idx_s)} games")

    def save(step, vec, view=""):
        np.savez(out / f"{tag}{view}_t{step}.npz", key=A["key"], emb=vec.astype(np.float16), step=np.array(step))
        log(f"  vectors after {step} steps written{' (symmetry view ' + view[1:] + ')' if view else ''}")

    mix = None
    if a.mix_fold:
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{a.mix_fold}").glob("games_s*_p*.csv"))]
        parts = {x: [] for x in ("pid", "GL", "POS", "MOV", "LEN", "SUBJ")}
        with mp.get_context("fork").Pool(12) as pool:
            for i in range(0, len(files), 48):
                for _key, pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                    for name, arr in (("pid", pid), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):
                        parts[name].append(arr)
        mix = {k: np.concatenate(x) for k, x in parts.items()}
        mix["owner"] = np.unique(mix.pop("pid"), return_inverse=True)[1]
    if a.mix_original:
        assert a.mode == "official" and mix is None
        side = a.pool[:-1]  # pub2 -> pub, priv2 -> priv
        c0 = pd.read_csv(Path(a.tests_dir) / f"player_identification_candidates_{'public' if side == 'pub' else 'private'}.csv", dtype=str, usecols=["game_id", "player_id"])
        own0 = {f"c:{i}:{g}": p for i, (g, p) in enumerate(zip(c0.game_id, c0.player_id))}
        parts = {x: [] for x in ("pid", "GL", "POS", "MOV", "LEN", "SUBJ")}
        for f in sorted(glob.glob(f"{a.mix_original}/ply_*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f)
            key = z["key"].astype(str)
            m = np.array([k.split(":")[2].startswith(side + "_g_") for k in key])
            if m.any():
                parts["pid"].append(np.array([own0[k] for k in key[m]]))
                for name in ("GL", "POS", "MOV", "LEN", "SUBJ"):
                    parts[name].append(z[name][m])
        mix = {k: np.concatenate(x) for k, x in parts.items()}
        assert all(p.startswith(side + "_p_") for p in np.unique(mix["pid"]))
        mix["owner"] = np.unique(mix.pop("pid"), return_inverse=True)[1]
        log(f"original pool '{side}': {len(mix['owner'])} candidate games of {int(mix['owner'].max()) + 1} candidates in every episode's second half")
    finetune(net, A, owner, sorted(set(a.steps)), save, a.lr, a.p_ep, a.m_ep, 1000 + a.seed, log, a.horizon, mix, a.warm, a.views, a.sym_aug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
