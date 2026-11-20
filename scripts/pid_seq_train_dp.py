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
  --fork <spec file>         several networks of one job on one shared copy of the games (see main_fork)
  --bank / --margin          cross-batch memory negatives, cosine margin (METHOD_NOTES 2.64)
  --cls-loss cosface         classification over all training players with a cosine margin (METHOD_NOTES 2.56)
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


SHARED = ("GL", "POS", "MOV", "LEN", "SUBJ")
SHARED_ANON = False  # set by main_fork: the big arrays live in anonymous shared memory
PRELOADED = None  # set by main_fork in every forked trainer: the games read once by the parent
_KEEP = []


def load_games(smoke, log, all_folds=False, ext_ply=None, ext_select=None, share_dir=None, ext_second=None, sup_ply=None, sup_stats=None, ext_deep=None):
    """M90's training games as padded arrays (m88_seqid.load_games without the test part). The worker pool starts before any
    large array exists and every part is written straight into preallocated arrays: the compute nodes run with strict
    overcommit, and a pool forked from a large parent exhausted the node's commit limit in the rank session's first run."""
    share = Path(share_dir) if share_dir else None
    if share is not None and (share / "DONE").exists():  # arrays prepared once per job in shared memory (scripts/nano4/pid_seq_train_shared.sh)
        G = {k: np.load(share / f"{k}.npy", mmap_mode="r") for k in SHARED}
        G.update({k: np.load(share / f"{k}.npy") for k in ("key", "pid", "fold")})
        log(f"games attached from {share}: {len(G['key'])} ({G['GL'].nbytes / 1e9:.1f} GB shared; process memory {_rss_gb():.1f} GB); games of fold 0: {int((G['fold'] == 0).sum())}")
        return G
    if share is not None:
        share.mkdir(parents=True, exist_ok=True)

    def alloc(name, shape, dtype, fill=0):  # in shared memory when asked: every network of the job then maps the same pages
        if share is None and SHARED_ANON:  # anonymous shared mapping: forked trainers (main_fork) read the same pages, counted once
            import mmap
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            mm = mmap.mmap(-1, max(nbytes, 1), flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            _KEEP.append(mm)
            arr = np.frombuffer(mm, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
            if fill != 0:
                arr[:] = fill
            return arr
        if share is None:
            return np.zeros(shape, dtype) if fill == 0 else np.full(shape, fill, dtype)
        arr = np.lib.format.open_memmap(share / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        if fill != 0:
            arr[:] = fill
        return arr

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
        empty = [str(d) for d in ext_ply if not any(Path(d).glob("ply_x_*.npz"))]
        if empty:  # a missing folder used to load silently less data (METHOD_NOTES 2.148)
            raise FileNotFoundError(f"external ply folders without parts: {empty}")
        fx = sorted(f for d in ext_ply for f in Path(d).glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz"))
        files2 += fx[:5] if smoke else fx
        import csv
        csv.field_size_limit(10 ** 9)
        for sel in ext_select:
            with open(sel, newline="") as fh:
                ext_own.update((r["game_key"], "x:" + r["account"]) for r in csv.DictReader(fh))
    deep_keys = set()  # METHOD_NOTES 2.126: selections (also named in --ext-select) that add MORE games to accounts selected before; a trainer drops them with --no-deep
    if ext_ply and ext_deep:
        for sel in ext_deep:
            with open(sel, newline="") as fh:
                deep_keys.update(r["game_key"] for r in csv.DictReader(fh))
    sec = {}  # METHOD_NOTES 2.73: the OTHER player of an external game, when that account trains too: game key -> (player, colour 0 / 1)
    if ext_ply and ext_second:
        for f in ext_second:
            with open(f, newline="") as fh:
                sec.update((r["game_key"], ("x:" + r["account"], 0 if r["color"] == "B" else 1)) for r in csv.DictReader(fh))
        sec = {k: v for k, v in sec.items() if k in ext_own}
    with mp.get_context("fork").Pool(12) as pool:
        n1 = pool.map(M.count_part, files, chunksize=8)
        n2 = pool.map(M.count_m90, files2, chunksize=8)
        n_old, N = int(sum(n1)), int(sum(n1)) + int(sum(n2))
        NA = N + len(sec)  # room for the second-side copies (an upper bound; the arrays are cut to what is used)
        GL, POS, MOV = alloc("GL", (NA, T, 128), np.float16), alloc("POS", (NA, T), np.int16, 362), alloc("MOV", (NA, T), np.int8)
        LEN, SUBJ, keys, pids, a = alloc("LEN", (NA,), np.int16), alloc("SUBJ", (NA,), np.int8), [], [], 0
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
    second = np.zeros(N, bool)
    if sec:  # the same stored game once more, read from the other player's side (flipped subject colour, that player's id)
        src = np.array([i for i in range(n_old, N) if key[i] in sec], np.int64)
        n2_ = len(src)
        want = np.array([sec[k][1] for k in key[src]], np.int8)
        bad = int((want != 1 - SUBJ[src]).sum())
        if bad:
            raise ValueError(f"{bad} second-side rows whose colour is not the other side of the stored subject")
        for j in range(0, n2_, 20000):
            r = src[j:j + 20000]
            GL[N + j:N + j + len(r)], POS[N + j:N + j + len(r)], MOV[N + j:N + j + len(r)], LEN[N + j:N + j + len(r)] = GL[r], POS[r], MOV[r], LEN[r]
        SUBJ[N:N + n2_] = want
        key = np.concatenate([key, key[src]])
        pid = np.concatenate([pid, np.array([sec[k][0] for k in key[src]])])
        fold = np.concatenate([fold, np.full(n2_, -9, np.int8)])
        second = np.concatenate([second, np.ones(n2_, bool)])
        log(f"second side: {n2_} external games are read once more from the other player's side ({len(set(pid[N:].tolist()))} players)")
        N += n2_
    GL, POS, MOV, LEN, SUBJ = GL[:N], POS[:N], MOV[:N], LEN[:N], SUBJ[:N]
    log(f"games loaded: {N} ({n_old} stored of folds {'0-4' if all_folds else '1-4'} + {N - n_old} extracted; {GL.nbytes / 1e9:.1f} GB; process memory {_rss_gb():.1f} GB); "
        f"games of fold 0: {int((fold == 0).sum())}, {'TRAINED ON (--all-folds)' if all_folds else 'left out everywhere'}")
    if share is not None:
        for arr in (GL, POS, MOV, LEN, SUBJ):
            arr.flush()
        np.save(share / "key.npy", key); np.save(share / "pid.npy", pid); np.save(share / "fold.npy", fold)
        (share / "DONE").write_text("ok\n")
        log(f"arrays shared in {share}")
    sup_idx, SUPD = np.full(N, -1, np.int32), None
    if sup_ply:  # METHOD_NOTES 2.88 (probe B): supervised per-ply features of some games, padded like GL and shared the same way
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pid_sup
        sup_idx, SUPD = pid_sup.dense(pid_sup.load_sup(sup_ply, sup_stats), key.tolist(), T, alloc=lambda shape, dtype: alloc("SUP", shape, dtype))
        log(f"supervised per-ply features: {int((sup_idx >= 0).sum())} of {N} games, {SUPD.shape[2]} dimensions ({SUPD.nbytes / 1e9:.1f} GB)")
    deep = np.fromiter((k in deep_keys for k in key.tolist()), bool, len(key)) if deep_keys else np.zeros(len(key), bool)
    if deep_keys:
        log(f"deeper selections: {int(deep.sum())} of the loaded games ({len(set(pid[deep].tolist()))} accounts)")
    return dict(key=key, pid=pid, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, fold=fold, second=second, sup_idx=sup_idx, SUP=SUPD, deep=deep)


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
    ap.add_argument("--sup-ply", nargs="*", default=None, help="directories of scripts/pid_sup_ply_extract.py parts (supervised per-ply features; METHOD_NOTES 2.88)")
    ap.add_argument("--sup-stats", default=None, help="the stats file of the supervised stores (pid_sup.write_stats): one scale for all of them")
    ap.add_argument("--sup-dims", type=int, nargs=2, default=None, metavar=("FROM", "TO"), help="put the supervised features into these input dimensions (0 64: in place of the pooled half, 64 128: of the played-point half)")
    ap.add_argument("--sup-only", action="store_true", help="with --sup-dims: every other input dimension is set to its mean")
    ap.add_argument("--sup-require", action="store_true", help="train only on games that have supervised features (the same games for every arm of the probe)")
    ap.add_argument("--sup-name", default=None, help="name of the supervised feature set, recorded in the checkpoint so that the pool fine-tune reads the matching pool features (None: the first set, METHOD_NOTES 2.88; js: the joint projection with the policy-expected vector + 4 scalars, 2.110)")
    ap.add_argument("--ext-second", nargs="*", default=None, help="second.csv files of scripts/ext_fox_census.py: external games are also read from the other player's side")
    ap.add_argument("--no-second", action="store_true", help="leave the second-side rows out (a control arm inside a job that loaded them)")
    ap.add_argument("--ext-deep", nargs="*", default=None, help="select.csv files (also given in --ext-select) of DEEPER selections: more games of accounts selected before (METHOD_NOTES 2.126)")
    ap.add_argument("--no-deep", action="store_true", help="leave the games of the deeper selections out (a control arm inside a job that loaded them)")
    ap.add_argument("--ext-share", type=float, default=None, help="share of an episode's players that are external (default: no quota)")
    ap.add_argument("--arch", default=None, help="D,LAYERS,HEADS,FFN of the transformer (M90: 384,8,8,1536); stored in the checkpoint, the loaders rebuild the same network")
    ap.add_argument("--sym-aug", action="store_true")
    ap.add_argument("--drop-open", type=float, default=0.0, help="share of the games of a batch whose opening is hidden (board vectors and played points of the first K plies, "
                                                                 "K drawn in 20 .. --drop-open-max): the network must also tell players apart from the middle game (METHOD_NOTES 2.60)")
    ap.add_argument("--drop-open-max", type=int, default=100)
    ap.add_argument("--sym-game", action="store_true", help="with --sym-aug: every GAME gets its own random board symmetry (orientation carries no identity then) instead of one per episode")
    ap.add_argument("--p-ep", type=int, default=None)
    ap.add_argument("--ext-share-end", type=float, default=None)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--bank", action="store_true", help="cross-batch memory (METHOD_NOTES 2.64): every training player's prototype from its last visit is a negative "
                                                        "for the games of later episodes, so the most similar players are present at every step, not only the 255 of the episode")
    ap.add_argument("--bank-start", type=int, default=1000, help="first step that uses the memory")
    ap.add_argument("--bank-ema", action="store_true", help="the memory holds prototypes computed by the EMA network (slowly moving, as a momentum encoder) instead of the online one")
    ap.add_argument("--bank-age", type=int, default=0, help="only memory entries written in the last this many steps count (0: all)")
    ap.add_argument("--bank-ramp", type=int, default=0, help="steps over which the memory's share of the loss rises from 0 to 1 after --bank-start")
    ap.add_argument("--win", type=float, default=0.0, help="share of the games of a batch shown only through a random window of plies (true ply positions kept; the rest hidden); "
                                                           "evaluation always reads whole games (METHOD_NOTES 2.66; the rank session's idea)")
    ap.add_argument("--win-len", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--bank-topk", type=int, default=0, help="only the k highest memory negatives of each game (0: all)")
    ap.add_argument("--margin", type=float, default=0.0, help="cosine margin taken off a game's score against its own player")
    ap.add_argument("--cls-loss", choices=("none", "cosface"), default="none",
                    help="classification over ALL training players with an additive cosine margin (face / speaker recognition; METHOD_NOTES 2.56): "
                         "every step each game is told apart from every player's centre, not only from the 255 others of its episode")
    ap.add_argument("--cls-margin", type=float, default=0.2)
    ap.add_argument("--cls-scale", type=float, default=30.0)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--no-episode-loss", action="store_true", help="train with the classification loss alone")
    ap.add_argument("--share-dir", default=None, help="a directory in shared memory (/dev/shm/...): the game arrays are built there once and mapped by every network of the job")
    ap.add_argument("--prepare-only", action="store_true", help="build the shared arrays and stop")
    ap.add_argument("--rank-ep", choices=("class", "band"), default=None,
                    help="hard negatives (METHOD_NOTES 2.55): an episode's competition players come from ONE rank class (class) or from a class and its two neighbours (band), "
                         "its external accounts from one rank folder; classes and folders drawn in proportion to their players")
    ap.add_argument("--rank-ep-share", type=float, default=1.0, help="share of the steps drawn that way; the others are drawn as before")
    ap.add_argument("--fix-chan", type=float, default=0.0, help="this network never sees a fixed random share of the 128 input dimensions (its own subset, from --seed): their scale is set so large that they standardise to zero, in training and wherever the checkpoint is used (METHOD_NOTES 2.97)")
    ap.add_argument("--ext-bag", type=float, default=0.0, help="this network trains on a random share of the external accounts (its own subset, from --seed)")
    ap.add_argument("--ema-only", action="store_true", help="write only the EMA checkpoints (official networks: the raw ones are never used)")
    ap.add_argument("--snap", nargs="*", type=int, default=[], help="also write the checkpoints at these steps, as ckpt_<net>_<tag>k<step/1000>[e].pt (snapshots of one long run)")
    ap.add_argument("--m-ep", type=int, default=None, help="games per player in an episode (default 8; METHOD_NOTES 2.69)")
    ap.add_argument("--min-games", type=int, default=None, help="an EXTERNAL account trains with at least this many usable games (default 8); competition players always need 8")
    ap.add_argument("--mask-pos", type=float, default=0.0, help="per game a random share up to this of the plies have their played POINT hidden (the board vectors stay): the complement of --mask (METHOD_NOTES 2.80)")
    ap.add_argument("--mask-both", action="store_true", help="--mask hides the played point of the masked plies as well (whole plies dropped at random)")
    ap.add_argument("--chan-drop", type=float, default=0.0, help="per game every one of the 128 input dimensions is replaced by its mean with this probability")
    ap.add_argument("--drop-tail", type=float, default=0.0, help="share of the games of a batch whose plies from a random ply K on are hidden (K uniform in --drop-tail-min .. 220)")
    ap.add_argument("--drop-tail-min", type=int, default=100)
    ap.add_argument("--jitter", type=float, default=0.0, help="Gaussian noise on the per-ply trunk vectors in training, in units of each dimension's standard deviation (METHOD_NOTES 2.72)")
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing of the episode loss")
    ap.add_argument("--ext-max-plies", type=int, default=0, help="external games are shown only up to this ply (0: all; as if they had been stored that short; METHOD_NOTES 2.70)")
    ap.add_argument("--ext-cap", type=int, default=0, help="use only the first this many games of every external account (0: all)")
    ap.add_argument("--ext-accounts", type=int, default=0, help="use only this many external accounts: the first N of one fixed random order of the accounts with 8 or more usable games, "
                                                               "so the subsets of different N are nested (0: all)")
    ap.add_argument("--own-by-games", choices=("linear", "sqrt"), default=None,
                    help="with --ext-share: the competition players of an episode are drawn with probability proportional to their number of usable games (linear) or to its "
                         "square root, instead of uniformly; the external accounts stay uniform (METHOD_NOTES 2.124; the rank session's M161)")
    ap.add_argument("--ext-by-games", choices=("linear", "sqrt"), default=None,
                    help="with --ext-share: the external accounts of an episode are drawn by their number of usable games as well (METHOD_NOTES 2.126; "
                         "their depths differ once --ext-second or deeper selections are loaded)")
    ap.add_argument("--nn-ep", type=float, default=0.0, help="share of the steps whose competition players are LOOK-ALIKE clusters (METHOD_NOTES 2.140): seeds drawn as usual, "
                                                             "each joined by its nearest competition players by the prototypes of their last visit; the external accounts stay random")
    ap.add_argument("--nn-k", type=int, default=8, help="--nn-ep: players per cluster (the seed and its k - 1 nearest)")
    ap.add_argument("--nn-start", type=int, default=4000, help="--nn-ep: first step that may draw clusters (the prototypes are collected from step 0)")
    ap.add_argument("--drop-fold", type=int, nargs="*", default=[], help="competition folds left out of the training in addition to the held-out one "
                                                                         "(a measurement of what one more fold of competition players is worth; METHOD_NOTES 2.131)")
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
    G = PRELOADED if PRELOADED is not None else load_games(a.smoke, log, a.all_folds, a.ext_ply, a.ext_select, a.share_dir, a.ext_second, a.sup_ply, a.sup_stats, a.ext_deep)
    if PRELOADED is not None:
        log(f"games shared by the job: {len(G['key'])} ({G['GL'].nbytes / 1e9:.1f} GB, read once; this process holds {_rss_gb():.1f} GB)")
    if a.prepare_only:
        return 0
    held = set(json.load(open(M.M82 / "held_keys.json")))
    players, owner = np.unique(G["pid"], return_inverse=True)
    C, fold = len(players), G["fold"]
    is_held = np.array([k in held for k in G["key"]])
    allowed = np.ones(len(fold), bool) if a.all_folds else fold != 0
    is_ext = np.char.startswith(players.astype(str), "x:")  # per player
    use = ~is_held & allowed & (fold != v)
    if a.drop_fold:
        use &= ~np.isin(fold, a.drop_fold)
    sec_row = G.get("second", np.zeros(len(fold), bool))
    if a.no_second:
        use &= ~sec_row
    if a.no_deep:
        use &= ~G.get("deep", np.zeros(len(fold), bool))
    sup_idx = G.get("sup_idx")
    if a.sup_dims:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        global pid_sup
        import pid_sup
    if a.sup_require:
        use &= sup_idx >= 0
    if a.sup_dims or a.sup_require:
        assert sup_idx is not None and G.get("SUP") is not None, "--sup-dims / --sup-require need --sup-ply"
    if a.m_ep:
        M.M_EP = a.m_ep
    if a.ext_cap or a.ext_accounts:  # fewer external accounts and / or fewer games of each (METHOD_NOTES 2.69)
        e_idx = np.flatnonzero(use & is_ext[owner])
        eo = owner[e_idx]
        od = np.argsort(eo, kind="stable")
        e_idx, eo = e_idx[od], eo[od]
        rk = np.arange(len(eo)) - np.searchsorted(eo, eo, side="left")  # position of a game among its account's usable games
        out_ = np.zeros(len(eo), bool)
        if a.ext_accounts:
            n1_ = np.bincount(owner[use & is_ext[owner] & ~sec_row], minlength=C)  # own (first-side) games only: the subset is the same with and without second sides
            full = np.flatnonzero(n1_ >= M.MIN_GAMES)  # accounts with at least 8 usable games of their own, in the order of their names
            out_ |= ~np.isin(eo, np.random.default_rng(20261114).permutation(full)[:a.ext_accounts])
        if a.ext_cap:
            out_ |= rk >= a.ext_cap
        use[e_idx[out_]] = False
    if a.ext_bag:  # a different random part of the external accounts for every network: members that err differently
        ext_p = np.flatnonzero(is_ext)
        out_p = ext_p[np.random.default_rng(777 + a.seed).random(len(ext_p)) >= a.ext_bag]
        use &= ~np.isin(owner, out_p)
    cnt = np.bincount(owner[use], minlength=C)
    tp = np.flatnonzero(np.where(is_ext, cnt >= max(a.min_games or M.MIN_GAMES, M.M_EP), cnt >= max(M.MIN_GAMES, M.M_EP)))
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
    p_own = None
    if a.own_by_games:
        assert a.ext_share is not None and not a.rank_ep, "--own-by-games needs --ext-share and excludes --rank-ep"
        w_own = cnt[tp_own].astype(np.float64)
        w_own = w_own if a.own_by_games == "linear" else np.sqrt(w_own)
        p_own = w_own / w_own.sum()
        top = np.sort(p_own)[::-1]
        log(f"competition players drawn by games ({a.own_by_games}): games per player 10/50/90/99% {np.percentile(cnt[tp_own], [10, 50, 90, 99]).round().astype(int).tolist()}; "
            f"the busiest 10% of the players get {top[:max(len(top) // 10, 1)].sum():.2f} of the draws (uniform: 0.10)")
    p_ext = None
    if a.ext_by_games:
        assert a.ext_share is not None and not a.rank_ep and len(tp_ext), "--ext-by-games needs --ext-share and external accounts, and excludes --rank-ep"
        w_ext = cnt[tp_ext].astype(np.float64)
        w_ext = w_ext if a.ext_by_games == "linear" else np.sqrt(w_ext)
        p_ext = w_ext / w_ext.sum()
        top = np.sort(p_ext)[::-1]
        log(f"external accounts drawn by games ({a.ext_by_games}): games per account 10/50/90/99% {np.percentile(cnt[tp_ext], [10, 50, 90, 99]).round().astype(int).tolist()}; "
            f"the busiest 10% of the accounts get {top[:max(len(top) // 10, 1)].sum():.2f} of the draws (uniform: 0.10)")
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
    if a.fix_chan:
        hidden_dims = np.sort(np.random.default_rng(4242 + a.seed).choice(128, int(round(128 * a.fix_chan)), replace=False))
        net.sd[torch.as_tensor(hidden_dims, device=net.sd.device)] = 1e6
        log(f"input dimensions never seen by this network: {len(hidden_dims)} of 128 ({hidden_dims[:8].tolist()} ...)")
    ema = copy.deepcopy(net).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    bank = bank_ok = bcol = None
    if a.bank:  # no parameters: the prototypes are copied from the network's own outputs
        bank = torch.zeros(len(tp), M.D_OUT, device=dev)
        bank_ok = torch.zeros(len(tp), dtype=torch.bool, device=dev)
        bank_step = torch.full((len(tp),), -10 ** 9, dtype=torch.long, device=dev)
        bcol = torch.full((C,), -1, dtype=torch.long, device=dev)
        bcol[torch.as_tensor(tp, device=dev)] = torch.arange(len(tp), device=dev)
        log(f"cross-batch memory: {len(tp)} slots, used from step {a.bank_start}{f', top {a.bank_topk} per game' if a.bank_topk else ''}"
            f"{', entries from the EMA network' if a.bank_ema else ''}{f', entries up to {a.bank_age} steps old' if a.bank_age else ''}{f', ramp over {a.bank_ramp} steps' if a.bank_ramp else ''}; margin {a.margin}")
    nn_proto = nn_ok = None
    if a.nn_ep > 0:  # the last prototype of every competition player, to find who looks like whom
        assert a.ext_share is not None and not a.rank_ep, "--nn-ep needs --ext-share and excludes --rank-ep"
        nn_proto = torch.zeros(C, M.D_OUT, device=dev)
        nn_ok = np.zeros(C, bool)
        log(f"look-alike episodes: {a.nn_ep:g} of the steps from step {a.nn_start}, clusters of {a.nn_k} among {len(tp_own)} competition players")
    nn_steps = 0
    centres = col = None
    if a.cls_loss != "none":  # one centre per training player; the centres are not part of the saved network
        centres = torch.nn.Parameter(torch.randn(len(tp), M.D_OUT, device=dev) * 0.01)
        col = torch.full((C,), -1, dtype=torch.long, device=dev)
        col[torch.as_tensor(tp, device=dev)] = torch.arange(len(tp), device=dev)
        log(f"classification loss {a.cls_loss}: {len(tp)} centres, margin {a.cls_margin}, scale {a.cls_scale}, weight {a.cls_weight}{', episode loss off' if a.no_episode_loss else ''}")
    opt = torch.optim.AdamW([{"params": net.parameters(), "weight_decay": M.WD}] + ([{"params": [centres], "weight_decay": 0.0}] if centres is not None else []), lr=M.LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / M.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, horizon) / horizon)))
    state_file, start, hist = out / f"state_{name}.pt", 0, []
    if state_file.exists():
        st = torch.load(state_file, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, hist = st["step"], st["hist"]
        rng = np.random.default_rng(990 + v + 1000 * a.seed + 7 * start)
        log(f"resumed at step {start}")
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731

    def sup_in(rows_, gl_):
        """The batch's input with the supervised features in their dimensions (--sup-dims); unchanged without the option."""
        if not a.sup_dims:
            return gl_
        si = sup_idx[rows_]
        z = torch.zeros(len(rows_), T, G["SUP"].shape[2], device=dev)
        has = np.flatnonzero(si >= 0)
        if len(has):
            z[torch.as_tensor(has, device=dev)] = tb(G["SUP"][si[has]]).float()
        return pid_sup.put(gl_.float(), z, net.mu, net.sd, a.sup_dims, a.sup_only)

    def embed(model, rows):
        model.eval()
        res = np.zeros((len(rows), M.D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]
                res[i:i + 1024] = model(sup_in(r, tb(G["GL"][r])), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
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
    meta = {"step": steps, "horizon": horizon, "drop": a.drop, "mask": a.mask, "ema": a.ema, "seed": a.seed, "net": v, "players_trained": int(len(tp)), "games": int(use.sum()), "all_folds": bool(a.all_folds), "ext_players": int(len(tp_ext)), "ext_share": a.ext_share, "ext_share_end": a.ext_share_end, "sym_aug": bool(a.sym_aug), "sym_game": bool(a.sym_game), "drop_open": a.drop_open, "drop_open_max": a.drop_open_max, "p_ep": int(P), "rank_ep": a.rank_ep, "rank_ep_share": a.rank_ep_share, "bank": bool(a.bank), "bank_start": a.bank_start, "bank_topk": a.bank_topk, "bank_ema": bool(a.bank_ema), "bank_age": a.bank_age, "bank_ramp": a.bank_ramp, "win": a.win, "win_len": list(a.win_len), "margin": a.margin, "cls_loss": a.cls_loss, "cls_margin": a.cls_margin, "cls_scale": a.cls_scale, "cls_weight": a.cls_weight, "episode_loss": not a.no_episode_loss}

    def write_ckpt(tag, at):
        for sfx, model, label in (("", net, "raw"), ("e", ema, "ema")):
            if a.ema_only and sfx == "":
                continue
            sd_ = cpu(model)  # one copy, under both names the loaders know (two separate copies doubled every file)
            torch.save({"best_state": sd_, "net": sd_, "best": {**meta, "step": at, "state": label}, "step": at, "arch": arch,
                        "sup": ({"dims": list(a.sup_dims), "only": bool(a.sup_only), "name": a.sup_name} if a.sup_dims else None)}, out / f"ckpt_{v}_{tag}{sfx}{'_smoke' if a.smoke else ''}.pt")

    net.train()
    from o26_parallel_forward import make_forward
    training_forward = make_forward(net)
    log(f"training forward across {torch.cuda.device_count()} GPU(s); full episode loss on GPU 0")
    ar, tl, ta, tc, t_last, every = np.arange(P), 0.0, 0.0, 0.0, time.time(), (50 if a.smoke else 500)
    for step in range(start, steps):
        if a.ext_share is not None and a.ext_share_end is not None:
            n_ext = min(int(round(P * (a.ext_share + (a.ext_share_end - a.ext_share) * step / max(steps - 1, 1)))), len(tp_ext))
        if strata is not None and rng.random() < a.rank_ep_share:
            pl = rank_draw("own", P) if a.ext_share is None or n_ext == 0 else np.concatenate([rank_draw("ext", n_ext), rank_draw("own", P - n_ext)])
        elif nn_proto is not None and step >= a.nn_start and rng.random() < a.nn_ep and int(nn_ok[tp_own].sum()) >= 4 * (P - n_ext):
            n_own = P - n_ext
            okm = nn_ok[tp_own]
            valid = tp_own[okm]
            seeds = rng.choice(valid, n_own // a.nn_k, replace=False, p=None if p_own is None else p_own[okm] / p_own[okm].sum())
            sim = (nn_proto[tb(seeds)] @ nn_proto[tb(valid)].T).cpu().numpy()
            used = np.zeros(C, bool)
            used[seeds] = True
            chosen = []
            for i_, sd_ in enumerate(seeds):  # the seed and its nearest players that no earlier cluster has taken
                near = valid[np.argsort(-sim[i_])[:4 * a.nn_k + len(seeds)]]
                near = near[~used[near]][:a.nn_k - 1]
                used[near] = True
                chosen += [sd_, *near.tolist()]
            if len(chosen) < n_own:
                free = tp_own[~used[tp_own]]
                chosen += rng.choice(free, n_own - len(chosen), replace=False).tolist()
            pl = np.concatenate([rng.choice(tp_ext, n_ext, replace=False, p=p_ext), np.array(chosen, dtype=tp_own.dtype)])
            nn_steps += 1
        else:
            pl = rng.choice(tp, P, replace=False) if a.ext_share is None else np.concatenate([rng.choice(tp_ext, n_ext, replace=False, p=p_ext), rng.choice(tp_own, P - n_ext, replace=False, p=p_own)])
        u = rng.random((P, W))
        u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0
        rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M.M_EP]].reshape(-1)
        gl = sup_in(rows, tb(G["GL"][rows]))
        if a.jitter > 0:  # the same game never looks exactly the same twice
            gl = gl + (a.jitter * net.sd.to(gl.dtype))[None, None] * torch.randn_like(gl)
        hide = None
        if a.mask > 0:
            hit = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
            if a.mask_both:
                hide = hit
        if a.chan_drop > 0:
            cd = torch.rand(len(rows), 1, gl.shape[-1], device=dev) < a.chan_drop
            gl = torch.where(cd, net.mu.to(gl.dtype)[None, None], gl)
        if a.mask_pos > 0:  # only the played point is hidden
            hp = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask_pos
            hide = hp if hide is None else (hide | hp)
        if a.drop_tail > 0:
            cut_t = torch.where(torch.rand(len(rows), device=dev) < a.drop_tail, torch.randint(a.drop_tail_min, 221, (len(rows),), device=dev), torch.full((len(rows),), T + 1, dtype=torch.long, device=dev))
            tail = torch.arange(T, device=dev)[None] >= cut_t[:, None]
            gl = torch.where(tail[..., None], net.mu.to(gl.dtype)[None, None], gl)
            hide = tail if hide is None else (hide | tail)
        if a.drop_open > 0:
            cut = torch.where(torch.rand(len(rows), device=dev) < a.drop_open, torch.randint(20, a.drop_open_max + 1, (len(rows),), device=dev), torch.zeros(len(rows), dtype=torch.long, device=dev))
            opening = torch.arange(T, device=dev)[None] < cut[:, None]
            gl = torch.where(opening[..., None], net.mu.to(gl.dtype)[None, None], gl)
            hide = opening if hide is None else (hide | opening)
        if a.win > 0:
            len_w = tb(G["LEN"][rows].astype(np.int64))
            w_len = torch.as_tensor(a.win_len, device=dev)[torch.randint(len(a.win_len), (len(rows),), device=dev)]
            w_start = (torch.rand(len(rows), device=dev) * (len_w - w_len).clamp(min=0).float()).long()
            t_idx = torch.arange(T, device=dev)[None]
            outside = ((t_idx < w_start[:, None]) | (t_idx >= (w_start + w_len)[:, None])) & (torch.rand(len(rows), device=dev) < a.win)[:, None]
            gl = torch.where(outside[..., None], net.mu.to(gl.dtype)[None, None], gl)
            hide = outside if hide is None else (hide | outside)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pos_np = G["POS"][rows].astype(np.int64)
            if sym_tab is not None:
                pos_np = sym_tab[rng.integers(8, size=len(rows))[:, None], pos_np] if a.sym_game else sym_tab[rng.integers(8)][pos_np]
            pos_t = tb(pos_np)
            if hide is not None:
                pos_t = torch.where(hide, torch.full_like(pos_t, 362), pos_t)  # 362 = the padding point: no move, no displacement
            len_np = G["LEN"][rows].astype(np.int64)
            if a.ext_max_plies:
                len_np = np.where(is_ext[owner[rows]], np.minimum(len_np, a.ext_max_plies), len_np)
            mov_t, len_t, subj_t = tb(G["MOV"][rows].astype(np.int64)), tb(len_np), tb(G["SUBJ"][rows].astype(np.int64))
            z = training_forward(gl, pos_t, mov_t, len_t, subj_t)
        z = z.float().reshape(P, M.M_EP, M.D_OUT)
        tot, sc = z.sum(1), net.log_scale.exp()
        logits = sc * z.reshape(-1, M.D_OUT) @ F.normalize(tot, dim=-1).T
        own_ = sc * (z * F.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)  # a game against its own player's prototype without itself
        tgt = tb(np.repeat(ar, M.M_EP))
        logits = logits.scatter(1, tgt[:, None], (own_ - sc * a.margin)[:, None])
        loss = F.cross_entropy(logits, tgt, label_smoothing=a.smooth)
        if bank is not None and step >= a.bank_start:
            bslots = bcol[tb(pl)]
            off = ~bank_ok if not a.bank_age else ~(bank_ok & (step - bank_step <= a.bank_age))
            off[bslots] = True  # players without a usable entry, and this episode's own players (their fresh prototypes are already in `logits`)
            if not bool(off.all()):
                lb = (sc * (z.reshape(-1, M.D_OUT) @ bank.T)).masked_fill(off[None], -1e4)
                if a.bank_topk:
                    lb = lb.topk(min(a.bank_topk, lb.shape[1]), dim=1).values
                lam = min(1.0, (step - a.bank_start + 1) / a.bank_ramp) if a.bank_ramp else 1.0
                loss = (1 - lam) * loss + lam * F.cross_entropy(torch.cat([logits, lb], 1), tgt)
        if centres is not None:
            ct = col[tb(np.repeat(pl, M.M_EP))]
            cos = z.reshape(-1, M.D_OUT) @ F.normalize(centres, dim=-1).T
            l_cls = F.cross_entropy(a.cls_scale * (cos - a.cls_margin * F.one_hot(ct, len(tp)).to(cos.dtype)), ct)
            loss = a.cls_weight * l_cls if a.no_episode_loss else loss + a.cls_weight * l_cls
            tc += float((cos.argmax(1) == ct).float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
            if bank is not None:
                bslots = bcol[tb(pl)]
                if a.bank_ema:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        zb = ema(gl, pos_t, mov_t, len_t, subj_t)
                    zb = zb.float().reshape(P, M.M_EP, M.D_OUT)
                else:
                    zb = z.detach()
                bank[bslots] = F.normalize(zb.sum(1), dim=-1)
                bank_ok[bslots] = True
                bank_step[bslots] = step
            if nn_proto is not None:
                own_m = ~is_ext[pl]
                nn_proto[tb(pl[own_m])] = F.normalize(z.detach().sum(1)[tb(np.flatnonzero(own_m))], dim=-1)
                nn_ok[pl[own_m]] = True
        tl += float(loss.detach())
        ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % every == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, episode top-1 {ta / every:.3f} of {P}, scale {float(sc.detach()):.1f}, {(time.time() - t_last) / every:.2f} s/step" + (f", all-player top-1 {tc / every:.3f}" if centres is not None else ""))
            tl, ta, tc, t_last = 0.0, 0.0, 0.0, time.time()
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


def main_fork(argv):
    """Several networks of one job on ONE copy of the games (METHOD_NOTES 2.57).
        pid_seq_train.py --fork <spec file> [--smoke] [--all-folds] [--ext-ply ...] [--ext-select ...]
    The games are read once into anonymous shared memory, then one trainer is forked per line of the spec file:
        <GPU id> <log file> <arguments of a normal run: --net v --tag t ...>
    A trainer sets its GPU before CUDA starts and writes to its log. Why: on the shared nodes every private copy (95-180 GB)
    is slow to allocate when the page cache is full, and eight copies run into the node's commit limit; /dev/shm is not
    reliable (it was full of another user's files on 25a-hgpn069)."""
    global SHARED_ANON, PRELOADED
    import shlex
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--ext-ply", nargs="*", default=None)
    ap.add_argument("--ext-select", nargs="*", default=None)
    ap.add_argument("--ext-second", nargs="*", default=None)
    ap.add_argument("--sup-ply", nargs="*", default=None)
    ap.add_argument("--sup-stats", default=None)
    ap.add_argument("--ext-deep", nargs="*", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s_: print(f"{s_} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    specs = [shlex.split(line) for line in Path(a.fork).read_text().splitlines() if line.strip() and not line.startswith("#")]
    SHARED_ANON = True
    G = load_games(a.smoke, log, a.all_folds, a.ext_ply, a.ext_select, None, a.ext_second, a.sup_ply, a.sup_stats, a.ext_deep)
    for k in SHARED:
        G[k].flags.writeable = False  # the trainers only read; a stray write would be seen by all of them
    data = (["--smoke"] if a.smoke else []) + (["--all-folds"] if a.all_folds else []) + (["--ext-ply", *a.ext_ply] if a.ext_ply else []) + (["--ext-select", *a.ext_select] if a.ext_select else []) + (["--ext-second", *a.ext_second] if a.ext_second else []) + (["--sup-ply", *a.sup_ply] if a.sup_ply else []) + (["--sup-stats", a.sup_stats] if a.sup_stats else []) + (["--ext-deep", *a.ext_deep] if a.ext_deep else [])
    pids = {}
    for gpu, logfile, *rest in specs:
        sys.stdout.flush()
        pid = os.fork()
        if pid == 0:
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu
                fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
                os.dup2(fd, 1); os.dup2(fd, 2)
                sys.stdout = os.fdopen(1, "w", buffering=1); sys.stderr = os.fdopen(2, "w", buffering=1)
                PRELOADED = G
                rc = main(rest + data)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                rc = 1
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(int(rc or 0))
        pids[pid] = logfile
        log(f"trainer {pid} on GPU {gpu}: {' '.join(rest)} -> {logfile}")
    failed = 0
    while pids:
        pid, status = os.wait()
        ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        log(f"trainer {pid} ({pids.pop(pid)}) {'done' if ok else 'FAILED'}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    sys.exit(main_fork(sys.argv[1:]) if "--fork" in sys.argv[1:] else main())
