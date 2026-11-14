#!/usr/bin/env python
"""Probe Q (METHOD_NOTES 2.117): does the OTHER side of a game help to say who the subject is?

The sequence networks read every ply of a game and return a vector for the colour flagged as the subject. Flagging the other
colour gives a vector of the OPPONENT from the same stored plies, with no new extraction (the rank session's M91 did this on
training rows). Nothing in the player-identification pipeline reads the opponent so far. What could be in it:
  S0  who is on the other side resembles the subject (servers pair players of similar strength): cos(o_q, P_c)
  S1  the same opponent again: a player meets some opponents repeatedly, so the opponent of a question game may also be the
      opponent in one of the candidate's reference games: max over the candidate's games r of cos(o_q, o_r)
  S2  the candidate's usual kind of opponent: the mean of those cosines
  S3  the opponent is himself a candidate X of the pool, and X and the candidate meet in another reference game (seen from
      either folder): sum over X of weight(o_q looks like X) x link(candidate, X)
Labelled check on the OFFICIAL CANDIDATES only (owners given by the organisers; no question game is loaded; the Public and the
Private pool separately, never mixed): every reference game in turn is a one-game question with its own game left out of
its owner's side (leave-one-game-out), networks as they are (no pool fine-tune). Per network every score matrix is
standardised per row and the networks are averaged, as the pipeline does for the sequence component.
Reported per pool: diagnostics (is the nearest opponent vector a game of the same candidate; is an opponent ever a candidate
of the pool), the score (e^-(rank-1) over the top five) of each signal alone and added to the own-side score with a weight
chosen on the OTHER pool, with a bootstrap over candidates.
Run from the AI_CUP_2026-1 checkout, one GPU:
  python -I <this> --nets o7Fs0e o7Fs1e ... --out results/final/v2_probe_opp.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
import pid_v2_seq_finetune as FT  # noqa: E402  (load_net, embed)

WGRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
SIGNALS = ("S0", "S1", "S1e", "S2", "S3")
HINGE = (2.0, 3.0, 4.0, 5.0)  # S1h<t>: only the part of the standardised S1e above t counts (a repeated opponent is rare and then decisive)


ALL = SIGNALS + tuple(f"S1h{t:g}" for t in HINGE) + tuple(f"P1h{t:g}" for t in HINGE) + tuple(f"U1h{t:g}" for t in HINGE)


def load_pool(pool, tests_dir, ply_dir):
    """Per-ply arrays of the pool's candidate games sorted by owner, owner index [G], number of candidates."""
    tagname = "public" if pool == "pub2" else "private"
    c = pd.read_csv(Path(tests_dir) / f"player_identification_candidates2_{tagname}.csv", dtype=str, keep_default_na=False, usecols=["game_id", "player_id"])
    players = sorted(c.player_id.unique())
    pidx = {p: i for i, p in enumerate(players)}
    own = {f"c:{i}:{g}": pidx[p] for i, (g, p) in enumerate(zip(c.game_id, c.player_id))}
    parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
    for f in sorted(glob.glob(f"{ply_dir}/ply_*.npz")):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        key = z["key"].astype(str)
        m = np.array([k in own for k in key])
        if m.any():
            for name in parts:
                parts[name].append(key[m] if name == "key" else z[name][m])
    A = {k: np.concatenate(x) for k, x in parts.items()}
    if set(A["key"].tolist()) != set(own):
        raise ValueError(f"{pool}: {len(set(own) - set(A['key'].tolist()))} candidate games without per-ply vectors")
    owner = np.array([own[k] for k in A["key"]])
    o = np.argsort(owner, kind="stable")
    A = {k: v[o] for k, v in A.items()}
    owner = owner[o]
    cnt = np.bincount(owner, minlength=len(players))
    if not (cnt == cnt[0]).all():
        raise ValueError(f"{pool}: candidates with different numbers of games ({cnt.min()}-{cnt.max()})")
    if not set(np.unique(A["SUBJ"]).tolist()) <= {0, 1}:
        raise ValueError("SUBJ is expected to be 0 / 1")
    return A, owner, len(players), int(cnt[0])


def unit(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def zrow(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-9)


def net_scores(U, O, owner, C, M, tau):
    """Score matrices [G, C] of one network (games sorted by owner, M per candidate) and the game-by-game cosines."""
    G = len(owner)
    ar = torch.arange(G, device=U.device)
    U, O = unit(U), unit(O)
    Psum = torch.zeros(C, U.shape[1], device=U.device, dtype=U.dtype).index_add_(0, owner, U)
    P = unit(Psum)
    Pl = unit(Psum[owner] - U)  # the owner's prototype without this game
    own = U @ P.T
    own[ar, owner] = (U * Pl).sum(1)
    s0 = O @ P.T  # the opponent against every candidate's prototype
    s0[ar, owner] = (O * Pl).sum(1)
    OO = O @ O.T
    UU = U @ U.T
    oo = OO.clone()
    oo.fill_diagonal_(-2.0)
    s1 = oo.view(G, C, M).max(-1).values  # the same opponent again (the owner: his other M - 1 games)
    oo.fill_diagonal_(0.0)
    div = torch.full((G, C), float(M), device=U.device, dtype=U.dtype)
    div[ar, owner] = M - 1
    s2 = oo.view(G, C, M).sum(-1) / div
    # S3: links between candidates through the opponents of their reference games
    a3 = s0.view(C, M, C)  # [candidate c, his game j, candidate X]: X looks like the opponent of that game
    top = a3.topk(2, dim=1)
    t1, t2, arg = top.values[:, 0], top.values[:, 1], top.indices[:, 0]  # [C, C]
    off = ~torch.eye(C, dtype=torch.bool, device=U.device)
    mu, sd = t1[off].mean(), t1[off].std()
    z1, z2 = (t1 - mu) / sd, (t2 - mu) / sd
    link = torch.maximum(z1, z1.T) * off  # X an opponent in c's games, or c an opponent in X's games
    w = torch.softmax(zrow(s0) / tau, dim=1)  # who the opponent of this game looks like
    s3 = w @ link.T
    local = (ar % M)  # index of a game inside its owner's block
    z1l = torch.where(arg[owner] == local[:, None], z2[owner], z1[owner])  # the owner's links without this game
    linkl = torch.maximum(z1l, z1.T[owner])
    linkl[ar, owner] = 0.0
    s3[ar, owner] = (w * linkl).sum(1)
    return {"own": own, "S0": s0, "S1": s1, "S2": s2, "S3": s3}, OO, UU


def game_scores(S, owner):
    """Competition score of every game: e^-(rank-1) of the owner if he is in the top five, else 0."""
    ar = torch.arange(len(owner), device=S.device)
    rank = (S > S[ar, owner][:, None]).sum(1)
    return torch.where(rank < 5, torch.exp(-rank.double()), torch.zeros_like(rank, dtype=torch.double)).cpu().numpy()


def boot(diff, owner, C, n, rng):
    """Mean and 95% interval of a per-game difference, candidates resampled."""
    per = np.bincount(owner, weights=diff, minlength=C)
    cnt = np.bincount(owner, minlength=C)
    idx = rng.integers(0, C, size=(n, C))
    m = per[idx].sum(1) / cnt[idx].sum(1)
    return [float(diff.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", required=True, help="checkpoint names without ckpt_-1_ and .pt (plain-input networks only)")
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")
    ap.add_argument("--tests-dir", default=str(R2 / "data" / "tests_1111"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    rng = np.random.default_rng(0)
    res = {"nets": a.nets, "tau": a.tau, "pools": {}}
    keep = {}
    for pool in ("pub2", "priv2"):
        A, owner_np, C, M = load_pool(pool, a.tests_dir, a.ply_dir)
        G = len(owner_np)
        B = dict(A)
        B["SUBJ"] = (1 - A["SUBJ"]).astype(A["SUBJ"].dtype)
        owner = torch.as_tensor(owner_np).cuda()
        idx = np.arange(G)
        acc = {k: torch.zeros(G, C, device="cuda", dtype=torch.float64) for k in ("own",) + SIGNALS if k != "S1e"}
        OOm = torch.zeros(G, G, device="cuda", dtype=torch.float32)
        UUm = torch.zeros(G, G, device="cuda", dtype=torch.float32)
        single = {k: [] for k in ("own", "S0", "S1", "S2", "S3")}
        for name in a.nets:
            net = FT.load_net(Path(a.ckpt_dir) / f"ckpt_-1_{name}.pt")
            if net.sup_cfg:
                raise ValueError(f"{name} reads supervised per-ply features, which exist for the subject's plies only")
            U = torch.as_tensor(FT.embed(net, A, idx)).cuda().double()
            O = torch.as_tensor(FT.embed(net, B, idx)).cuda().double()
            del net
            S, OO, UU = net_scores(U, O, owner, C, M, a.tau)
            for k, v in S.items():
                acc[k] += zrow(v)
                single[k].append(float(game_scores(v, owner).mean()))
            OOm += OO.float()
            UUm += UU.float()
            log(f"{pool} {name}: alone own {single['own'][-1]:.4f}  S0 {single['S0'][-1]:.4f}  S1 {single['S1'][-1]:.4f}  S2 {single['S2'][-1]:.4f}  S3 {single['S3'][-1]:.4f}")
        K = len(a.nets)
        Z = {k: v / K for k, v in acc.items()}
        OOm /= K
        UUm /= K
        ar = torch.arange(G, device="cuda")
        # diagnostics on the network-averaged cosines
        same = owner[:, None] == owner[None, :]
        d = {}
        for nm, X in (("opponent", OOm), ("own", UUm)):
            Y = X.clone()
            Y.fill_diagonal_(-2.0)
            nn = Y.argmax(1)
            d[f"nearest_{nm}_same_candidate"] = float((owner[nn] == owner).float().mean())
            top = Y.topk(5, dim=1).indices
            d[f"top5_{nm}_same_candidate"] = float((owner[top] == owner[:, None]).any(1).float().mean())
            pos = Y[same & ~torch.eye(G, dtype=torch.bool, device="cuda")]
            neg = Y[~same]
            sub = neg[torch.randint(0, len(neg), (2_000_000,), device="cuda")]
            qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], device="cuda")
            thr = torch.quantile(sub.float(), qs)
            d[f"{nm}_pairs_other_candidates_quantiles"] = [float(x) for x in thr]
            d[f"{nm}_pairs_same_candidate_share_above"] = [float((pos > t).float().mean()) for t in thr]  # 0.5 / 0.1 / 0.01 / 0.001 / 0.0001 if nothing is there
        d["null_nearest"] = (M - 1) / (G - 1)
        Y = OOm.clone()
        Y.fill_diagonal_(-2.0)
        Z["S1e"] = zrow(Y.view(G, C, M).max(-1).values.double())  # the same opponent again, on the averaged cosines
        # the same PAIRING again: one reference game in which both the opponent and the subject look like those of this game
        # (P1), and as a control the own side alone in its best single reference game (U1)
        Yp = (OOm + UUm).clone()
        Yp.fill_diagonal_(-4.0)
        zp = zrow(Yp.view(G, C, M).max(-1).values.double())
        Yu = UUm.clone()
        Yu.fill_diagonal_(-2.0)
        zu = zrow(Yu.view(G, C, M).max(-1).values.double())
        del Yp, Yu
        for t in HINGE:
            Z[f"S1h{t:g}"] = torch.relu(Z["S1e"] - t)
            Z[f"P1h{t:g}"] = torch.relu(zp - t)
            Z[f"U1h{t:g}"] = torch.relu(zu - t)
        # is the opponent ever a candidate of the pool: row maxima of the opponent side against those of wrong candidates on the own side
        zo, zw = Z["S0"].clone(), Z["own"].clone()
        zo[ar, owner] = -9.0
        zw[ar, owner] = -9.0
        mo, mw = zo.max(1).values, zw.max(1).values
        qs = torch.tensor([0.5, 0.9, 0.99, 0.999], device="cuda", dtype=torch.float64)
        d["rowmax_opponent_vs_candidates_quantiles"] = [float(x) for x in torch.quantile(mo, qs)]
        d["rowmax_own_vs_wrong_candidates_quantiles"] = [float(x) for x in torch.quantile(mw, qs)]
        d["rowmax_own_vs_owner_quantiles"] = [float(x) for x in torch.quantile(Z["own"][ar, owner], qs)]
        d["share_opponent_rowmax_above_own_wrong_q99"] = float((mo > torch.quantile(mw, 0.99)).double().mean())
        d["share_opponent_rowmax_above_own_wrong_q999"] = float((mo > torch.quantile(mw, 0.999)).double().mean())
        zw1 = Z["S1e"].clone()
        zw1[ar, owner] = -9.0
        base = game_scores(Z["own"], owner)
        sc = {"own": base}
        for k in ALL:
            sc[k] = game_scores(Z[k], owner)
            for w in WGRID:
                sc[f"own+{w}*{k}"] = game_scores(Z["own"] + w * Z[k], owner)
        keep[pool] = (sc, owner_np, C)
        res["pools"][pool] = {"games": G, "candidates": C, "games_per_candidate": M, "single_network_mean": {k: float(np.mean(v)) for k, v in single.items()},
                              "diagnostics": d, "score": {k: float(v.mean()) for k, v in sc.items()}}
        log(f"{pool}: {K} networks together: own {base.mean():.4f}; alone " + "  ".join(f"{k} {sc[k].mean():.4f}" for k in SIGNALS))
        log(f"{pool}: share of games whose owner has a standardised S1e above " + ", ".join(f"{t:g}: {float((Z['S1e'][ar, owner] > t).double().mean()):.4f}" for t in HINGE)
            + "; any wrong candidate above " + ", ".join(f"{t:g}: {float(((zw1 > t).any(1)).double().mean()):.4f}" for t in HINGE))
        log(f"{pool}: nearest opponent vector is a game of the same candidate {d['nearest_opponent_same_candidate']:.4f} (nothing there: {d['null_nearest']:.4f}; own side {d['nearest_own_same_candidate']:.4f}); "
            f"same-candidate opponent pairs above the other-candidate quantiles .99/.999/.9999: " + "/".join(f"{x:.4f}" for x in d["opponent_pairs_same_candidate_share_above"][2:]))
        log(f"{pool}: opponent row maximum against candidates above the own-side wrong-candidate q99 / q999: {d['share_opponent_rowmax_above_own_wrong_q99']:.4f} / {d['share_opponent_rowmax_above_own_wrong_q999']:.4f} (nothing there: 0.0100 / 0.0010)")
        del OOm, UUm, Z, acc, Y, same
        torch.cuda.empty_cache()
    # added to the own side: the weight chosen on the other pool
    res["added"] = {}
    for k in ALL:
        row = {}
        for pool, other in (("pub2", "priv2"), ("priv2", "pub2")):
            sc_o = keep[other][0]
            w = max(WGRID, key=lambda x: sc_o[f"own+{x}*{k}"].mean())
            sc, own_np, C = keep[pool]
            row[pool] = {"weight_from_other_pool": w, "diff": boot(sc[f"own+{w}*{k}"] - sc["own"], own_np, C, a.boot, rng),
                         "by_weight": {str(x): float((sc[f"own+{x}*{k}"] - sc["own"]).mean()) for x in WGRID}}
        res["added"][k] = row
        log(f"own + {k}: " + "; ".join(f"{p} w={r['weight_from_other_pool']} {r['diff'][0]:+.4f} [{r['diff'][1]:+.4f}, {r['diff'][2]:+.4f}]" for p, r in row.items())
            + " | by weight (pub2) " + " ".join(f"{x}:{v:+.4f}" for x, v in row["pub2"]["by_weight"].items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
