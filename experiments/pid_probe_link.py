#!/usr/bin/env python
"""Are the candidates of the official pools PEOPLE WE ALREADY KNOW under another id? (METHOD_NOTES 2.166)

The file checks of 11-04 / 11-11 established that no official GAME is in the training set. Whether an official CANDIDATE
is the same person as a training player (another anonymous id, other games), or as a candidate of the original pool of
the same side, was never measured. If it were so, that candidate would have dozens to hundreds of further reference
games - the one thing the bottleneck analysis says is worth a lot (notes 2.139: reference games 4 -> 7 = +0.045).
Measurement only, no answer of any question is used, fold 0 is not read, the two sides are never mixed for a decision.
  1  every competition training game of folds 1-4 is embedded with K networks that were trained on these players
     (o22Ss0-3); one prototype per player
  2  POSITIVE CONTROL: the held-out games of training players (never trained on; results/m82/held_keys.json), up to 10
     per player, against the prototypes -> what "a known person's unseen games" look like (own rank, cosine, row z)
  3  the candidates of the added pools (10 reference games, vectors before any pool fine-tune) and of the original
     pools (the 10-game sets of the old-pool simulation) against the training prototypes: the same statistics
  4  pool against pool: the added pool's candidates against the original pool's candidates of the SAME side (control:
     the other side, and an original candidate's own question games against his own 10-game set)
Printed: quantiles of the best cosine and of its row z per group, and how many candidates lie above the positive
control's 5 % / 25 % quantile (a linked candidate would lie there; an unlinked one lies with the bulk).
From the AI_CUP_2026-1 checkout, one GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import pid_seq_train as ST  # noqa: E402
import pid_v2_seq_finetune as FT  # noqa: E402
import aicup_pid  # noqa: E402,F401
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402

M = ST.M
QS = (5, 25, 50, 75, 95, 99)


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def stats(S, own=None):
    """Per row of a similarity matrix: best column, its cosine, row z of the best; with `own`: rank and z of the own column."""
    mu, sd = S.mean(1), S.std(1) + 1e-12
    best = S.argmax(1)
    out = {"best": best, "cos": S.max(1), "z": (S.max(1) - mu) / sd}
    if own is not None:
        o = S[np.arange(len(S)), own]
        out.update(own_rank=(S > o[:, None]).sum(1), own_cos=o, own_z=(o - mu) / sd)
    return out


def q(x):
    return [round(float(v), 3) for v in np.percentile(x, QS)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--nets", nargs="+", default=["o22Ss0", "o22Ss1", "o22Ss2", "o22Ss3"])
    ap.add_argument("--alias", nargs="+", default=["o22Fs0", "o22Fs1", "o22Fs2", "o22Fs3"], help="the names the pool vectors of these networks are stored under")
    ap.add_argument("--ft-dir", default=str(ROOT / "data" / "seq_ft"))
    ap.add_argument("--old-dir", default=str(ROOT / "data" / "pid_v2_seq_ft_oldsim"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--protos", default=None, help="save the training prototypes (and the held-out query prototypes of the control) here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    T = Path(a.tests_dir)
    G = ST.load_games(a.smoke, log)  # competition games only; the parts of fold 0 are never embedded below
    held = set(json.load(open(M.M82 / "held_keys.json")))
    is_held = np.array([k in held for k in G["key"]])
    fold = G["fold"]
    ok = (fold >= 1) & (fold <= 4)
    players, owner = np.unique(G["pid"], return_inverse=True)
    n_tr = np.bincount(owner[ok & ~is_held], minlength=len(players))
    keep_p = np.flatnonzero(n_tr >= 8)
    col = np.full(len(players), -1)
    col[keep_p] = np.arange(len(keep_p))
    rows_tr = np.flatnonzero(ok & ~is_held & (col[owner] >= 0))
    rows_he = np.flatnonzero(ok & is_held & (col[owner] >= 0))
    log(f"training players (folds 1-4, at least 8 trained games): {len(keep_p)}; trained games {len(rows_tr)}, held-out games {len(rows_he)} of {len(np.unique(owner[rows_he]))} players; fold 0 games in memory but not used: {int((fold == 0).sum())}")
    # pools: candidates' 10-game sets (and, for the original pools, the simulated questions as the same-person control)
    pools = {}
    for tag, pool in (("public", "pub2"), ("private", "priv2")):
        cands, _q, _ids, _sgf = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
        pools[pool] = {"names": sorted(cands), "sets": [[k for k, _ in cands[p]] for p in sorted(cands)], "file": lambda al, pool=pool: Path(a.ft_dir) / f"official_{pool}_net-1_{al}e_t0.npz"}
    for tag, side in (("public", "pub"), ("private", "priv")):
        z = np.load(Path(a.old_dir) / f"oldsim_{side}_s0_net-1_{a.alias[0]}e_t0.npz")
        cands, questions, truth, _sgf = v2.original_sim(T / f"player_identification_candidates_{tag}.csv", set(z["key"].astype(str).tolist()), 0)
        names = sorted(cands)
        own_q = {p: [] for p in names}
        for qg, p in zip(questions, truth):
            own_q[p] += [k for k, _ in qg]
        pools[side] = {"names": names, "sets": [[k for k, _ in cands[p]] for p in names], "other": [own_q[p] for p in names], "file": lambda al, side=side: Path(a.old_dir) / f"oldsim_{side}_s0_net-1_{al}e_t0.npz"}
    acc, keep_store = {}, {}
    for net_tag, al in zip(a.nets, a.alias):
        net = FT.load_net(Path(a.ckpt_dir) / f"ckpt_-1_{net_tag}e.pt")
        order = np.sort(np.concatenate([rows_tr, rows_he]))
        E = FT.embed(net, G, order, batch=2048)
        e_tr = E[np.searchsorted(order, rows_tr)]
        proto = np.zeros((len(keep_p), E.shape[1]), np.float64)
        np.add.at(proto, col[owner[rows_tr]], e_tr)
        U = unit(proto)
        # positive control: held-out games of known players (at most 10 each)
        hp = col[owner[rows_he]]
        e_he = E[np.searchsorted(order, rows_he)]
        qp = np.zeros_like(proto)
        cnt = np.zeros(len(keep_p), int)
        for i in np.random.default_rng(0).permutation(len(hp)):
            if cnt[hp[i]] < 2:
                qp[hp[i]] += e_he[i]
                cnt[hp[i]] += 1
        ctrl = np.flatnonzero(cnt >= (1 if a.smoke else 2))
        if not len(ctrl):  # a start-up check on a few parts may hold no held-out game: any ten players stand in
            ctrl = np.arange(min(10, len(keep_p)))
            qp[ctrl] = proto[ctrl]
        acc.setdefault("control", []).append((unit(qp[ctrl]) @ U.T, ctrl, cnt[ctrl]))
        keep_store.setdefault("U", []).append(U.astype(np.float32)); keep_store.setdefault("H", []).append(unit(qp[ctrl]).astype(np.float32)); keep_store["ctrl"] = ctrl
        protos = {}
        for name, P in pools.items():
            z = np.load(P["file"](al))
            vec = dict(zip(z["key"].astype(str).tolist(), z["emb"].astype(np.float64)))
            protos[name] = unit(np.stack([np.sum([vec[k] for k in s], 0) for s in P["sets"]]))
            if "other" in P:
                protos[name + "_q"] = unit(np.stack([np.sum([vec[k] for k in s], 0) for s in P["other"]]))
            acc.setdefault(name + " vs training", []).append(protos[name] @ U.T)
        for new, old, cross in (("pub2", "pub", "priv"), ("priv2", "priv", "pub")):
            acc.setdefault(f"{new} vs original {old} (same side)", []).append(protos[new] @ protos[old].T)
            acc.setdefault(f"{new} vs original {cross} (other side, null)", []).append(protos[new] @ protos[cross].T)
            acc.setdefault(f"{new} vs {new} (other candidates, null)", []).append(protos[new] @ protos[new].T - 2 * np.eye(len(protos[new])))
        for side in ("pub", "priv"):
            acc.setdefault(f"original {side}: own question games vs own 10-game set (same person, unseen by the network)", []).append(protos[side + "_q"] @ protos[side].T)
        log(f"network {net_tag}: {len(order)} training games embedded, {len(keep_p)} prototypes, control players {len(ctrl)}")
        del net, E
    rank_of = np.array([k.split(":")[0] for k in G["key"]])
    first = np.zeros(len(keep_p), np.int64)
    first[col[owner[rows_tr]][::-1]] = rows_tr[::-1]  # the first trained game of every player
    pr = rank_of[first]
    if a.protos:
        np.savez_compressed(a.protos, players=players[keep_p].astype(str), n_games=n_tr[keep_p], rank=pr, U=np.stack(keep_store["U"]), H=np.stack(keep_store["H"]), ctrl=keep_store["ctrl"], nets=np.array(a.nets), alias=np.array(a.alias))
        log(f"wrote {a.protos}")
    out = {"nets": a.nets, "groups": {}}
    S = np.mean([x[0] for x in acc["control"]], 0)
    ctrl = acc["control"][0][1]
    st = stats(S, own=ctrl)
    out["groups"]["control: held-out games of known training players"] = {"n": int(len(ctrl)), "own_rank_0": float((st["own_rank"] == 0).mean()), "own_cos": q(st["own_cos"]), "own_z": q(st["own_z"])}
    lo5, lo25 = np.percentile(st["own_cos"], 5), np.percentile(st["own_cos"], 25)
    z5, z25 = np.percentile(st["own_z"], 5), np.percentile(st["own_z"], 25)
    print(f"\nPOSITIVE CONTROL - {len(ctrl)} training players, their 2 held-out games against {S.shape[1]} prototypes: own player first in {(st['own_rank'] == 0).mean():.3f}; "
          f"own cosine {QS} %: {q(st['own_cos'])}; own row z: {q(st['own_z'])}", flush=True)
    print(f"   a linked candidate should have a best cosine above about {lo25:.3f} (25 % point; 5 % point {lo5:.3f}) and a row z above about {z25:.1f} (5 % point {z5:.1f})\n")
    for name, mats in acc.items():
        if name == "control":
            continue
        S = np.mean(mats, 0)
        same = "own question games" in name
        st = stats(S, own=np.arange(len(S)) if same else None)
        cos, z = (st["own_cos"], st["own_z"]) if same else (st["cos"], st["z"])
        res = {"n": int(len(S)), "cos": q(cos), "z": q(z), "above_control_cos5": int((cos >= lo5).sum()), "above_control_cos25": int((cos >= lo25).sum()), "above_control_z5": int((z >= z5).sum()), "above_control_z25": int((z >= z25).sum())}
        if same:
            res["own_rank_0"] = float((st["own_rank"] == 0).mean())
        if "vs training" in name:
            top = np.argsort(-z)[:5]
            res["top"] = [[pools[name.split()[0]]["names"][i], str(players[keep_p[st["best"][i]]])[:8], round(float(cos[i]), 3), round(float(z[i]), 2)] for i in top]
        out["groups"][name] = res
        print(f"{name}: n {len(S)}; best cosine {QS} %: {q(cos)}; row z: {q(z)}; above the control's 5 % / 25 % point: cosine {res['above_control_cos5']} / {res['above_control_cos25']}, z {res['above_control_z5']} / {res['above_control_z25']}"
              + (f"; own first {res['own_rank_0']:.3f}" if same else ""), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
