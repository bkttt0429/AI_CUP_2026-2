#!/usr/bin/env python
"""Opponent side as a new component for the v2 added pools (METHOD_NOTES 2.38). The sequence identity model also gives a
vector for the OTHER side of every game (scripts/pid_v2_seq_embed.py --flip): who the player was playing against. If a player
meets the same opponents again (regular partners, rematches) or the same kind of opponents, the opponents of a question's
games resemble the opponents of its owner's candidate games.
Components (question q with games g, candidate c with games h; o(.) = opponent vector):
  opp_max    mean over g of max over h of cos(o(g), o(h))          the same opponent again
  opp_mean   cos(sum o(g), sum o(h))                                the same kind of opponents
Pools and reference as pid_v2_sim11.py's chosen arm (all networks after --steps of the pool fine-tune, emb2); arms add one
or both components to the reference, weights fitted on the other folds' pools. Rule (fixed before the run): an arm passes
when its gain has a player-bootstrap CI above 0 and at least 6 of the 8 pools gain.
Diagnostic per question game: the rank of the owner among the 600 candidates by max cos (chance: top-1 1/600).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
ARMS = {"ref": (), "max": ("opp_max",), "mean": ("opp_mean",), "both": ("opp_max", "opp_mean")}


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def opponent_components(cands, questions, opp):
    """{"opp_max", "opp_mean"}: [Q, C]; per_game [total question games, C] of max cos (for the diagnostic)."""
    players = sorted(cands)
    H = [unit(np.stack([opp[k] for k, _ in cands[p]])) for p in players]
    Hm = unit(np.stack([h.sum(0) for h in H]))
    mx, mean, per_game = [], [], []
    for q in questions:
        G = unit(np.stack([opp[k] for k, _ in q]))
        m = np.stack([(G @ h.T).max(1) for h in H], axis=1)  # [n, C]
        per_game.append(m)
        mx.append(m.mean(0))
        mean.append(unit(G.sum(0)) @ Hm.T)
    return {"opp_max": np.stack(mx), "opp_mean": np.stack(mean)}, per_game


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--opp-name", default="_m90s0e_opp", help="data/seq/fold<v><name>.npz of pid_v2_seq_embed.py --flip")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731

    def ft_file(v_, s, name):
        if a.steps == 0:
            return ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz"
        if name == "m88":
            return Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if a.steps == 50 else f"sim_f{v_}_s{s}_t100.npz")
        return Path(a.ft_dir) / f"sim_f{v_}_s{s}{nets[name][1]}_t{a.steps}.npz"

    runs, diag = [], []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        opp = vecs(ROOT / "data" / "seq" / f"fold{v_}{a.opp_name}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys()) & games.game_key.isin(opp.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            seq = standardise(np.mean([standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(ft_file(v_, s, name)))["seq"]) for name in nets], axis=0))
            O, per_game = opponent_components(cands, questions, opp)
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            Z = {c: standardise(x) for c, x in S.items()}
            Z.update(seq=seq, **{k: standardise(x) for k, x in O.items()})
            runs.append({"Z": Z, "truth": truth, "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_})
            rank = np.concatenate([(m > m[:, [t]]).sum(1) for m, t in zip(per_game, truth)])  # owner's rank per question game (0 = best)
            own = np.concatenate([m[:, t] for m, t in zip(per_game, truth)])
            other = np.concatenate([np.delete(m, t, axis=1).ravel() for m, t in zip(per_game[:200], truth[:200])])
            diag.append({"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()), "top30": float((rank < 30).mean()), "own_q": np.percentile(own, [50, 90, 99]).tolist(),
                         "other_q": np.percentile(other, [50, 90, 99, 99.9]).tolist(), "own_above_other_p999": float((own > np.percentile(other, 99.9)).mean())})
            al = lambda M: float(score(topk(M, 5), truth).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: alone opp_max {al(O['opp_max']):.3f} opp_mean {al(O['opp_mean']):.3f} (chance {sum(np.exp(-i) for i in range(5)) / len(players):.4f}); per question game: owner top-1 "
                  f"{diag[-1]['top1']:.4f} top-5 {diag[-1]['top5']:.4f} top-30 {diag[-1]['top30']:.3f} (chance {1 / len(players):.4f} / {5 / len(players):.4f} / {30 / len(players):.3f}); max cos owner median/p90/p99 "
                  f"{[round(x, 3) for x in diag[-1]['own_q']]}, others median/p90/p99/p99.9 {[round(x, 3) for x in diag[-1]['other_q']]}; owner above others' p99.9: {diag[-1]['own_above_other_p999']:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out, per_q, per_pool = {"nets": {k: list(x) for k, x in nets.items()}, "steps": a.steps, "opp": a.opp_name, "diagnostic": diag, "arms": {}, "comparisons": {}}, {}, {}
    for arm, extra in ARMS.items():
        comps = (*COMPS, *extra)
        mats = lambda r: [r["Z"][c] for c in comps]  # noqa: E731
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r)), 5), r["truth"]))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES}, "weights_all_pools": dict(zip(comps, fus.w.tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  weights {({c: round(w, 2) for c, w in zip(comps, fus.w.tolist()) if c in ('seq', 'emb', *extra)})}", flush=True)
    for arm in ARMS:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 0.75 * len(runs))
        d["by_n"] = {str(m): float(per_q[arm][nn == m].mean() - per_q["ref"][nn == m].mean()) for m in SIZES}
        out["comparisons"][f"{arm} vs ref"] = d
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)} -> {'PASS' if d['pass'] else 'no'}; by n {({k: round(x, 4) for k, x in d['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
