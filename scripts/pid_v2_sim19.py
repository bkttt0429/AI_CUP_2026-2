#!/usr/bin/env python
"""Follow-ups of the error analysis (METHOD_NOTES 2.48) on the simulated pools, multi-network ft50 reference:
  A  every component alone, and the fusion, on the 1-game questions by the owner's colour (which evidence is missing for White)
  B  colour-matched prototypes of the sequence component (a game against the candidate's games of the SAME colour; a candidate
     with fewer than --min-col games of that colour keeps its full prototype): as one more component (+cc), instead of the
     sequence component (cc), and fusion weights fitted separately for Black and for White 1-game questions (bycol)
  C  ceiling of a rank signal: candidates whose TRUE rank class is more than k classes from the owner's are removed
     (k = 1, 0) - what a perfect "same strength" filter would add
  D  single networks by the owner's rank class: --pairs a:b ... of fold-vector tags (e.g. the network with external players
     against its control), sequence component alone, per class
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
from aicup.metric import LEVELS  # noqa: E402
from aicup_pid import v2  # noqa: E402
from aicup_pid.components import parse_all  # noqa: E402
from aicup_pid.embed import PlayerEmbedding  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731


def colour_matched(cands, questions, vec, min_col):
    players = sorted(cands)
    full = unit(np.stack([np.sum([vec[k] for k, _ in cands[p]], axis=0) for p in players]).astype(np.float64))
    proto = {}
    for col in "BW":
        P = full.copy()
        for i, p in enumerate(players):
            g = [vec[k] for k, c in cands[p] if c == col]
            if len(g) >= min_col:
                P[i] = unit(np.sum(g, axis=0).astype(np.float64))
        proto[col] = P
    return np.stack([np.mean([unit(vec[k].astype(np.float64)) @ proto[c].T for k, c in q], axis=0) for q in questions])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True, help="as pid_v2_sim11.py")
    ap.add_argument("--pairs", nargs="*", default=[], help="D: a:b fold-vector tags (data/seq/fold<v><tag>.npz), only for the folds that have both")
    ap.add_argument("--min-col", type=int, default=3)
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
    rank_of, lvl = dict(zip(games.player_id, games["rank"])), {r: i for i, r in enumerate(LEVELS)}
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    ft = lambda v_, s, name: Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if name == "m88" else f"sim_f{v_}_s{s}{nets[name][1]}_t50.npz")  # noqa: E731
    tags = sorted({t for pr in a.pairs for t in pr.split(":")})
    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        have = {t: vecs(ROOT / "data" / "seq" / f"fold{v_}{t}.npz") for t in tags if (ROOT / "data" / "seq" / f"fold{v_}{t}.npz").exists()}
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            Z = {c: standardise(x) for c, x in S.items()}
            V = [vecs(ft(v_, s, name)) for name in nets]
            one = lambda **kw: v2.components(cands, questions, moves, sgfs, cfg, **kw)  # noqa: E731
            Z["seq"] = standardise(np.mean([standardise(one(which=("seq",), seq=v)["seq"]) for v in V], axis=0))
            Z["cc"] = standardise(np.mean([standardise(colour_matched(cands, questions, v, a.min_col)) for v in V], axis=0))
            players = sorted(cands)
            r = {"Z": Z, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_,
                 "prank": np.array([lvl[rank_of[p]] for p in players]), "col1": np.array([q[0][1] for q in questions]),
                 "single": {t: standardise(one(which=("seq",), seq=have[t])["seq"]) for t in have},
                 "ncol": np.array([[sum(c == col for _, c in cands[p]) for col in "BW"] for p in players])}
            runs.append(r)
            print(f"pool fold {v_} seed {s}: candidates with fewer than {a.min_col} Black / White games {int((r['ncol'][:, 0] < a.min_col).sum())} / {int((r['ncol'][:, 1] < a.min_col).sum())} of {len(players)} ({time.time() - t0:.0f} s)", flush=True)
    nn_ = np.concatenate([r["n"] for r in runs])
    col = np.concatenate([r["col1"] for r in runs])
    orank = np.concatenate([r["prank"][r["truth"]] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out = {"levels": list(LEVELS), "A": {}, "B": {}, "C": {}, "D": {}}

    def cross(comps, group=None):
        """Per-question scores with cross-fitted weights; group(r) -> labels: one weight vector per label."""
        res = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            if group is None:
                fus = Fusion().fit([np.vstack([o["Z"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
                F = fus.scores([r["Z"][c] for c in comps])
            else:
                F, lab = np.zeros_like(r["Z"]["seq"]), group(r)
                for u in np.unique(lab):
                    fus = Fusion().fit([np.vstack([o["Z"][c][group(o) == u] for o in others]) for c in comps], np.concatenate([o["truth"][group(o) == u] for o in others]))
                    F[lab == u] = fus.scores([r["Z"][c][lab == u] for c in comps])
            res.append(F)
        return res

    sc_of = lambda Fs: np.concatenate([score(topk(F, 5), r["truth"]) for F, r in zip(Fs, runs)])  # noqa: E731
    F_ref = cross(COMPS)
    ref = sc_of(F_ref)
    one_b, one_w = (nn_ == 1) & (col == "B"), (nn_ == 1) & (col == "W")
    # A: components alone on the 1-game questions by colour
    for c in (*COMPS, "cc"):
        x = np.concatenate([score(topk(r["Z"][c], 5), r["truth"]) for r in runs])
        out["A"][c] = {"black": float(x[one_b].mean()), "white": float(x[one_w].mean())}
    out["A"]["fused"] = {"black": float(ref[one_b].mean()), "white": float(ref[one_w].mean())}
    print("A  1-game questions, Black / White owner: " + "  ".join(f"{c} {x['black']:.3f}/{x['white']:.3f}" for c, x in out["A"].items()), flush=True)
    # B: colour-matched prototypes and colour-specific weights
    arms = {"+cc": cross((*COMPS, "cc")), "cc": cross((*BASE, "cc")),
            "bycol": cross(COMPS, lambda r: np.where(r["n"] == 1, np.where(r["col1"] == "B", "1B", "1W"), "multi")),
            "bycol+cc": cross((*COMPS, "cc"), lambda r: np.where(r["n"] == 1, np.where(r["col1"] == "B", "1B", "1W"), "multi"))}
    print(f"B  reference {ref.mean():.4f}  by n {({int(m): round(float(ref[nn_ == m].mean()), 3) for m in SIZES})}", flush=True)
    for arm, Fs in arms.items():
        x = sc_of(Fs)
        d = paired_player_bootstrap(x, ref, pid, n=2000)
        pools = [float(score(topk(F, 5), r["truth"]).mean() - score(topk(G, 5), r["truth"]).mean()) for F, G, r in zip(Fs, F_ref, runs)]
        d.update(score=float(x.mean()), pools_positive=int(sum(p > 0 for p in pools)), by_n={str(m): float((x - ref)[nn_ == m].mean()) for m in SIZES}, one_black=float((x - ref)[one_b].mean()), one_white=float((x - ref)[one_w].mean()))
        out["B"][arm] = d
        print(f"B  {arm}: {x.mean():.4f}  vs reference {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)}; by n {({k: round(v_, 4) for k, v_ in d['by_n'].items()})}; "
              f"1-game Black {d['one_black']:+.4f}, White {d['one_white']:+.4f}", flush=True)
    # C: a perfect same-strength filter
    for k in (1, 0):
        x = np.concatenate([score(topk(np.where(np.abs(r["prank"][None] - r["prank"][r["truth"]][:, None]) <= k, F, -1e9), 5), r["truth"]) for F, r in zip(F_ref, runs)])
        out["C"][f"within_{k}"] = {"score": float(x.mean()), "diff": float((x - ref).mean()), "by_n": {str(m): float((x - ref)[nn_ == m].mean()) for m in SIZES}}
        print(f"C  only candidates within {k} class(es) of the owner's true class: {x.mean():.4f} ({(x - ref).mean():+.4f}); by n {({m: round(v_, 4) for m, v_ in out['C'][f'within_{k}']['by_n'].items()})}", flush=True)
    # D: single networks by the owner's class
    for pr in a.pairs:
        ta, tb = pr.split(":")
        rs = [r for r in runs if ta in r["single"] and tb in r["single"]]
        if not rs:
            continue
        xa = np.concatenate([score(topk(r["single"][ta], 5), r["truth"]) for r in rs])
        xb = np.concatenate([score(topk(r["single"][tb], 5), r["truth"]) for r in rs])
        rk = np.concatenate([r["prank"][r["truth"]] for r in rs])
        out["D"][pr] = {"pools": len(rs), "a": float(xa.mean()), "b": float(xb.mean()), "by_class": {LEVELS[i]: {"n": int((rk == i).sum()), "a": float(xa[rk == i].mean()), "b": float(xb[rk == i].mean())} for i in range(len(LEVELS))}}
        print(f"D  {ta} against {tb} ({len(rs)} pools), sequence component alone: {xa.mean():.4f} vs {xb.mean():.4f}; by class (a - b): "
              + "  ".join(f"{LEVELS[i]} {xa[rk == i].mean() - xb[rk == i].mean():+.3f}" for i in range(len(LEVELS))), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
