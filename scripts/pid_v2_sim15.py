#!/usr/bin/env python
"""How much is one more fold of training players worth to the sequence component, at the ENSEMBLE level and after the pool
fine-tune (METHOD_NOTES 2.40)?  The official files average 20 networks: 4 that saw every training player (net -1 of the rank
session's M88 / M90 runs) and 16 that each miss one fold. If more training players make a network better on unseen players,
more all-player networks (new seeds) would lift the official ensemble; no simulation can hold players out of an all-player
network, so the effect is measured one step down: networks trained on 3 folds (scripts/pid_seq_train.py, the M90 recipe "b",
and the variants "r1" / "r2", all about equal next to the base components) against the M90 networks trained on 4 folds, on
the simulated pools of the fold that none of them saw.

Arms (sequence component = mean of the row-standardised matrices of the arm's networks, weights cross-fitted on the other
fold's pools; --arms name=net+net+...); every arm is scored without fine-tune (t0) and after --steps fine-tune steps.
Comparisons (--compare a:b ...): paired player bootstrap of the per-question scores, pools up.
  --nets name=<fold file tag>:<fine-tune tag>   fold vectors data/seq/fold<v><tag>.npz, fine-tuned <ft-dir>/sim_f<v>_s<seed><fttag>_t<step>.npz
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    arms = {x.split("=", 1)[0]: x.split("=", 1)[1].split("+") for x in a.arms}
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    ft_file = lambda v_, s, name, t: ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz" if t == 0 else Path(a.ft_dir) / f"sim_f{v_}_s{s}{nets[name][1]}_t{t}.npz"  # noqa: E731

    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")  # the same games as every other simulation of this fold
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            seqm = {(name, t): standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(ft_file(v_, s, name, t)))["seq"]) for name in nets for t in (0, a.steps)}
            players = sorted(cands)
            r = {"Z": {c: standardise(x) for c, x in S.items()}, "seq": seqm, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_}
            runs.append(r)
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: seq alone (0 / {a.steps} steps) " + "  ".join(f"{name} {al(seqm[(name, 0)]):.3f}/{al(seqm[(name, a.steps)]):.3f}" for name in nets) + f" ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])

    def mats(r, names, t):
        seq = standardise(np.mean([r["seq"][(n_, t)] for n_ in names], axis=0))
        return [seq if c == "seq" else r["Z"][c] for c in COMPS]

    out, per_q, per_pool = {"nets": {k: list(x) for k, x in nets.items()}, "steps": a.steps, "folds": a.folds, "arms": {}, "comparisons": {}}, {}, {}
    for arm, names in arms.items():
        for t in (0, a.steps):
            key, sc, alone = f"{arm}@{t}", [], []
            for r in runs:
                others = [o for o in runs if o["fold"] != r["fold"]]
                fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, names, t) for o in others])], np.concatenate([o["truth"] for o in others]))
                sc.append(score(topk(fus.scores(mats(r, names, t)), 5), r["truth"]))
                alone.append(float(score(topk(mats(r, names, t)[-1], 5), r["truth"]).mean()))
            per_q[key], per_pool[key] = np.concatenate(sc), [float(x.mean()) for x in sc]
            out["arms"][key] = {"nets": names, "score": float(per_q[key].mean()), "seq_alone": float(np.mean(alone)), "per_pool": per_pool[key], "by_n": {str(m): float(per_q[key][nn == m].mean()) for m in SIZES}}
            print(f"arm {key} ({'+'.join(names)}): seq alone {out['arms'][key]['seq_alone']:.4f}, fused {out['arms'][key]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][key]['by_n'].items()})}", flush=True)
    for pair in a.compare:
        x, y = pair.split(":")
        for t in (0, a.steps):
            d = paired_player_bootstrap(per_q[f"{x}@{t}"], per_q[f"{y}@{t}"], pid, n=2000)
            d["pools_positive"] = int(sum(p > q for p, q in zip(per_pool[f"{x}@{t}"], per_pool[f"{y}@{t}"])))
            d["by_n"] = {str(m): float(per_q[f"{x}@{t}"][nn == m].mean() - per_q[f"{y}@{t}"][nn == m].mean()) for m in SIZES}
            out["comparisons"][f"{x} vs {y} @{t}"] = d
            print(f"{x} vs {y} @{t}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)}; by n {({k: round(v, 4) for k, v in d['by_n'].items()})}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
