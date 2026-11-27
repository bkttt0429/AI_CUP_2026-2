#!/usr/bin/env python
"""Seventh simulation of the v2 added pools (METHOD_NOTES 2.33): recipes for the pool fine-tune of the sequence network, against
the adopted one (c0 = 100 steps, lr 1e-4, cosine horizon 1000). Pools as pid_v2_sim2.py; the sequence component uses the vectors
of scripts/pid_v2_seq_finetune.py for the named recipe, the other components are unchanged, weights from the other folds' pools.
  --arms name=<file suffix> ...   e.g. c0=_t100 c1=_c1_t50 c4a=_c4_t300 (files <ft-dir>/sim_f<v>_s<seed><suffix>.npz)
Screening (seed 0, four pools): the recipe with the highest cross-fitted score; if it is not c0 and beats it by >= 0.002, it is
run on the seed-1 pools too. Adoption (seeds 0 and 1, eight pools): gain over c0 with a player-bootstrap CI above 0 and at
least 6 of 8 pools up."""
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
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", default="c0")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    arms = dict(x.split("=", 1) for x in a.arms)
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    emb_of = {k: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / f"emb2_p{k}.pt", map_location="cpu", weights_only=False)) for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    runs = []
    for v_ in (1, 2, 3, 4):
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb_of["12" if v_ in (1, 2) else "34"], which=BASE)
            players = sorted(cands)
            r = {"Z": {k: standardise(x) for k, x in S.items()}, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_, "seq": {}}
            for name, suffix in arms.items():
                r["seq"][name] = standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vecs(Path(a.ft_dir) / f"sim_f{v_}_s{s}{suffix}.npz"))["seq"])
            runs.append(r)
            print(f"pool fold {v_} seed {s}: seq alone " + " ".join(f"{k} {score(topk(x, 5), r['truth']).mean():.4f}" for k, x in r["seq"].items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, name: [r["seq"][name] if c == "seq" else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}, "seeds": a.seeds}, {}, {}
    for name in arms:
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, name) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, name)), 5), r["truth"]))
        per_q[name], per_pool[name] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][name] = {"score": float(per_q[name].mean()), "per_pool": per_pool[name], "by_n": {str(m): float(per_q[name][nn == m].mean()) for m in SIZES},
                             "seq_alone": float(np.mean([score(topk(r["seq"][name], 5), r["truth"]).mean() for r in runs]))}
        print(f"arm {name}: fused {out['arms'][name]['score']:.4f} (seq alone {out['arms'][name]['seq_alone']:.4f})  by n {({k: round(x, 3) for k, x in out['arms'][name]['by_n'].items()})}", flush=True)
    for name in arms:
        if name == a.ref:
            continue
        d = paired_player_bootstrap(per_q[name], per_q[a.ref], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[name], per_pool[a.ref])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 0.75 * len(runs))
        out["comparisons"][f"{name} vs {a.ref}"] = d
        print(f"{name} vs {a.ref}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)} -> {'PASS' if d['pass'] else 'no'}", flush=True)
    best = max(arms, key=lambda k: out["arms"][k]["score"])
    out["best"] = best
    print(f"best: {best} ({out['arms'][best]['score'] - out['arms'][a.ref]['score']:+.4f} over {a.ref})", flush=True)
    for name in arms:
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, name) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["arms"][name]["weights_all_pools"] = dict(zip(COMPS, fus.w.tolist()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
