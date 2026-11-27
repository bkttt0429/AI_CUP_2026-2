#!/usr/bin/env python
"""Fifth simulation of the v2 added pools (METHOD_NOTES 2.31): the sequence model fine-tuned on the pool's own candidates
(scripts/pid_v2_seq_finetune.py) against the same network as it is. Pools: fold 1-4, seed 0, as pid_v2_sim2.py; for every
fine-tune length t the sequence component uses the vectors after t steps, the other components are unchanged, and the weights
are fitted on the other folds' pools with the same t.
Fixed before the run: t = 0 must reproduce the reference; the adopted length is the one with the highest cross-fitted score,
and only if its gain over t = 0 has a player-bootstrap CI above 0 and at least 3 of the 4 pools gain."""
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
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 100, 300, 600, 1000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    emb_of = {k: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / f"emb2_p{k}.pt", map_location="cpu", weights_only=False)) for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v_ in (1, 2, 3, 4):
        z = np.load(ROOT / "data" / "seq" / f"fold{v_}.npz")
        seq0 = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        cands, questions, owner, block, sgfs, info = build(g, a.seed)
        moves = parse_all(sgfs, a.workers)
        S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb_of["12" if v_ in (1, 2) else "34"], which=COMPS, seq=seq0)
        players = sorted(cands)
        r = {"Z": {k: standardise(x) for k, x in S.items()}, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_, "seq": {}}
        for t in a.steps:
            y = np.load(Path(a.ft_dir) / f"sim_f{v_}_s{a.seed}_t{t}.npz")
            vec = dict(zip(y["key"].astype(str), y["emb"].astype(np.float32)))
            r["seq"][t] = standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=vec)["seq"])
        runs.append(r)
        print(f"fold {v_}: seq alone stored {score(topk(r['Z']['seq'], 5), r['truth']).mean():.4f}; after t steps " + " ".join(f"{t}: {score(topk(r['seq'][t], 5), r['truth']).mean():.4f}" for t in a.steps)
              + f" ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, t: [r["seq"][t] if c == "seq" else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}}, {}, {}
    for t in a.steps:
        sc, ws = [], []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, t) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, t)), 5), r["truth"]))
            ws.append(fus.w.tolist())
        per_q[t], per_pool[t] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][str(t)] = {"score": float(per_q[t].mean()), "per_pool": per_pool[t], "by_n": {str(m): float(per_q[t][nn == m].mean()) for m in SIZES},
                               "seq_alone": float(np.mean([score(topk(r["seq"][t], 5), r["truth"]).mean() for r in runs])), "weights_mean": dict(zip(COMPS, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"t = {t}: fused {out['arms'][str(t)]['score']:.4f} (seq alone {out['arms'][str(t)]['seq_alone']:.4f})  by n {({k: round(x, 3) for k, x in out['arms'][str(t)]['by_n'].items()})}  "
              f"pools {' '.join(f'{x:.3f}' for x in per_pool[t])}  seq weight {out['arms'][str(t)]['weights_mean']['seq']}, emb {out['arms'][str(t)]['weights_mean']['emb']}", flush=True)
    passing = []
    for t in a.steps:
        if t == 0:
            continue
        d = paired_player_bootstrap(per_q[t], per_q[0], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[t], per_pool[0])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 3)
        out["comparisons"][f"{t} vs 0"] = d
        if d["pass"]:
            passing.append(t)
        print(f"t = {t} vs 0: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/4 -> {'PASS' if d['pass'] else 'no'}", flush=True)
    best = max(passing, key=lambda t: out["arms"][str(t)]["score"]) if passing else 0
    out["chosen_steps"] = best
    if best:
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, best) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["weights_all_pools"] = dict(zip(COMPS, fus.w.tolist()))
        print(f"chosen: {best} steps; weights on all four pools {dict(zip(COMPS, np.round(fus.w, 3).tolist()))}", flush=True)
    else:
        print("chosen: no fine-tune", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
