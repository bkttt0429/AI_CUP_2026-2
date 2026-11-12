#!/usr/bin/env python
"""Ninth simulation of the v2 added pools (METHOD_NOTES 2.33): the embedding on rows WITH phase means (emb3: 256-dim row + the
means of the player's plies in the opening / middle / late thirds, 640 dims; scripts/pid_v2_embed.py --phase-dir) against emb2.
The 8 single-fold pools of pid_v2_sim2.py; weights from the other folds' pools.
Arms against ref = base+seq with emb2 (fixed before the run; rule: CI above 0 and at least 6 of 8 pools gain):
  emb3     emb2 replaced by emb3 (one seed)        emb3x3   by the mean of three emb3 seeds
  emb2+3   emb3 added as a tenth component          emb2x5   five emb2 seeds (known: +0.0028, for reference)
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
ARMS = {"ref": (COMPS, {}), "emb3": (COMPS, {"emb": "emb3"}), "emb3x3": (COMPS, {"emb": "emb3x3"}), "emb2x5": (COMPS, {"emb": "emb2x5"}), "emb2+3": ((*COMPS, "emb3"), {})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--phase-dir", default="/work/u2507115/pid_v2/phase_rows")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: [load(f"emb2_p{k}.pt" if s == 0 else f"emb2_p{k}_s{s}.pt") for s in range(5)] for k in ("12", "34")}
    e3 = {k: [load(f"emb3_p{k}.pt" if s == 0 else f"emb3_p{k}_s{s}.pt") for s in range(3)] for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v_ in (1, 2, 3, 4):
        z = np.load(ROOT / "data" / "seq" / f"fold{v_}.npz")
        seq = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq.keys())]
        for s in (0, 1):
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            k = "12" if v_ in (1, 2) else "34"
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2[k][0], which=COMPS, seq=seq)
            keys = {x for games_ in (*cands.values(), *questions) for x, _ in games_}
            rows3 = v2.load_phase_rows(AICUP1, a.phase_dir, keep=keys) if not runs or True else None
            if set(rows3) != keys:
                raise ValueError(f"{len(keys) - len(rows3)} pool games without phase rows")
            one = lambda rr, m: standardise(v2.components(cands, questions, moves, sgfs, cfg, rr, m, which=("emb",))["emb"])  # noqa: E731
            m3 = [one(rows3, m) for m in e3[k]]
            var = {"emb3": m3[0], "emb3x3": standardise(np.mean(m3, axis=0)), "emb2x5": standardise(np.mean([one(rows, m) for m in e2[k]], axis=0))}
            players = sorted(cands)
            r = {"Z": {**{c: standardise(x) for c, x in S.items()}, "emb3": m3[0]}, "var": var, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]),
                 "owner": owner, "fold": v_}
            runs.append(r)
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: alone emb2 {al(r['Z']['emb']):.3f} emb3 {al(var['emb3']):.3f} emb3x3 {al(var['emb3x3']):.3f} emb2x5 {al(var['emb2x5']):.3f} ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, comps, spec: [r["var"][spec[c]] if c in spec else r["Z"][c] for c in comps]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}}, {}, {}
    for arm, (comps, spec) in ARMS.items():
        sc, ws = [], []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, comps, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, comps, spec)), 5), r["truth"]))
            ws.append(fus.w.tolist())
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "weights_mean": dict(zip(comps, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k_: round(x, 3) for k_, x in out['arms'][arm]['by_n'].items()})}  weights {out['arms'][arm]['weights_mean']}", flush=True)
    for arm in ARMS:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 6)
        out["comparisons"][f"{arm} vs ref"] = d
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/8 -> {'PASS' if d['pass'] else 'no'}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
