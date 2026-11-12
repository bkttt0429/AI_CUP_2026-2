#!/usr/bin/env python
"""Tenth simulation of the v2 added pools (METHOD_NOTES 2.34): the two changes that passed on their own, together, and the
weights of the next official file. 8 single-fold pools of pid_v2_sim2.py; weights from the other folds' pools.
  ref    sequence vectors after the 100-step pool fine-tune + emb2 (the 11-13 live configuration, one network)
  ft     the shorter fine-tune (--ft-suffix) + emb2            e3    100-step fine-tune + three emb3 seeds
  both   the shorter fine-tune + three emb3 seeds
Rule (fixed before the run): `both` is the next configuration if it beats ref with a player-bootstrap CI above 0 and at least
6 of 8 pools up; otherwise the better of ft / e3 that passes the same rule alone."""
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
ARMS = {"ref": ("seq100", "emb2"), "ft": ("seqS", "emb2"), "e3": ("seq100", "emb3x3"), "both": ("seqS", "emb3x3")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--ft-suffix", required=True, help="e.g. _g_t50")
    ap.add_argument("--phase-dir", default="/work/u2507115/pid_v2/phase_rows")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    e3 = {k: [load(f"emb3_p{k}.pt" if s == 0 else f"emb3_p{k}_s{s}.pt") for s in range(3)] for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    runs = []
    for v_ in (1, 2, 3, 4):
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in (0, 1):
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            k = "12" if v_ in (1, 2) else "34"
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2[k], which=BASE)
            keys = {x for games_ in (*cands.values(), *questions) for x, _ in games_}
            rows3 = v2.load_phase_rows(AICUP1, a.phase_dir, keep=keys)
            one = lambda **kw: v2.components(cands, questions, moves, sgfs, cfg, **kw)  # noqa: E731
            var = {"emb2": standardise(S["emb"]), "emb3x3": standardise(np.mean([standardise(one(rows=rows3, emb=m, which=("emb",))["emb"]) for m in e3[k]], axis=0)),
                   "seq100": standardise(one(which=("seq",), seq=vecs(Path(a.ft_dir) / f"sim_f{v_}_s{s}_t100.npz"))["seq"]),
                   "seqS": standardise(one(which=("seq",), seq=vecs(Path(a.ft_dir) / f"sim_f{v_}_s{s}{a.ft_suffix}.npz"))["seq"])}
            players = sorted(cands)
            runs.append({"Z": {c: standardise(x) for c, x in S.items()}, "var": var, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]),
                         "owner": owner, "fold": v_})
            print(f"pool fold {v_} seed {s} ready ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, spec: [r["var"][spec[0]] if c == "seq" else r["var"][spec[1]] if c == "emb" else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}, "ft_suffix": a.ft_suffix}, {}, {}
    for arm, spec in ARMS.items():
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, spec)), 5), r["truth"]))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, spec) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "weights_all_pools": dict(zip(COMPS, fus.w.tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k_: round(x, 3) for k_, x in out['arms'][arm]['by_n'].items()})}  weights {({c: round(w, 3) for c, w in out['arms'][arm]['weights_all_pools'].items()})}", flush=True)
    for arm in ("ft", "e3", "both"):
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
