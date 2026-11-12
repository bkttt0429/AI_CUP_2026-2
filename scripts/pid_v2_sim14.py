#!/usr/bin/env python
"""Single-network comparison of sequence networks on the simulated v2 pools (METHOD_NOTES 2.39): which training recipe gives
the better network for players it has not seen. For every network name (fold vectors data/seq/fold<v><name>.npz of
pid_v2_seq_embed.py --stage folds) and every pool of --folds x --seeds: the sequence component alone, and fused with the base
components (weights fitted on the other folds' pools). No pool fine-tune (the recipes are compared as trained).
Comparisons: every name against --ref, paired player bootstrap. Rule (fixed before the run): a recipe is better than the
reference when the CI of the fused gain is above 0 and at least three quarters of the pools gain.
  --names b=_nb r1=_nr1 r1e=_nr1e m90=_m90s0e ...
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
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    names = dict(x.split("=", 1) for x in a.names)
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: load(f"emb2_p{k}.pt") for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    runs = []
    for v_ in a.folds:
        V = {k: vecs(ROOT / "data" / "seq" / f"fold{v_}{sfx}.npz") for k, sfx in names.items()}
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")  # the pools of the earlier simulations (same games drawn)
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2["12" if v_ in (1, 2) else "34"], which=BASE)
            players = sorted(cands)
            truth = np.array([players.index(p) for p in owner])
            Z = {c: standardise(x) for c, x in S.items()}
            seq = {k: standardise(v2.components(cands, questions, moves, sgfs, cfg, which=("seq",), seq=V[k])["seq"]) for k in names}
            runs.append({"Z": Z, "seq": seq, "truth": truth, "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_})
            print(f"pool fold {v_} seed {s}: seq alone " + "  ".join(f"{k} {float(score(topk(seq[k], 5), truth).mean()):.3f}" for k in names) + f" ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    out, alone, fused, pools_f = {"names": names, "ref": a.ref, "folds": a.folds, "arms": {}, "comparisons": {}}, {}, {}, {}
    for k in names:
        mats = lambda r: [r["seq"][k] if c == "seq" else r["Z"][c] for c in COMPS]  # noqa: E731
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]] if len(a.folds) > 1 else [o for o in runs if o is not r]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r)), 5), r["truth"]))
        fused[k], pools_f[k] = np.concatenate(sc), [float(x.mean()) for x in sc]
        alone[k] = np.concatenate([score(topk(r["seq"][k], 5), r["truth"]) for r in runs])
        out["arms"][k] = {"seq_alone": float(alone[k].mean()), "fused": float(fused[k].mean()), "fused_per_pool": pools_f[k], "fused_by_n": {str(m): float(fused[k][nn == m].mean()) for m in SIZES},
                          "alone_by_n": {str(m): float(alone[k][nn == m].mean()) for m in SIZES}}
        print(f"{k}: seq alone {alone[k].mean():.4f}, fused {fused[k].mean():.4f}  by n {({m_: round(x, 3) for m_, x in out['arms'][k]['fused_by_n'].items()})}", flush=True)
    for k in names:
        if k == a.ref:
            continue
        da, df = paired_player_bootstrap(alone[k], alone[a.ref], pid, n=2000), paired_player_bootstrap(fused[k], fused[a.ref], pid, n=2000)
        up = int(sum(x > y for x, y in zip(pools_f[k], pools_f[a.ref])))
        ok = bool(df["ci95"][0] > 0 and up >= 0.75 * len(runs))
        out["comparisons"][f"{k} vs {a.ref}"] = {"alone": da, "fused": df, "pools_positive": up, "pass": ok}
        print(f"{k} vs {a.ref}: seq alone {da['diff']:+.4f} [{da['ci95'][0]:+.4f}, {da['ci95'][1]:+.4f}]; fused {df['diff']:+.4f} [{df['ci95'][0]:+.4f}, {df['ci95'][1]:+.4f}], pools up {up}/{len(runs)} -> {'BETTER' if ok else 'no'}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
