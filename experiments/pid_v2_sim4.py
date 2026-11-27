#!/usr/bin/env python
"""Fourth simulation of the v2 added pools (METHOD_NOTES 2.31): adapting the game vectors to the pool with the pool's own
candidates (600 players x 10 games, ids given by the organisers; no question is used).

Within-class covariance normalisation, as in speaker verification: W = the covariance of the candidates' game vectors around
their own player's mean (6,000 vectors), shrunk towards a multiple of the identity; every vector is multiplied by W^(-1/2) and
renormalised, then prototypes and question vectors are formed as before. Directions in which one player's games scatter are
shrunk, in the pool's own distribution. Same single-fold pools and cross-fitting as pid_v2_sim2.py / sim3.py.
Arms (fixed before the run), each replacing the corresponding component of ref = base+seq:
  seq_w<s>   sequence vectors normalised, shrinkage s in 0.1 / 0.3 / 0.6
  emb_w<s>   embedding vectors normalised, s = 0.3
  both_w0.3  both
Rule: as sim3 (CI above 0 and at least 6 of 8 pools gain); the best passing arm goes to --fusion-out.
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
ARMS = {"ref": {}, "seq_w0.1": {"seq": 0.1}, "seq_w0.3": {"seq": 0.3}, "seq_w0.6": {"seq": 0.6}, "emb_w0.3": {"emb": 0.3}, "both_w0.3": {"seq": 0.3, "emb": 0.3}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--emb12", default=str(ROOT / "results" / "embed" / "emb2_p12.pt"))
    ap.add_argument("--emb34", default=str(ROOT / "results" / "embed" / "emb2_p34.pt"))
    ap.add_argument("--seq-dir", default=str(ROOT / "data" / "seq"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    base_doc = json.loads(Path(a.cfg_from).read_text())
    cfg = base_doc["cfg"]
    emb_of = {k: PlayerEmbedding.from_state(torch.load(p, map_location="cpu", weights_only=False)) for k, p in (("12", a.emb12), ("34", a.emb34))}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v_ in (1, 2, 3, 4):
        z = np.load(Path(a.seq_dir) / f"fold{v_}.npz")
        seq = dict(zip(z["key"].astype(str), z["emb"].astype(np.float32)))
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            emb = emb_of["12" if v_ in (1, 2) else "34"]
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb, which=COMPS, seq=seq)
            keys = sorted({k for games_ in (*cands.values(), *questions) for k, _ in games_})
            embv = dict(zip(keys, emb.embed(np.stack([rows[k] for k in keys]))))
            var = {}
            for comp, vecs in (("seq", seq), ("emb", embv)):
                for sh in (0.1, 0.3, 0.6):
                    if comp == "emb" and sh != 0.3:
                        continue
                    var[(comp, sh)] = standardise(v2.pool_normalised(vecs, cands, questions, sh))
            players = sorted(cands)
            runs.append({"Z": {k: standardise(x) for k, x in S.items()}, "var": var, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]),
                         "owner": owner, "info": {**info, "fold": v_, "seed": s}})
            alone = {f"{c}_w{sh}": float(score(topk(x, 5), runs[-1]["truth"]).mean()) for (c, sh), x in var.items()}
            print(f"pool fold {v_} seed {s}: alone seq {score(topk(runs[-1]['Z']['seq'], 5), runs[-1]['truth']).mean():.3f} emb {score(topk(runs[-1]['Z']['emb'], 5), runs[-1]['truth']).mean():.3f} "
                  + " ".join(f"{k} {x:.3f}" for k, x in alone.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, spec: [r["var"][(c, spec[c])] if c in spec else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_pool, per_q = {"pools": [r["info"] for r in runs], "arms": {}, "comparisons": {}}, {}, {}
    for arm, spec in ARMS.items():
        sc, ws = [], []
        for r in runs:
            others = [o for o in runs if o["info"]["fold"] != r["info"]["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, spec)), 5), r["truth"]))
            ws.append(fus.w.tolist())
        per_pool[arm], per_q[arm] = [float(x.mean()) for x in sc], np.concatenate(sc)
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "weights_mean": dict(zip(COMPS, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  weights {out['arms'][arm]['weights_mean']}", flush=True)
    passing = []
    for arm in ARMS:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=a.n_boot)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 6)
        out["comparisons"][f"{arm} vs ref"] = d
        if d["pass"]:
            passing.append(arm)
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/8 -> {'PASS' if d['pass'] else 'no'}", flush=True)
    best = max(passing, key=lambda k: out["arms"][k]["score"]) if passing else "ref"
    out["chosen"] = best
    print(f"chosen: {best}", flush=True)
    if a.fusion_out and best != "ref":
        spec = ARMS[best]
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, spec) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        doc = {**base_doc, "weights": fus.w.tolist(), "pool_normalise": spec, "arm": best, "simulated_score": out["arms"][best]["score"]}
        Path(a.fusion_out).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: {dict(zip(COMPS, np.round(fus.w, 3).tolist()))} pool_normalise {spec}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
