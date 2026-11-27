#!/usr/bin/env python
"""Eighth simulation of the v2 added pools (METHOD_NOTES 2.33), CPU only, 8 single-fold pools as pid_v2_sim2.py:
  emb_ft<t>   the embedding fine-tuned t = 50 / 100 / 300 steps on the pool's own candidates (aicup_pid.embed.adapted, lr 1e-4)
  seq_cm<l>   colour-matched sequence prototypes: a question game is compared with the candidate's games of the same colour, the
              other colour's games weighted l (0 or 0.5); the question's score is the sum of its games' cosines
Each arm replaces one component of ref = base+seq. Rule (fixed before the run): CI above 0 and at least 6 of the 8 pools gain."""
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
from aicup_pid.embed import PlayerEmbedding, adapted  # noqa: E402
from aicup_pid.fusion import Fusion, standardise, topk  # noqa: E402
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_v2_sim2 import BASE, SIZES, build, score  # noqa: E402

COMPS = (*BASE, "seq")
ARMS = {"ref": {}, "emb_ft50": {"emb": "emb_ft50"}, "emb_ft100": {"emb": "emb_ft100"}, "emb_ft300": {"emb": "emb_ft300"}, "seq_cm0": {"seq": "seq_cm0"}, "seq_cm0.5": {"seq": "seq_cm0.5"}}


def colour_matched(seq, cands, questions, lam):
    players = sorted(cands)
    unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)  # noqa: E731
    protos = {}
    for c in ("B", "W"):
        P = []
        for p in players:
            same = [seq[k] for k, col in cands[p] if col == c]
            other = [seq[k] for k, col in cands[p] if col != c]
            v = (np.sum(same, axis=0) if same else 0) + lam * (np.sum(other, axis=0) if other else 0)
            if not same and lam == 0:
                v = np.sum(other, axis=0)  # no game of that colour: all of the candidate's games
            P.append(v)
        protos[c] = unit(np.stack(P).astype(np.float64))
    return np.stack([np.sum([protos[col] @ seq[k].astype(np.float64) for k, col in q], axis=0) for q in questions])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(a.workers)
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    emb_of = {k: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / f"emb2_p{k}.pt", map_location="cpu", weights_only=False), device="cpu") for k in ("12", "34")}
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
            emb = emb_of["12" if v_ in (1, 2) else "34"]
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, emb, which=COMPS, seq=seq)
            by = {p: np.stack([rows[k] for k, _ in cands[p]]) for p in cands}
            var = {f"emb_ft{t}": standardise(v2.components(cands, questions, moves, sgfs, cfg, rows, adapted(emb, by, steps=t, seed=s), which=("emb",))["emb"]) for t in (50, 100, 300)}
            for lam in (0, 0.5):
                var[f"seq_cm{lam:g}"] = standardise(colour_matched(seq, cands, questions, lam))
            players = sorted(cands)
            r = {"Z": {k: standardise(x) for k, x in S.items()}, "var": var, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner, "fold": v_}
            runs.append(r)
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: alone emb {al(r['Z']['emb']):.3f} " + " ".join(f"{k} {al(x):.3f}" for k, x in var.items()) + f" | seq {al(r['Z']['seq']):.3f} ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, spec: [r["var"][spec[c]] if c in spec else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}}, {}, {}
    for arm, spec in ARMS.items():
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, spec)), 5), r["truth"]))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES}}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}", flush=True)
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
