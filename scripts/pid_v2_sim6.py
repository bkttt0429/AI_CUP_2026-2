#!/usr/bin/env python
"""Sixth simulation of the v2 added pools (METHOD_NOTES 2.32): more members and pool adaptation for the two learnt components,
on the 8 single-fold pools of pid_v2_sim2.py (fold 1-4 x seeds 0, 1), weights fitted on the other folds' pools.
Arms against ref = base+seq (fixed before the run; rule: player-bootstrap CI above 0 and at least 6 of the 8 pools gain):
  seq2     sequence component = mean of two states of net v (best held-out step and last step), row-standardised
  emb5     embedding component = mean of five seeds of emb2
  seq_ft   sequence vectors after the 100-step fine-tune on the pool's candidates (pid_v2_seq_finetune.py, horizon 1000); the
           first four pools chose 100 among four lengths (+0.0058 [-0.0010, +0.0125]), the four seed-1 pools are new
  seq2+emb5, and seq_ft+emb5 (report)
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
ARMS = {"ref": {}, "seq2": {"seq": "seq2"}, "emb5": {"emb": "emb5"}, "seq_ft": {"seq": "seq_ft"}, "seq2+emb5": {"seq": "seq2", "emb": "emb5"}, "seq_ft+emb5": {"seq": "seq_ft", "emb": "emb5"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cfg-from", default=str(ROOT / "results" / "pid_fusion_v2b.json"))
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda p: PlayerEmbedding.from_state(torch.load(p, map_location="cpu", weights_only=False))  # noqa: E731
    embs = {k: [load(ROOT / "results" / "embed" / (f"emb2_p{k}.pt" if s == 0 else f"emb2_p{k}_s{s}.pt")) for s in range(5)] for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731
    runs = []
    for v_ in (1, 2, 3, 4):
        seq, seq_last = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz"), vecs(ROOT / "data" / "seq" / f"fold{v_}_last.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq.keys())]
        for s in (0, 1):
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            el = embs["12" if v_ in (1, 2) else "34"]
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, el[0], which=COMPS, seq=seq)
            one = lambda **kw: v2.components(cands, questions, moves, sgfs, cfg, rows, **kw)  # noqa: E731
            var = {"seq2": standardise(one(which=("seq",), seq=[seq, seq_last])["seq"]),
                   "emb5": standardise(np.mean([standardise(one(which=("emb",), emb=m)["emb"]) for m in el], axis=0)),
                   "seq_ft": standardise(one(which=("seq",), seq=vecs(Path(a.ft_dir) / f"sim_f{v_}_s{s}_t100.npz"))["seq"])}
            players = sorted(cands)
            r = {"Z": {k: standardise(x) for k, x in S.items()}, "var": var, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]), "owner": owner,
                 "fold": v_, "seed": s}
            runs.append(r)
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: alone seq {al(r['Z']['seq']):.3f} seq2 {al(var['seq2']):.3f} seq_ft {al(var['seq_ft']):.3f} | emb {al(r['Z']['emb']):.3f} emb5 {al(var['emb5']):.3f} ({time.time() - t0:.0f} s)", flush=True)
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])
    mats = lambda r, spec: [r["var"][spec[c]] if c in spec else r["Z"][c] for c in COMPS]  # noqa: E731
    out, per_q, per_pool = {"arms": {}, "comparisons": {}}, {}, {}
    for arm, spec in ARMS.items():
        sc, ws = [], []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, spec)), 5), r["truth"]))
            ws.append(fus.w.tolist())
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES},
                            "weights_mean": dict(zip(COMPS, np.round(np.mean(ws, axis=0), 3).tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k: round(x, 3) for k, x in out['arms'][arm]['by_n'].items()})}  seq weight {out['arms'][arm]['weights_mean']['seq']}, emb {out['arms'][arm]['weights_mean']['emb']}", flush=True)
    for arm in ARMS:
        if arm == "ref":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["ref"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["ref"])))
        d["new_pools_diff"] = float(np.mean([x - y for x, y, r in zip(per_pool[arm], per_pool["ref"], runs) if r["seed"] == 1]))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 6)
        out["comparisons"][f"{arm} vs ref"] = d
        print(f"{arm} vs ref: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/8, seed-1 pools {d['new_pools_diff']:+.4f} -> {'PASS' if d['pass'] else 'no'}", flush=True)
    # second look (written 11-12 after the first output, which showed seq_ft passing and emb5 alone not): does emb5 add on top of seq_ft? same rule
    d = paired_player_bootstrap(per_q["seq_ft+emb5"], per_q["seq_ft"], pid, n=2000)
    d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool["seq_ft+emb5"], per_pool["seq_ft"])))
    d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 6)
    out["comparisons"]["seq_ft+emb5 vs seq_ft"] = d
    print(f"seq_ft+emb5 vs seq_ft: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/8 -> {'PASS' if d['pass'] else 'no'}", flush=True)
    for arm, spec in ARMS.items():
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, spec) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm]["weights_all_pools"] = dict(zip(COMPS, fus.w.tolist()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
