#!/usr/bin/env python
"""Multi-network simulation of the v2 added pools (METHOD_NOTES 2.36): the configuration decisions taken again NEXT TO SEVERAL
sequence networks, as the official files have (five and more fine-tuned networks averaged), not next to one.

Every fold v has several networks that never saw its players: the rank session's M88 net v and its M90 nets v of seeds 0-2
(all 1M games). Pools: fold 1-4 x seeds 0, 1 as pid_v2_sim2.py. Sequence component = mean of the row-standardised matrices
of the chosen networks, each after t steps of the pool fine-tune (pid_v2_seq_finetune.py, lr 1e-4, cosine horizon 1000).
Arms (fixed before the run), weights fitted on the other folds' pools:
  base     all networks, 100 steps, emb2            <- the multi-network analogue of the live file (v2d)
  one      M88 alone, 100 steps, emb2               (the old single-network setting)
  m90      the M90 networks only, 100 steps, emb2
  ft0 / ft50   all networks, no fine-tune / 50 steps, emb2
  e3 / e5      all networks, 100 steps, three emb3 seeds / five emb2 seeds
  ft50e3       all networks, 50 steps, three emb3 seeds
Rule: an arm replaces base when its gain has a player-bootstrap CI above 0 and at least 6 of the 8 pools gain; among passing
arms the best by score. The chosen arm's weights (all pools) go to the next official file.
  --nets name=<fold file tag>:<fine-tune tag> ...   e.g. m88=:  m90s0=_m90s0:_m90s0
     fold vectors data/seq/fold<v><tag>.npz; fine-tuned <ft-dir>/sim_f<v>_s<seed><fttag>_t<step>.npz
     (M88's 50-step vectors are the earlier files sim_f<v>_s<seed>_g_t50.npz)
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
    ap.add_argument("--phase-dir", default="/work/u2507115/pid_v2/phase_rows")
    ap.add_argument("--nets", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    nets = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split(":")) for x in a.nets}
    m90 = [k for k in nets if k != "m88"]
    cfg = json.loads(Path(a.cfg_from).read_text())["cfg"]
    load = lambda name: PlayerEmbedding.from_state(torch.load(ROOT / "results" / "embed" / name, map_location="cpu", weights_only=False))  # noqa: E731
    e2 = {k: [load(f"emb2_p{k}.pt" if s == 0 else f"emb2_p{k}_s{s}.pt") for s in range(5)] for k in ("12", "34")}
    e3 = {k: [load(f"emb3_p{k}.pt" if s == 0 else f"emb3_p{k}_s{s}.pt") for s in range(3)] for k in ("12", "34")}
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    vecs = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731

    def ft_file(v_, s, name, step):
        if step == 0:
            return ROOT / "data" / "seq" / f"fold{v_}{nets[name][0]}.npz"
        if name == "m88":
            return Path(a.ft_dir) / (f"sim_f{v_}_s{s}_g_t50.npz" if step == 50 else f"sim_f{v_}_s{s}_t100.npz")
        return Path(a.ft_dir) / f"sim_f{v_}_s{s}{nets[name][1]}_t{step}.npz"

    runs = []
    for v_ in a.folds:
        seq0 = vecs(ROOT / "data" / "seq" / f"fold{v_}.npz")
        g = games[(games.fold == v_) & games.game_key.isin(rows.keys()) & games.game_key.isin(seq0.keys())]
        for s in a.seeds:
            cands, questions, owner, block, sgfs, info = build(g, s)
            moves = parse_all(sgfs, a.workers)
            k = "12" if v_ in (1, 2) else "34"
            S = v2.components(cands, questions, moves, sgfs, cfg, rows, e2[k][0], which=BASE)
            keys = {x for games_ in (*cands.values(), *questions) for x, _ in games_}
            rows3 = v2.load_phase_rows(AICUP1, a.phase_dir, keep=keys)
            one = lambda **kw: v2.components(cands, questions, moves, sgfs, cfg, **kw)  # noqa: E731
            seqm = {(name, t): standardise(one(which=("seq",), seq=vecs(ft_file(v_, s, name, t)))["seq"]) for name in nets for t in (0, 50, 100)}
            emb = {"emb2": standardise(S["emb"]), "emb2x5": standardise(np.mean([standardise(one(rows=rows, emb=m, which=("emb",))["emb"]) for m in e2[k]], axis=0)),
                   "emb3x3": standardise(np.mean([standardise(one(rows=rows3, emb=m, which=("emb",))["emb"]) for m in e3[k]], axis=0))}
            players = sorted(cands)
            r = {"Z": {c: standardise(x) for c, x in S.items()}, "seq": seqm, "emb": emb, "truth": np.array([players.index(p) for p in owner]), "n": np.array([len(q) for q in questions]),
                 "owner": owner, "fold": v_}
            runs.append(r)
            al = lambda M: float(score(topk(M, 5), r["truth"]).mean())  # noqa: E731
            print(f"pool fold {v_} seed {s}: seq alone (0 / 50 / 100 steps) " + "  ".join(f"{name} {al(seqm[(name, 0)]):.3f}/{al(seqm[(name, 50)]):.3f}/{al(seqm[(name, 100)]):.3f}" for name in nets)
                  + f"  all nets {al(np.mean([seqm[(n_, 0)] for n_ in nets], axis=0)):.3f}/{al(np.mean([seqm[(n_, 50)] for n_ in nets], axis=0)):.3f}/{al(np.mean([seqm[(n_, 100)] for n_ in nets], axis=0)):.3f}"
                  + f" ({time.time() - t0:.0f} s)", flush=True)
    ALL = list(nets)
    arms = {"base": (ALL, 100, "emb2"), "one": (["m88"], 100, "emb2"), "ft0": (ALL, 0, "emb2"), "ft50": (ALL, 50, "emb2"), "e3": (ALL, 100, "emb3x3"), "e5": (ALL, 100, "emb2x5"),
            "ft50e3": (ALL, 50, "emb3x3")}
    if m90:
        arms["m90"] = (m90, 100, "emb2")
    nn = np.concatenate([r["n"] for r in runs])
    pid = np.concatenate([np.char.add(r["owner"].astype(str), f"|{i}") for i, r in enumerate(runs)])

    def mats(r, spec):
        names, t, e = spec
        seq = standardise(np.mean([r["seq"][(n_, t)] for n_ in names], axis=0))
        return [seq if c == "seq" else r["emb"][e] if c == "emb" else r["Z"][c] for c in COMPS]

    out, per_q, per_pool = {"arms": {}, "comparisons": {}, "nets": {k: list(x) for k, x in nets.items()}, "folds": a.folds}, {}, {}
    cross = len(a.folds) > 1
    for arm, spec in arms.items():
        sc = []
        for r in runs:
            others = [o for o in runs if o["fold"] != r["fold"]] if cross else [o for o in runs if o is not r]
            fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(o, spec) for o in others])], np.concatenate([o["truth"] for o in others]))
            sc.append(score(topk(fus.scores(mats(r, spec)), 5), r["truth"]))
        per_q[arm], per_pool[arm] = np.concatenate(sc), [float(x.mean()) for x in sc]
        fus = Fusion().fit([np.vstack(col) for col in zip(*[mats(r, spec) for r in runs])], np.concatenate([r["truth"] for r in runs]))
        out["arms"][arm] = {"score": float(per_q[arm].mean()), "per_pool": per_pool[arm], "by_n": {str(m): float(per_q[arm][nn == m].mean()) for m in SIZES}, "spec": [spec[0], spec[1], spec[2]],
                            "weights_all_pools": dict(zip(COMPS, fus.w.tolist()))}
        print(f"arm {arm}: {out['arms'][arm]['score']:.4f}  by n {({k_: round(x, 3) for k_, x in out['arms'][arm]['by_n'].items()})}  seq weight {fus.w[-1]:.2f}, emb {fus.w[COMPS.index('emb')]:.2f}", flush=True)
    passing = []
    for arm in arms:
        if arm == "base":
            continue
        d = paired_player_bootstrap(per_q[arm], per_q["base"], pid, n=2000)
        d["pools_positive"] = int(sum(x > y for x, y in zip(per_pool[arm], per_pool["base"])))
        d["pass"] = bool(d["ci95"][0] > 0 and d["pools_positive"] >= 0.75 * len(runs))
        out["comparisons"][f"{arm} vs base"] = d
        if d["pass"] and arm not in ("one", "m90"):
            passing.append(arm)
        print(f"{arm} vs base: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}], pools up {d['pools_positive']}/{len(runs)} -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out["chosen"] = max(passing, key=lambda k_: out["arms"][k_]["score"]) if passing else "base"
    print(f"chosen: {out['chosen']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
