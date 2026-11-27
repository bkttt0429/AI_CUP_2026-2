#!/usr/bin/env python
"""Re-release preparation: the models on simulated pools whose games are randomly rotated / mirrored (METHOD_NOTES 2.22).

Organisers may re-release the official files with every game under a random board symmetry. Then the orientation
component, the orientation block of the reference model and the raw opening carry no signal (the habit "opens in the
upper-right corner" is gone), while opening (symmetry-canonical), style, local shapes and, approximately, the trunk
rows survive. This script measures that on the component-CV pools (pid_ref_cv.run_pool layout: make_pool seed 0, fold
pairs 1,2 / 3,4, references from the other pair), each in two conditions:
  plain  the training games as they are (trunk rows from --store-plain)
  sym    every candidate and question game under aicup_pid.symmetry.sym_of(key, --sym-seed) (trunk rows from
         --store-sym, extracted with pid_trunk_extract.py --sym-seed); the reference players' games stay as released,
         as the training data would
Arms: the current main document with its fitted weights as they are (f3oreft_fixed), the same components cross-fitted
on the other pool of the same condition (f3oreft_cf), and the symmetry-invariant set opening + style + local + ref
without its orientation block (ref0) + trunk, cross-fitted (sym_cf); plus every component alone.
Pre-specified: in the sym condition sym_cf vs f3oreft_fixed (the robust model beats the model we would otherwise upload).
Report only: the same in the plain condition (the price of dropping orientation when nothing was rotated), sym_cf vs
f3oreft_cf in both. --fusion-out writes the sym document (components named as predict_pid.py knows them: ref with
cfg ref_orient_raw 0) fitted on both sym pools. Fold 0 is refused.

  python -I scripts/pid_sym_cv.py --store-plain data/trunk/train --store-sym data/trunk/train_sym0 --sym-seed 0 \\
      --main results/pid_fusion3o_ref_trunk.json --workers 12 --out results/sym/sym_cv.json --fusion-out results/pid_fusion_sym.json \\
      --fusion-main-out results/pid_fusion3o_ref_trunk_symfit.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import features_all, fit_from_parsed, fit_local, fit_orient, fit_ref, parse_all, score_components, score_ref  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.ref import reference_players  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from aicup_pid.symmetry import transform_games  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
MAIN = ("opening", "style", "local", "orient", "ref", "trunk")
SYM = ("opening", "style", "local", "ref0", "trunk")
CONDITIONS = ("plain", "sym")
PRE = [("sym", "sym_cf", "f3oreft_fixed")]
REPORT = [("plain", "sym_cf", "f3oreft_fixed"), ("sym", "sym_cf", "f3oreft_cf"), ("plain", "sym_cf", "f3oreft_cf"), ("sym", "f3oreft_cf", "f3oreft_fixed")]


def run_pool(games, folds_df, folds, ref_folds, cfg, stores, sym_seed, workers, log=print):
    """Both conditions of one pool: {condition: {S: {component: [Q, C]}, players, truth}}, plus questions."""
    t0 = time.time()
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=pools.N_CANDIDATES, cand_games=pools.CAND_GAMES, seed=0,
                                      pool_id="p" + "".join(map(str, folds)), order="shuffle")
    refs = reference_players(games, folds_df, ref_folds, min_black=cfg["ref_min_black"], cap=cfg["ref_cap"], seed=0, exclude=set(cands["player_id"]))
    pool_keys = set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks}
    ref_keys = {k for ks in refs.values() for k in ks}
    sgfs = {k: v[0] for k, v in load_games(sorted(pool_keys | ref_keys), aicup_pid.TRAIN_DIR).items()}
    ref_moves = parse_all({k: sgfs[k] for k in ref_keys}, workers)
    ref_models = {raw: fit_ref(refs, ref_moves, {**cfg, "ref_orient_raw": raw}) for raw in (cfg["ref_orient_raw"], 0)}
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    q_keys = [list(k) for k in qs["game_keys"]]
    log(f"pool {folds}: {len(by_player)} players, {len(qs)} questions, {len(refs)} references; loaded in {time.time() - t0:.0f} s", flush=True)
    out = {}
    for cond in CONDITIONS:
        t1 = time.time()
        pg = {k: sgfs[k] for k in pool_keys}
        if cond == "sym":
            pg = transform_games(pg, pool_keys, sym_seed)
        moves, feats = parse_all(pg, workers), features_all(pg, workers)
        opening, style = fit_from_parsed(by_player, moves, feats, cfg)
        S = score_components(("opening", "style", "local", "orient"), opening, style, q_keys, moves, feats,
                             local=fit_local(by_player, moves, cfg), orient=fit_orient(by_player, moves, cfg))
        S["ref"] = score_ref(ref_models[cfg["ref_orient_raw"]], by_player, q_keys, moves)
        S["ref0"] = score_ref(ref_models[0], by_player, q_keys, moves)
        tm = fit_trunk(by_player, stores[cond], cfg)
        if tm.players != opening.players:
            raise ValueError("candidate order differs")
        S["trunk"] = score_trunk(tm, q_keys, stores[cond])
        truth = np.array([opening.index[p] for p in qs["player_id"]])
        out[cond] = {"S": S, "players": opening.players, "truth": truth}
        log(f"pool {folds} {cond}: components in {time.time() - t1:.0f} s", flush=True)
    return {"info": info, "questions": qs, **out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store-plain", nargs="+", required=True)
    ap.add_argument("--store-sym", nargs="+", required=True)
    ap.add_argument("--sym-seed", type=int, default=0)
    ap.add_argument("--main", default=str(ROOT / "results" / "pid_fusion3o_ref_trunk.json"))
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    ap.add_argument("--fusion-main-out", default=None, help="also write the main document's components with weights refitted on both sym pools")
    a = ap.parse_args(argv)
    t0 = time.time()
    main_doc = json.loads(Path(a.main).read_text())
    if tuple(main_doc["components"]) != MAIN:
        raise ValueError(f"{a.main}: components {main_doc['components']} are not {MAIN}")
    cfg = dict(main_doc["cfg"])
    specs = parse_pools(a.pools)
    if len(specs) != 2 or any(0 in f for f in specs):
        print("give exactly two fold pairs without fold 0", file=sys.stderr)
        return 2
    stores = {"plain": load_store(*a.store_plain), "sym": load_store(*a.store_sym)}
    print("trunk stores: " + ", ".join(f"{c} {len(s)} games" for c, s in stores.items()), flush=True)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    runs = [run_pool(games, folds_df, f, specs[1 - i], cfg, stores, a.sym_seed, a.workers) for i, f in enumerate(specs)]

    fixed = Fusion()
    fixed.w = np.array(main_doc["weights"])
    arms_c = {"f3oreft_cf": MAIN, "sym_cf": SYM}
    per_q = {c: {} for c in CONDITIONS}
    results = []
    for cond in CONDITIONS:
        for i, r in enumerate(runs):
            rc, others = r[cond], [o[cond] for j, o in enumerate(runs) if j != i]
            fused = {"f3oreft_fixed": fixed.scores([rc["S"][c] for c in MAIN])}
            weights = {}
            for name, comps in arms_c.items():
                fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
                weights[name], fused[name] = fus.w.tolist(), fus.scores([rc["S"][c] for c in comps])
            scores = {}
            for name, S in [*((c, rc["S"][c]) for c in (*MAIN, "ref0")), *fused.items()]:
                scores[name], pq = evaluate(S, rc["players"], r["questions"], rc["truth"])
                per_q[cond].setdefault(name, []).append(pq)
            results.append({"condition": cond, "pool": r["info"], "scores": scores, "weights": weights})
            print(f"{cond} pool {r['info']['folds']}: " + "  ".join(f"{k} {v['score']:.4f}" for k, v in scores.items())
                  + f"  sym weights {np.round(weights['sym_cf'], 3).tolist()}", flush=True)
    pid = np.concatenate([r["questions"]["player_id"].to_numpy(dtype=str) for r in runs])
    n_games = np.concatenate([r["questions"]["n_games"].to_numpy() for r in runs])
    comps = {}
    for group, items in (("pre_specified", PRE), ("report_only", REPORT)):
        comps[group] = {}
        for cond, x, z in items:
            qx, qz = np.concatenate(per_q[cond][x]), np.concatenate(per_q[cond][z])
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q[cond][x], per_q[cond][z])]
            n5 = n_games == 5
            d["diff_n5"] = float(np.mean((qx - qz)[n5])) if n5.any() else None
            d.update({"per_pool_diff": per_pool, "by_n": {str(n): float(np.mean((qx - qz)[n_games == n])) for n in sorted(set(n_games.tolist()))},
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})
            comps[group][f"[{cond}] {x} vs {z}"] = d
            print(f"[{group}] [{cond}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
                  + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {c: {k: float(np.mean(np.concatenate(v))) for k, v in per_q[c].items()} for c in CONDITIONS},
           "comparisons": comps, "cfg": cfg, "sym_seed": a.sym_seed, "main": a.main, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    if a.fusion_out:
        fus = Fusion().fit([np.vstack([r["sym"]["S"][c] for r in runs]) for c in SYM], np.concatenate([r["sym"]["truth"] for r in runs]))
        doc = {"components": ["opening", "style", "local", "ref", "trunk"], "weights": fus.w.tolist(), "cfg": {**cfg, "ref_orient_raw": 0},
               "fitted_on_pools": [r["info"]["folds"] for r in runs], "fitted_on": f"sym condition (sym_seed {a.sym_seed}): every pool game randomly rotated / mirrored",
               "questions": int(len(pid)), "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", **pid_environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    if a.fusion_main_out:
        fus = Fusion().fit([np.vstack([r["sym"]["S"][c] for r in runs]) for c in MAIN], np.concatenate([r["sym"]["truth"] for r in runs]))
        doc = {**{k: v for k, v in main_doc.items() if k not in ("weights", "fitted_on_pools", "questions")}, "components": list(MAIN), "weights": fus.w.tolist(), "cfg": cfg,
               "fitted_on_pools": [r["info"]["folds"] for r in runs], "fitted_on": f"sym condition (sym_seed {a.sym_seed}): every pool game randomly rotated / mirrored",
               "questions": int(len(pid)), **pid_environment("cpu")}
        Path(a.fusion_main_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_main_out).write_text(to_json(doc, indent=1) + "\n")
        print(f"wrote {a.fusion_main_out}: weights {np.round(fus.w, 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
