#!/usr/bin/env python
"""Final-stage CV: several fusion documents on ONE-QUESTION-PER-PLAYER pools (the official structure), each with and
without the one-to-one assignment, compared pairwise against a baseline document's assigned arm.

The component CV scripts (pid_local_cv / pid_hd_cv / pid_orient_cv) rank a component's gain on make_pool pools without
assignment; the submission applies the assignment on one-question-per-player pools, where the remaining errors are
few and different (METHOD_NOTES 2.16). So the comparison that matters for the upload is <doc>+assign vs <baseline>+assign
here. All components named by any document are fitted once per pool (one orientation model per distinct orient_raw);
the documents' weights are used as given (they were fitted on the make_pool pools of the same fold sets, a shared
optimism). Pre-specified: every non-baseline document's assigned arm vs the baseline's assigned arm (player bootstrap,
both fold sets the same sign, overall >= 0.003 or n=5 >= 0.005); report only: the plain arms. Fold 0 is refused.

  python -I scripts/pid_final_cv.py --fusions fusion3o=results/pid_fusion3o.json raw3=results/pid_fusion3o_raw3.json \
      --baseline fusion3o --pools 1,2 3,4 --seeds 0 1 2 3 4 --workers 12 --out results/final/final_cv.json
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
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.assign import assign_topk  # noqa: E402
from aicup_pid.components import features_all, fit_from_parsed, fit_hd, fit_local, fit_opening_raw, fit_orient, fit_ref, parse_all, score_components, score_ref  # noqa: E402
from aicup_pid.ref import reference_players  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from pid_cv import parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005


def question_scores(idx, truth):
    """Official per-question scores of Top-5 lists."""
    out = np.zeros(len(truth))
    for q in range(len(truth)):
        hit = np.where(idx[q] == truth[q])[0]
        out[q] = np.exp(-hit[0]) if len(hit) else 0.0
    return out


def run_pool(games, folds_df, folds, seed, docs, a, log=print, ref_folds=None):
    t0 = time.time()
    cands, qs, info = pools.make_pool_one_per_player(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, per_size=a.per_size, seed=seed, pool_id=f"p{''.join(map(str, folds))}s{seed}", order=a.order)
    keys = set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks}
    cfg = next(iter(docs.values()))["cfg"]
    need_ref = any("ref" in d["components"] for d in docs.values())
    refs = {}
    if need_ref:
        rcfg = next(d["cfg"] for d in docs.values() if "ref" in d["components"])
        refs = reference_players(games, folds_df, ref_folds, min_black=rcfg.get("ref_min_black", 30), cap=rcfg.get("ref_cap", 40), seed=rcfg.get("ref_seed", 0), exclude=set(cands["player_id"]))
        keys |= {k for ks in refs.values() for k in ks}
    pool_keys = set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks}
    sgfs = {k: v[0] for k, v in load_games(sorted(keys), aicup_pid.TRAIN_DIR).items()}
    moves, feats = parse_all(sgfs, a.workers), features_all({k: sgfs[k] for k in pool_keys}, a.workers)
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    q_keys = [list(k) for k in qs["game_keys"]]
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    needed = {c for d in docs.values() for c in d["components"]}
    S = score_components(("opening", "style"), opening, style, q_keys, moves, feats)
    if "local" in needed:
        S["local"] = score_components(("local",), opening, style, q_keys, moves, feats, local=fit_local(by_player, moves, cfg))["local"]
    if "hd" in needed:
        S["hd"] = score_components(("hd",), opening, style, q_keys, moves, feats, hd=fit_hd(by_player, sgfs, cfg), sgfs=sgfs)["hd"]
    if "opening_raw" in needed:
        S["opening_raw"] = score_components(("opening_raw",), opening, style, q_keys, moves, feats, opening_raw=fit_opening_raw(by_player, moves, cfg))["opening_raw"]
    if "ref" in needed:
        rcfg = next(d["cfg"] for d in docs.values() if "ref" in d["components"])
        S["ref"] = score_ref(fit_ref(refs, moves, rcfg), by_player, q_keys, moves)
    if "trunk" in needed:
        tcfg = next(d["cfg"] for d in docs.values() if "trunk" in d["components"])
        S["trunk"] = score_trunk(fit_trunk(by_player, a.trunk_rows, tcfg), q_keys, a.trunk_rows)
    orient_S = {}
    for d in docs.values():
        if "orient" in d["components"]:
            n_raw = d["cfg"].get("orient_raw", 1)
            if n_raw not in orient_S:
                orient_S[n_raw] = score_components(("orient",), opening, style, q_keys, moves, feats, orient=fit_orient(by_player, moves, {**cfg, "orient_raw": n_raw}))["orient"]
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    n = qs["n_games"].to_numpy()
    per_q, summary = {}, {}
    for name, d in docs.items():
        fus = Fusion()
        fus.w = np.array(d["weights"])
        parts = [orient_S[d["cfg"].get("orient_raw", 1)] if c == "orient" else S[c] for c in d["components"]]
        S_f = fus.scores(parts)
        for arm, idx in ((name, topk(S_f, 5)), (name + "+assign", assign_topk(S_f, 5))):
            per_q[arm] = question_scores(idx, truth)
            summary[arm] = {"score": float(per_q[arm].mean()), "top1": float((idx[:, 0] == truth).mean()), "wrong": int((idx[:, 0] != truth).sum())}
    log(f"pool {folds} seed {seed}: {len(truth)} questions, {time.time() - t0:.0f} s" + (f", {len(refs)} references" if refs else "") + "; " + "  ".join(f"{k} {v['score']:.4f}" for k, v in summary.items()), flush=True)
    return {"info": info, "per_q": per_q, "pid": qs["player_id"].to_numpy(dtype=str), "n": n, "summary": summary, "folds": folds, "seed": seed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fusions", nargs="+", required=True, help="name=path of fusion documents")
    ap.add_argument("--baseline", required=True, help="name of the baseline document")
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--per-size", type=int, default=100)
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--trunk-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "train")], help="trunk rows of the training games (scripts/pid_trunk_extract.py --keys)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        docs = {}
        for spec in a.fusions:
            name, path = spec.split("=", 1)
            docs[name] = json.load(open(path))
        if a.baseline not in docs:
            raise ValueError(f"baseline {a.baseline} is not among the documents")
        specs = parse_pools(a.pools)
        if any(0 in f for f in specs):
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        a.trunk_rows = load_store(*a.trunk_store) if any("trunk" in d["components"] for d in docs.values()) else {}
        runs = [run_pool(games, folds_df, f, s, docs, a, ref_folds=[x for g in specs if g != f for x in g]) for f in specs for s in a.seeds]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    arms = list(runs[0]["per_q"])
    cat = {arm: np.concatenate([r["per_q"][arm] for r in runs]) for arm in arms}
    pid = np.concatenate([r["pid"] for r in runs])
    n_games = np.concatenate([r["n"] for r in runs])
    fold_sets = [tuple(r["folds"]) for r in runs]
    comps = {"pre_specified": {}, "report_only": {}}
    base_a = a.baseline + "+assign"
    for name in docs:
        if name == a.baseline:
            continue
        for group, x, z in (("pre_specified", name + "+assign", base_a), ("report_only", name, a.baseline)):
            d = paired_player_bootstrap(cat[x], cat[z], pid, n=a.n_boot)
            per_fs = {}
            for fs in sorted(set(fold_sets)):
                m = np.concatenate([np.full(len(r["per_q"][x]), tuple(r["folds"]) == fs) for r in runs])
                per_fs[str(list(fs))] = float(np.mean(cat[x][m] - cat[z][m]))
            n5 = n_games == 5
            d["diff_n5"] = float(np.mean((cat[x] - cat[z])[n5])) if n5.any() else None
            d.update({"per_fold_set_diff": per_fs,                       "by_n": {str(k): float(np.mean((cat[x] - cat[z])[n_games == k])) for k in sorted(set(n_games.tolist()))},
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_fs.values()) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})
            comps[group][f"{x} vs {z}"] = d
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  fold sets " + " ".join(f"{v:+.4f}" for v in per_fs.values())
                  + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    overall = {arm: {"score": float(v.mean()), "wrong_top1": int(sum(r["summary"][arm]["wrong"] for r in runs))} for arm, v in cat.items()}
    print("overall: " + "  ".join(f"{k} {v['score']:.4f} ({v['wrong_top1']} wrong)" for k, v in overall.items()))
    out = {"pools": [{"pool": r["info"], "seed": r["seed"], "summary": r["summary"]} for r in runs], "overall": overall, "comparisons": comps, "baseline": a.baseline,
           "documents": {k: {"components": v["components"], "weights": v["weights"], "cfg": v["cfg"]} for k, v in docs.items()}, "seeds": a.seeds, "order": a.order,
           "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
