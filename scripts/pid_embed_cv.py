#!/usr/bin/env python
"""Line B8: the learnt player embedding (aicup_pid.embed) on top of fusion3o_ref_trunk, pre-specified (METHOD_NOTES 2.23).

Pools: pid_ref_cv.run_pool (make_pool seed 0, fold pairs 1,2 / 3,4, references from the other pair), trunk rows from
--store. For each pool the embedding is trained on the OTHER fold pair's players (every Black game with a row in --store,
>= min_games rows), so no pool player is ever seen; two variants:
  emb   training players only
  embx  training players + the 800 players of the void 10-30 official candidate pools (--old-official folder and its
        trunk rows --old-store; labels = the organisers' player ids; the probed Public answers are NOT used)
Rows under a random board symmetry (--aug-store, pid_trunk_extract --sym-seed) serve as augmentation where present.
Arms (weights cross-fitted on the other pool): f3oreft (current main), f3oreft_emb, f3oreft_embx, f3oref_embx (embx in
place of trunk). Pre-specified: f3oreft_embx vs f3oreft (player bootstrap, both pools the same sign, >= 0.003 overall or
>= 0.005 at n=5). Report only: f3oreft_emb vs f3oreft, embx vs trunk, emb vs trunk, f3oref_embx vs f3oreft.
--model-dir keeps each pool's embedding (emb[x]_p<folds>.pt) for pid_final_cv.py; --final-model-out trains embx on folds
1-4 + the old official pools for the new official files; --fusion-out writes f3oreft_embx's weights fitted on both pools.
Fold 0 is refused.

  python -I scripts/pid_embed_cv.py --store data/trunk/train data/trunk/train_rest --aug-store data/trunk/train_sym0 \\
      --old-official data/tests --old-store data/trunk/official --model-dir results/embed --out results/embed/embed_cv.json \\
      --fusion-out results/pid_fusion3o_ref_trunk_emb.json --final-model-out results/embed/embx_final.pt
"""
import argparse
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
from aicup_pid import pools  # noqa: E402
from aicup_pid.embed import PlayerEmbedding, score_embed  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402
import pid_ref_cv  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005
BASE = pid_ref_cv.FIVE  # opening, style, local, orient, ref
MAIN = (*BASE, "trunk")
FUSIONS = (("f3oreft", MAIN), ("f3oreft_emb", (*MAIN, "emb")), ("f3oreft_embx", (*MAIN, "embx")), ("f3oref_embx", (*BASE, "embx")))
PRE = [("f3oreft_embx", "f3oreft")]
REPORT = [("f3oreft_emb", "f3oreft"), ("embx", "trunk"), ("emb", "trunk"), ("f3oref_embx", "f3oreft")]
OFFICIAL = ("player_identification_candidates_public.csv", "player_identification_candidates_private.csv")


def old_official_rows(folder, store):
    """{'old:<player_id>': [G, 256]} of the old official candidate pools (keys c:<row>:<game_id> as pid_trunk_extract)."""
    out = {}
    for name in OFFICIAL:
        cands = pools.read_candidates(Path(folder) / name)
        for i, r in enumerate(cands.itertuples()):
            out.setdefault(f"old:{r.player_id}", []).append(store[f"c:{i}:{r.game_id}"])
    return {p: np.stack(v) for p, v in out.items()}


def training_rows(games, folds_df, folds, store, aug):
    """({player: [G, 256]}, {player: [G, 256] augmented, plain where no augmented row}) of the given folds' Black games."""
    g = games.merge(folds_df, on="player_id")
    g = g[g.fold.isin(list(folds)) & g.game_key.isin(store.keys())]
    rows, arows = {}, {}
    for p, keys in g.groupby("player_id")["game_key"]:
        keys = list(keys)
        rows[p] = np.stack([store[k] for k in keys])
        arows[p] = np.stack([aug.get(k, store[k]) for k in keys])
    return rows, arows


def train(rows, arows, cfg, log):
    t = time.time()
    m = PlayerEmbedding(**cfg).fit(rows, arows, log=log)
    log(f"  trained on {m.players_trained} players in {time.time() - t:.0f} s", flush=True)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", nargs="+", required=True)
    ap.add_argument("--aug-store", nargs="*", default=[])
    ap.add_argument("--old-official", default=str(ROOT / "data" / "tests"))
    ap.add_argument("--old-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "official")])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-out", default=None)
    ap.add_argument("--final-model-out", default=None, help="repo-relative path (recorded in the fusion document as emb_model_official)")
    ap.add_argument("--fusion-out-plain", default=None, help="fusion document of main + emb trained on training players only (no official game)")
    ap.add_argument("--final-plain-model-out", default=None, help="repo-relative path of the training-only embedding (folds 1-4), recorded in --fusion-out-plain")
    a = ap.parse_args(argv)
    t0 = time.time()
    log = print
    ecfg = {"steps": a.steps, "min_games": a.min_games, "seed": 0}
    defaults = argparse.Namespace(n_players=pools.N_CANDIDATES, cand_games=pools.CAND_GAMES, seed=0, order="shuffle", max_questions=0, workers=a.workers,
                                  ref_min_black=30, ref_cap=40)
    cfg = {"depth": 12, "alpha": 2.0, "beta": 5.0, "shrink": 0.1, "local_alpha": 2.0, "local_beta": 5.0, "local_start": 12, "orient_alpha": 2.0,
           "orient_beta": 5.0, "orient_depth": 12, "orient_raw": 1, "ref_alpha": 2.0, "ref_beta": 5.0, "ref_min_black": 30, "ref_cap": 40, "ref_orient_raw": 2,
           "ref_seed": 0, "trunk_feats": "all", "trunk_shrink": 0.1}
    specs = parse_pools(a.pools)
    if len(specs) != 2 or any(0 in f for f in specs):
        print("give exactly two fold pairs without fold 0", file=sys.stderr)
        return 2
    store, aug = load_store(*a.store), (load_store(*a.aug_store) if a.aug_store else {})
    old = old_official_rows(a.old_official, load_store(*a.old_store))
    log(f"stores: {len(store)} training rows, {len(aug)} augmentation rows, {len(old)} old official players ({sum(len(v) for v in old.values())} rows); "
        f"device {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    if a.model_dir:
        Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for i, f in enumerate(specs):
        r = pid_ref_cv.run_pool(games, folds_df, f, specs[1 - i], defaults, cfg)
        model = fit_trunk(r["by_player"], store, cfg)
        if model.players != r["players"]:
            raise ValueError("candidate order differs")
        r["S"]["trunk"] = score_trunk(model, r["q_keys"], store)
        rows, arows = training_rows(games, folds_df, specs[1 - i], store, aug)
        if set(rows) & set(r["by_player"]):
            raise ValueError("a pool player is in the embedding's training set")
        tag = "p" + "".join(map(str, f))
        for name, extra in (("emb", {}), ("embx", old)):
            log(f"pool {f}: training {name} on folds {specs[1 - i]}" + (" + old official pools" if extra else ""), flush=True)
            m = train({**rows, **extra}, {**arows, **extra}, ecfg, log)
            r["S"][name] = score_embed(m, r["by_player"], r["q_keys"], store)
            r.setdefault("emb_history", {})[name] = m.history
            if a.model_dir:
                torch.save(m.state_dict(), Path(a.model_dir) / f"{name}_{tag}.pt")
        runs.append(r)
    comps_all = (*MAIN, "emb", "embx")
    arms = (*comps_all, *[n for n, _ in FUSIONS])
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []
    for i, r in enumerate(runs):
        others = [o for j, o in enumerate(runs) if j != i]
        weights, fused = {}, {}
        for name, comps in FUSIONS:
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))
            weights[name], fused[name] = fus.w.tolist(), fus.scores([r["S"][c] for c in comps])
        scores = {}
        for c in comps_all:
            scores[c], pq = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])
            per_q_all[c].append(pq)
        for name, _ in FUSIONS:
            scores[name], pq = evaluate(fused[name], r["players"], r["questions"], r["truth"])
            per_q_all[name].append(pq)
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))
        n_all.append(r["questions"]["n_games"].to_numpy())
        results.append({"pool": r["info"], "scores": scores, "weights": weights, "emb_history": r["emb_history"]})
        log(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores) + f"  embx weights {np.round(weights['f3oreft_embx'], 3).tolist()}", flush=True)
    pid, n_games = np.concatenate(pids_all), np.concatenate(n_all)
    comps = {}
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
        comps[group] = {}
        for x, z in pairs:
            qx, qz = np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z])
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]
            n5 = n_games == 5
            d["diff_n5"] = float(np.mean((qx - qz)[n5])) if n5.any() else None
            d.update({"per_pool_diff": per_pool, "by_n": {str(n): float(np.mean((qx - qz)[n_games == n])) for n in sorted(set(n_games.tolist()))},
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})
            comps[group][f"{x} vs {z}"] = d
            log(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)
                + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)
    out = {"pools": results, "overall_score": {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}, "comparisons": comps, "cfg": cfg,
           "embed_cfg": ecfg, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cuda" if torch.cuda.is_available() else "cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    log(f"wrote {a.out}")
    if a.fusion_out:
        comps_ = (*MAIN, "embx")
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps_], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": [*MAIN, "emb"], "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),
               "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", "emb_model_official": a.final_model_out,
               **pid_environment("cpu")}
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")
        log(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")
    if a.fusion_out_plain:  # 11-10: the 10-30 pools stay live in the 11-11 release, so an embedding that never saw an official game is kept ready
        comps_ = (*MAIN, "emb")
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps_], np.concatenate([r["truth"] for r in runs]))
        doc = {"components": list(comps_), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),
               "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", "emb_model_official": a.final_plain_model_out,
               "note": "emb trained on training folds 1-4 only: no official candidate or question game enters any training", **pid_environment("cpu")}
        Path(a.fusion_out_plain).parent.mkdir(parents=True, exist_ok=True)
        Path(a.fusion_out_plain).write_text(to_json(doc, indent=1) + "\n")
        log(f"wrote {a.fusion_out_plain}: weights {np.round(fus.w, 4).tolist()}")
    if a.final_plain_model_out:
        rows, arows = training_rows(games, folds_df, (1, 2, 3, 4), store, aug)
        log("final emb: folds 1-4 only", flush=True)
        m = train(rows, arows, ecfg, log)
        dest = Path(a.final_plain_model_out) if Path(a.final_plain_model_out).is_absolute() else ROOT / a.final_plain_model_out
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), dest)
        log(f"wrote {dest}")
    if a.final_model_out:
        rows, arows = training_rows(games, folds_df, (1, 2, 3, 4), store, aug)
        log("final embx: folds 1-4 + old official pools", flush=True)
        m = train({**rows, **old}, {**arows, **old}, ecfg, log)
        dest = Path(a.final_model_out) if Path(a.final_model_out).is_absolute() else ROOT / a.final_model_out  # repo-relative, as the document records it
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), dest)
        log(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
