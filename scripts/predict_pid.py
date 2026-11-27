#!/usr/bin/env python
"""Official player-identification files -> submission_player.csv (CPU components: opening model + style model + fusion).

For each (candidates, questions) pair -- public first, then private -- the candidates' games fit the opening and style
models of that pool, every question's games are scored against the 400 candidates, the components are fused with the
weights of --fusion (scripts/pid_cv.py --fusion-out, or pid_local_cv.py --fusion-out for the three-component fusion3; the
components may be a subset, e.g. opening alone as the fallback) and the
best five distinct candidates are written; --assign on|auto replaces the Top-1 by the one-to-one assignment (line B3). The submission is
validated (header, order, pool membership, distinct ids, --expect-rows) before exit; a report JSON goes next to it.
Exit codes: 2 input/format problems, 1 validation failure.

  python -I scripts/predict_pid.py --candidates tests/player_identification_candidates_public.csv tests/player_identification_candidates_private.csv \\
      --questions tests/player_identification_test_public.csv tests/player_identification_test_private.csv \\
      --fusion results/pid_fusion.json --out results/submission_player.csv --expect-rows 800 --workers 12
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402
TRAIN_DIR = aicup_pid.TRAIN_DIR  # the shared data location (also puts AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import checks, pools  # noqa: E402
from aicup_pid.assign import assign_blocks, assign_topk, assignment_report  # noqa: E402
from aicup_pid.components import ALL_COMPONENTS, COMPONENTS, fit_hd, fit_local, fit_models, fit_opening_raw, fit_orient, fit_ref, parse_all, score_components, score_ref
from aicup_pid.paths import FOLDS_CSV
from aicup_pid.ref import reference_players
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402


def predict_pool(cand_path, q_path, fus, cfg, workers, comps=COMPONENTS, assign="off", log=print, ref_folds=(1, 2, 3, 4), trunk_dirs=(), emb_path=None, b3_dir=None, orig_keys=None):
    """[(question_id, top5 ids)], report dict for one pool. orig_keys: the game tuples of an earlier release's questions; when given, the
    assignment rule and the assignment run separately on the questions found there (block "orig") and on the others ("added")."""
    t0 = time.time()
    cands = pools.read_candidates(cand_path)
    questions = pools.read_questions(q_path)
    sgfs = {f"c:{i}:{r.game_id}": r.sgf for i, r in enumerate(cands.itertuples())}  # positional: game_id need not be unique
    by_player = {}
    for i, r in enumerate(cands.itertuples()):
        by_player.setdefault(r.player_id, []).append(f"c:{i}:{r.game_id}")
    q_keys, dropped = [], []
    for q in questions:
        keys = []
        for i, (sgf, colour) in enumerate(q["games"]):
            ok, info = check_game(sgf, colour or "B")
            if not ok:
                dropped.append({"question_id": q["question_id"], "game": i + 1, "error": info["error"]})
                continue
            key = f"q:{q['question_id']}:{i}"
            sgfs[key] = sgf
            keys.append(key)
        q_keys.append(keys)
    log(f"{Path(cand_path).name}: {len(by_player)} candidates, {len(cands)} games; {Path(q_path).name}: {len(questions)} questions, "
        f"{sum(len(k) for k in q_keys)} games ({len(dropped)} dropped)", flush=True)
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=workers)
    local = fit_local(by_player, moves, cfg) if "local" in comps else None
    hd = fit_hd(by_player, sgfs, cfg) if "hd" in comps else None
    orient = fit_orient(by_player, moves, cfg) if "orient" in comps else None
    opening_raw = fit_opening_raw(by_player, moves, cfg) if "opening_raw" in comps else None
    usable = [k for k in q_keys if k]
    S_ref = None
    if "ref" in comps:  # training players (folds named by the document) as the anonymous reference basis
        t1 = time.time()
        from aicup.scoring import load_games
        folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
        refs = reference_players(pools.black_games(aicup_pid.TRAIN_DIR), folds_df, ref_folds, min_black=cfg.get("ref_min_black", 30), cap=cfg.get("ref_cap", 40), seed=cfg.get("ref_seed", 0))
        ref_sgfs = {k: v[0] for k, v in load_games(sorted({k for ks in refs.values() for k in ks}), aicup_pid.TRAIN_DIR).items()}
        ref_moves = parse_all(ref_sgfs, workers)
        ref_model = fit_ref(refs, ref_moves, cfg)
        S_ref = score_ref(ref_model, by_player, usable, moves)
        log(f"reference basis: {len(refs)} training players from folds {ref_folds}, {len(ref_sgfs)} games, {ref_model.logp.shape[1]} tokens in {time.time() - t1:.0f} s", flush=True)
    S_trunk = None
    if "trunk" in comps:  # per-game trunk rows of the official games (scripts/pid_trunk_extract.py --official), keyed like sgfs here
        t1 = time.time()
        store = load_store(*trunk_dirs)
        S_trunk = score_trunk(fit_trunk(by_player, store, cfg), usable, store)
        log(f"trunk-style component from {len(store)} stored game rows in {time.time() - t1:.0f} s", flush=True)
    S_emb = None
    if "emb" in comps:  # line B8: learnt player embedding on the same trunk rows (trained by scripts/pid_embed_cv.py --final-model-out)
        t1 = time.time()
        import torch
        from aicup_pid.embed import PlayerEmbedding, score_embed
        if not emb_path:
            raise ValueError("the fusion names 'emb' but gives no emb_model_official")
        store = load_store(*trunk_dirs)
        emb_model = PlayerEmbedding.from_state(torch.load(str(ROOT / emb_path), map_location="cpu", weights_only=False))
        S_emb = score_embed(emb_model, by_player, usable, store)
        log(f"embedding component ({emb_path}, {emb_model.players_trained} training players) in {time.time() - t1:.0f} s", flush=True)
    S = score_components([c for c in comps if c != "b3"], opening, style, usable, moves, feats, local, hd, sgfs, orient, opening_raw, S_ref, S_trunk, S_emb)
    b3_info = None
    if "b3" in comps:  # line B3 (METHOD_NOTES 2.26): ct9's zq captured by experiments/pid_b3_official.py; only for the no-assignment fallback
        from aicup_pid.b3 import score_b3
        if not b3_dir:
            raise ValueError("the fusion names 'b3' but no --b3-dir was given")
        prefix = questions[0]["question_id"].split("_")[0]
        z = np.load(Path(b3_dir) / prefix / "b3.npz")
        cmap = pd.read_csv(Path(b3_dir) / prefix / "inputs" / "cands_map.csv", dtype=str, keep_default_na=False)
        if set(cmap.player_id) != set(by_player):
            raise ValueError(f"{b3_dir}/{prefix}: the captured candidates differ from {cand_path}")
        chunks = cmap.groupby("player_id")["question_id"].apply(list).to_dict()
        S["b3"], b3_info = score_b3(z["zq"], {q: i for i, q in enumerate(z["qids"])}, chunks, list(opening.players),
                                    [q["question_id"] for q, keys in zip(questions, q_keys) if keys])
        b3_info = {k: (len(v) if isinstance(v, list) else v) for k, v in b3_info.items()}
        log(f"b3 component ({b3_dir}/{prefix}): {b3_info}", flush=True)
    S_f = fus.scores([S[c] for c in comps]) if usable else np.zeros((0, len(opening.players)))
    n_games = np.array([q["declared_games"] for q, keys in zip(questions, q_keys) if keys], dtype=int)
    collisions = checks.top1_collisions(S_f, n_games)
    if orig_keys is None:
        applied = assign == "on" or (assign == "auto" and bool(collisions["one_to_one_likely_v2"]))
        idx = assign_topk(S_f, pools.TOP_K) if applied else topk(S_f, pools.TOP_K)
        assign_rep = {"mode": assign, "applied": applied, "top1_collisions": collisions, **({k: v for k, v in assignment_report(S_f, idx).items() if k == "n_top1_changed"} if applied else {"n_top1_changed": 0})}
    else:  # 11-11 release (METHOD_NOTES 2.27): original and added questions are separate sets over the same candidates
        labels = np.array(["orig" if tuple(sgf for sgf, _c in q["games"]) in orig_keys else "added" for q, keys in zip(questions, q_keys) if keys])
        base, blocks, apply = topk(S_f, pools.TOP_K), {}, {}
        for b in sorted(set(labels.tolist())):
            m = labels == b
            col = checks.top1_collisions(S_f[m], n_games[m])
            apply[b] = assign == "on" or (assign == "auto" and bool(col["one_to_one_likely_v2"]))
            blocks[b] = {"applied": apply[b], "top1_collisions": col}
        idx = assign_blocks(S_f, labels, apply, pools.TOP_K) if len(labels) else base
        for b in blocks:
            blocks[b]["n_top1_changed"] = int((idx[labels == b, 0] != base[labels == b, 0]).sum())
        applied = any(apply.values())
        assign_rep = {"mode": assign, "applied": applied, "top1_collisions": collisions, "n_top1_changed": int((idx[:, 0] != base[:, 0]).sum()) if len(labels) else 0, "blocks": blocks}
        log("assignment by block: " + ", ".join(f"{b}: {'applied' if v['applied'] else 'not applied'} (distinct Top-1 {v['top1_collisions']['distinct_top1']}/{v['top1_collisions']['n_questions']}, "
                                                 f"changed {v['n_top1_changed']})" for b, v in blocks.items()), flush=True)
    rows, j, fallback = [], 0, []
    for q, keys in zip(questions, q_keys):
        if keys:
            rows.append((q["question_id"], [opening.players[i] for i in idx[j]]))
            j += 1
        else:  # no usable game: the five first candidates (recorded; the score of such a question is luck)
            rows.append((q["question_id"], opening.players[:pools.TOP_K]))
            fallback.append(q["question_id"])
    margins = (np.sort(S_f, axis=1)[:, -1] - np.sort(S_f, axis=1)[:, -2]).tolist() if usable else []
    report = {"candidates": str(cand_path), "questions": str(q_path), "n_candidates": len(by_player), "n_candidate_games": int(len(cands)),
              "n_candidate_games_used": int(sum(len(k) for k in by_player.values())), "candidate_game_ids_unique": bool(cands["game_id"].is_unique),
              "n_questions": len(questions), "dropped_games": dropped, "fallback_questions": fallback, "median_top1_margin": float(np.median(margins)) if margins else None,
              "assign": assign_rep, "b3": b3_info,
              "seconds": time.time() - t0}
    return rows, report, set(by_player)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", nargs="+", required=True, help="candidate CSVs, public then private")
    ap.add_argument("--questions", nargs="+", required=True, help="test CSVs in the same order")
    ap.add_argument("--fusion", required=True, help="pid_cv.py --fusion-out JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-rows", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--trunk-store", nargs="*", default=None, help="trunk rows of these official files (default: the fusion document's trunk_store_official); a re-released set has its own store")
    ap.add_argument("--assign-blocks", nargs="*", default=None, help="test CSVs of an earlier release: the questions found there (same games) form block 'orig', the rest 'added'; "
                    "the assignment rule and the assignment then run block by block")
    ap.add_argument("--b3-dir", default=None, help="experiments/pid_b3_official.py work folder (<dir>/pub, <dir>/priv) for a fusion that names 'b3'")
    ap.add_argument("--assign", choices=["off", "auto", "on"], default="off",
                    help="line B3: one-to-one assignment of the Top-1 (auto = only when the Top-1 picks name >= 87.5%% distinct candidates, or >= 95%% "
                         "distinct among the more confident half with no more questions than candidates: aicup_pid.checks.top1_collisions)")
    a = ap.parse_args(argv)
    if len(a.candidates) != len(a.questions):
        ap.error("--candidates and --questions must pair up")
    try:
        doc = json.loads(Path(a.fusion).read_text())
        comps = list(doc["components"])
        if not comps or len(set(comps)) != len(comps) or any(c not in (*ALL_COMPONENTS, "b3") for c in comps):
            raise ValueError(f"{a.fusion}: components {comps} must be a non-empty subset of {[*ALL_COMPONENTS, 'b3']}")
        if len(doc["weights"]) != len(comps):
            raise ValueError(f"{a.fusion}: {len(doc['weights'])} weights for {len(comps)} components")
        fus = Fusion()
        fus.w = np.asarray(doc["weights"], dtype=float)
        rows, reports, pool_players = [], [], {}
        orig_keys = None
        if a.assign_blocks:
            orig_keys = {tuple(sgf for sgf, _c in q["games"]) for f in a.assign_blocks for q in pools.read_questions(f)}
        for cand_path, q_path in zip(a.candidates, a.questions):
            r, rep, players = predict_pool(cand_path, q_path, fus, doc.get("cfg", {}), a.workers, comps=comps, assign=a.assign, ref_folds=tuple(doc.get("ref_folds_official", (1, 2, 3, 4))),
                                           trunk_dirs=tuple(a.trunk_store) if a.trunk_store else (str(ROOT / doc.get("trunk_store_official", "data/trunk/official")),),
                                           emb_path=doc.get("emb_model_official"), b3_dir=a.b3_dir, orig_keys=orig_keys)
            prefix = next((p for p in pools.PREFIXES if r and r[0][0].startswith(p)), None)
            if prefix is not None:
                pool_players[prefix] = players
            rows += r
            reports.append(rep)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    pools.write_submission(rows, a.out)
    problems = pools.validate_submission(a.out, [q for q, _ in rows], pool_players, a.expect_rows)
    report = {"submission": str(a.out), "rows": len(rows), "pools": reports, "fusion": doc, "problems": problems, **pid_environment("cpu")}
    Path(a.out).with_suffix(".report.json").write_text(to_json(report, indent=1) + "\n")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"wrote {a.out}: {len(rows)} rows, {sum(len(r['fallback_questions']) for r in reports)} fallback questions; assignment "
          + ", ".join(f"{Path(r['questions']).name}: {'applied' if r['assign']['applied'] else 'not applied'} (distinct Top-1 {r['assign']['top1_collisions']['distinct_top1']}/{r['assign']['top1_collisions']['n_questions']}, changed {r['assign']['n_top1_changed']})" for r in reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
