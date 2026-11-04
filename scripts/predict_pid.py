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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402
TRAIN_DIR = aicup_pid.TRAIN_DIR  # the shared data location (also puts AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import checks, pools  # noqa: E402
from aicup_pid.assign import assign_topk, assignment_report  # noqa: E402
from aicup_pid.components import ALL_COMPONENTS, COMPONENTS, fit_hd, fit_local, fit_models, fit_orient, score_components  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402


def predict_pool(cand_path, q_path, fus, cfg, workers, comps=COMPONENTS, assign="off", log=print):
    """[(question_id, top5 ids)], report dict for one pool."""
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
    usable = [k for k in q_keys if k]
    S = score_components(comps, opening, style, usable, moves, feats, local, hd, sgfs, orient)
    S_f = fus.scores([S[c] for c in comps]) if usable else np.zeros((0, len(opening.players)))
    n_games = np.array([q["declared_games"] for q, keys in zip(questions, q_keys) if keys], dtype=int)
    collisions = checks.top1_collisions(S_f, n_games)
    applied = assign == "on" or (assign == "auto" and bool(collisions["one_to_one_likely"]))
    idx = assign_topk(S_f, pools.TOP_K) if applied else topk(S_f, pools.TOP_K)
    assign_rep = {"mode": assign, "applied": applied, "top1_collisions": collisions, **({k: v for k, v in assignment_report(S_f, idx).items() if k == "n_top1_changed"} if applied else {"n_top1_changed": 0})}
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
              "assign": assign_rep,
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
    ap.add_argument("--assign", choices=["off", "auto", "on"], default="off",
                    help="line B3: one-to-one assignment of the Top-1 (auto = only when the pool's Top-1 picks name >= 87.5%% distinct candidates)")
    a = ap.parse_args(argv)
    if len(a.candidates) != len(a.questions):
        ap.error("--candidates and --questions must pair up")
    try:
        doc = json.loads(Path(a.fusion).read_text())
        comps = list(doc["components"])
        if not comps or len(set(comps)) != len(comps) or any(c not in ALL_COMPONENTS for c in comps):
            raise ValueError(f"{a.fusion}: components {comps} must be a non-empty subset of {list(ALL_COMPONENTS)}")
        if len(doc["weights"]) != len(comps):
            raise ValueError(f"{a.fusion}: {len(doc['weights'])} weights for {len(comps)} components")
        fus = Fusion()
        fus.w = np.asarray(doc["weights"], dtype=float)
        rows, reports, pool_players = [], [], {}
        for cand_path, q_path in zip(a.candidates, a.questions):
            r, rep, players = predict_pool(cand_path, q_path, fus, doc.get("cfg", {}), a.workers, comps=comps, assign=a.assign)
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
