#!/usr/bin/env python
"""Official player-identification files -> submission_player.csv (CPU components: opening model + style model + fusion).

For each (candidates, questions) pair -- public first, then private -- the candidates' games fit the opening and style
models of that pool, every question's games are scored against the 400 candidates, the components are fused with the
weights of --fusion (scripts/pid_cv.py --fusion-out) and the best five distinct candidates are written. The submission is
validated (header, order, pool membership, distinct ids, --expect-rows) before exit; a report JSON goes next to it.
Exit codes: 2 input/format problems, 1 validation failure.

  python -I scripts/predict_pid.py --candidates tests/player_identification_candidates_public.csv tests/player_identification_candidates_private.csv \\
      --questions tests/player_identification_test_public.csv tests/player_identification_test_private.csv \\
      --fusion results/pid_fusion.json --out results/submission_player.csv --expect-rows 800 --workers 24
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
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.submission import check_game  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import COMPONENTS, fit_models, score_questions  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402


def predict_pool(cand_path, q_path, fus, cfg, workers, log=print):
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
    usable = [k for k in q_keys if k]
    S = score_questions(opening, style, usable, moves, feats)
    S_f = fus.scores([S[c] for c in COMPONENTS]) if usable else np.zeros((0, len(opening.players)))
    idx = topk(S_f, pools.TOP_K)
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
    a = ap.parse_args(argv)
    if len(a.candidates) != len(a.questions):
        ap.error("--candidates and --questions must pair up")
    try:
        doc = json.loads(Path(a.fusion).read_text())
        if list(doc["components"]) != list(COMPONENTS):
            raise ValueError(f"{a.fusion}: components {doc['components']} != {list(COMPONENTS)}")
        fus = Fusion()
        fus.w = np.asarray(doc["weights"], dtype=float)
        rows, reports, pool_players = [], [], {}
        for cand_path, q_path in zip(a.candidates, a.questions):
            r, rep, players = predict_pool(cand_path, q_path, fus, doc.get("cfg", {}), a.workers)
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
    report = {"submission": str(a.out), "rows": len(rows), "pools": reports, "fusion": doc, "problems": problems, **environment("cpu")}
    Path(a.out).with_suffix(".report.json").write_text(to_json(report, indent=1) + "\n")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"wrote {a.out}: {len(rows)} rows, {sum(len(r['fallback_questions']) for r in reports)} fallback questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
