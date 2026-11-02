#!/usr/bin/env python
"""The 11/04 checks of the official player-identification files, before any modelling or upload (fork plan section 3).

Per (candidates, questions) pair -- public then private -- this reports: legality of every record (check_game), whole-game
duplicates against the training set (8 board symmetries; check 1), duplicates between questions and candidates, between
questions, and between the two pools' candidates (2), length quantiles (4), the opening-distribution shift of the
question games against the candidate games with a split-half null (6), the within-library self-identification of the
candidates (7: 50 games fit the models, the other 50 form 5/10/15/20-game questions; the label-free estimate of how
separable the 400 candidates are), the main line's Top-1 collisions (8: are the questions one per candidate?), the
num_games counts (9) and the record properties (10). Everything is a report; the flags A1 (training overlap), A3(i)
(opening shift in both pools) and one_to_one_likely decide which line of the fork plan opens. Exit 2 on input errors.

  python -I scripts/pid_check_files.py --candidates data/tests/player_identification_candidates_public.csv data/tests/player_identification_candidates_private.csv \\
      --questions data/tests/player_identification_test_public.csv data/tests/player_identification_test_private.csv \\
      --fusion results/pid_fusion.json --out results/checks/official_checks.json --workers 24
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
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup_pid import checks, pools  # noqa: E402
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, parse_all, score_questions  # noqa: E402
from aicup_pid.dedup import build_training_index, hash_games  # noqa: E402
from aicup_pid.fusion import Fusion  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

SHIFT_DEPTHS = (4, 8)


def _lengths(hashed, keys):
    rows = [hashed[k] for k in keys if hashed[k][1] > 0]
    return checks.length_quantiles([r[1] for r in rows], [r[2] for r in rows])


def _fmt(x, nd=4):
    return "none" if x is None else f"{x:.{nd}f}"


def check_pool(cand_path, q_path, fus, cfg, a, index, log=print):
    """The checks of one pool: (result dict, {candidate key: hash}) ."""
    t0 = time.time()
    cands = pools.read_candidates(cand_path)
    questions = pools.read_questions(q_path)
    prefix = next((p for p in pools.PREFIXES if questions and questions[0]["question_id"].startswith(p)), None)
    cand_sgfs = {f"c:{i}:{r.game_id}": r.sgf for i, r in enumerate(cands.itertuples())}  # positional: game_id need not be unique
    owner = {f"c:{i}:{r.game_id}": r.player_id for i, r in enumerate(cands.itertuples())}
    q_sgfs, q_owner, q_keys = {}, {}, []
    for q in questions:
        keys = []
        for i, (sgf, _colour) in enumerate(q["games"]):
            k = f"q:{q['question_id']}:{i}"
            q_sgfs[k], q_owner[k] = sgf, q["question_id"]
            keys.append(k)
        q_keys.append(keys)
    leg_c, leg_q = checks.legality(cand_sgfs), checks.legality(q_sgfs)
    usable = set(leg_c["usable_keys"]) | set(leg_q["usable_keys"])
    hashed_c, hashed_q = hash_games(cand_sgfs, a.workers), hash_games(q_sgfs, a.workers)
    hc, hq = {k: v[0] for k, v in hashed_c.items()}, {k: v[0] for k, v in hashed_q.items()}
    overlap = None
    if index is not None:
        overlap = {"candidates": checks.summarise_overlap(checks.duplicates_against_index(hc, index), owner),
                   "questions": checks.summarise_overlap(checks.duplicates_against_index(hq, index), q_owner)}
    qc_dups, qq_dups = checks.duplicates_between(hq, hc), checks.duplicates_within(hq)
    sgfs = {k: cand_sgfs[k] for k in leg_c["usable_keys"]}
    sgfs.update({k: q_sgfs[k] for k in leg_q["usable_keys"]})
    moves, feats = parse_all(sgfs, a.workers), features_all(sgfs, a.workers)
    by_player = {}
    for k in leg_c["usable_keys"]:
        by_player.setdefault(owner[k], []).append(k)
    no_games = sorted(set(owner.values()) - set(by_player))
    q_moves = [moves[k] for k in leg_q["usable_keys"]]
    c_moves = [moves[k] for k in leg_c["usable_keys"]]
    shift = {str(k): checks.opening_shift(q_moves, c_moves, k, n_null=a.n_null, seed=a.seed) for k in SHIFT_DEPTHS}
    si = checks.self_identification(by_player, moves, feats, cfg, fus.w, seed=a.seed, model_games=a.model_games)
    # the main line on the official questions (as predict_pid.py fits it), for the Top-1 collisions and the margins
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    q_usable = [[k for k in keys if k in usable] for keys in q_keys]
    idx = [i for i, keys in enumerate(q_usable) if keys]
    if idx:
        S = score_questions(opening, style, [q_usable[i] for i in idx], moves, feats)
        S_f = fus.scores([S[c] for c in COMPONENTS])
    else:
        S_f = np.zeros((0, len(opening.players)))
    n_games = np.array([questions[i]["declared_games"] for i in idx], dtype=int)
    collisions = checks.top1_collisions(S_f, n_games)
    top1 = [[q["question_id"], None] for q in questions]
    for j, i in enumerate(idx):
        top1[i][1] = opening.players[int(np.argmax(S_f[j]))]
    srt = np.sort(S_f, axis=1)
    margins = (srt[:, -1] - srt[:, -2]) if len(idx) and S_f.shape[1] >= 2 else np.zeros(0)
    res = {"prefix": prefix, "candidates": str(cand_path), "questions": str(q_path), "n_candidates": int(cands["player_id"].nunique()),
           "n_candidate_games": int(len(cands)), "candidates_without_usable_games": no_games[:MAX_LIST] if (MAX_LIST := 20) else no_games,
           "n_questions": len(questions), "num_games": checks.num_games_counts(questions),
           "legality": {"candidates": {k: v for k, v in leg_c.items() if k != "usable_keys"}, "questions": {k: v for k, v in leg_q.items() if k != "usable_keys"}},
           "lengths": {"candidates": _lengths(hashed_c, cand_sgfs), "questions": _lengths(hashed_q, q_sgfs)},
           "training_overlap": overlap, "question_vs_candidate_dups": len(qc_dups), "question_vs_candidate_examples": qc_dups[:20],
           "question_vs_question_dups": len(qq_dups), "question_vs_question_examples": qq_dups[:20],
           "opening_shift": shift, "self_identification": si,
           "main_line": {"top1_collisions": collisions, "margin_quantiles": checks._quantiles(margins), "questions_without_usable_games": len(questions) - len(idx), "top1": top1},
           "records": {"candidates": checks.record_properties(cand_sgfs, cands["game_id"].tolist()), "questions": checks.record_properties(q_sgfs, [])},
           "seconds": time.time() - t0}
    log(f"{prefix}: {res['n_candidates']} candidates / {res['n_candidate_games']} games, {len(questions)} questions {res['num_games']}; "
        f"illegal records: candidates {leg_c['n_bad']}, questions {leg_q['n_bad']}; candidates without a usable game {len(no_games)}", flush=True)
    if overlap is not None:
        oc, oq = overlap["candidates"], overlap["questions"]
        log(f"{prefix}: training overlap (A1) candidates {oc['n_dup_games']}/{oc['n_games_checked']} games, {oc['n_players_with_dups']} players "
            f"(single training id {oc['players_single_training_id']}, same id {oc['same_id_players']}); questions {oq['n_dup_games']}/{oq['n_games_checked']}", flush=True)
    else:
        log(f"{prefix}: training overlap (A1) not checked (--no-training)", flush=True)
    log(f"{prefix}: question-vs-candidate duplicates {len(qc_dups)}, question-vs-question {len(qq_dups)}", flush=True)
    log(f"{prefix}: opening shift (A3 i) " + "; ".join(f"k={k} observed {_fmt(s['observed'])} null max {_fmt(s['null_max'])} ratio {_fmt(s['ratio_to_null_max'], 2)} flag {s['flag']}"
                                                      for k, s in shift.items()), flush=True)
    if si["scores"]:
        f = si["scores"]["fusion"]
        by5 = f["by_n"].get("5", {}).get("score")
        log(f"{prefix}: self-identification (A3 ii) fusion score {f['score']:.4f} top1 {f['top1']:.3f} top5 {f['top5']:.3f} n=5 {_fmt(by5)}; "
            f"opening {si['scores']['opening']['score']:.4f} style {si['scores']['style']['score']:.4f} ({si['n_questions']} questions, {si['n_candidates']} candidates, {si['n_skipped']} skipped)", flush=True)
    else:
        log(f"{prefix}: self-identification (A3 ii) not computable ({si['n_candidates']} candidates, {si['n_skipped']} skipped)", flush=True)
    log(f"{prefix}: main line Top-1 distinct {collisions['distinct_top1']}/{collisions['n_questions']} (random {collisions['expected_random']:.0f}) "
        f"one-to-one likely {collisions['one_to_one_likely']}; margin median {_fmt((res['main_line']['margin_quantiles'] or {}).get('p50'))}; {res['seconds']:.0f} s", flush=True)
    return res, hc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", nargs="+", required=True, help="candidate CSVs, public then private")
    ap.add_argument("--questions", nargs="+", required=True, help="test CSVs in the same order")
    ap.add_argument("--fusion", required=True, help="pid_cv.py --fusion-out JSON (main-line weights and cfg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-dir", default=str(aicup_pid.TRAIN_DIR))
    ap.add_argument("--no-training", action="store_true", help="skip the training-set overlap (check 1)")
    ap.add_argument("--max-train-games", type=int, default=None, help="smoke tests only")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--model-games", type=int, default=50)
    a = ap.parse_args(argv)
    if len(a.candidates) != len(a.questions):
        ap.error("--candidates and --questions must pair up")
    t0 = time.time()
    try:
        doc = json.loads(Path(a.fusion).read_text())
        if list(doc["components"]) != list(COMPONENTS):
            raise ValueError(f"{a.fusion}: components {doc['components']} != {list(COMPONENTS)}")
        fus = Fusion()
        fus.w = np.asarray(doc["weights"], dtype=float)
        cfg = doc.get("cfg", {})
        index = None
        if not a.no_training:
            index = build_training_index(a.train_dir, workers=a.workers, max_games=a.max_train_games)
            print(f"training index: {len(index.meta)} games in {index.seconds:.0f} s", flush=True)
        results, hashes = [], []
        for cand_path, q_path in zip(a.candidates, a.questions):
            r, hc = check_pool(cand_path, q_path, fus, cfg, a, index)
            results.append(r)
            hashes.append(hc)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    cross = len(checks.duplicates_between(hashes[0], hashes[1])) if len(hashes) == 2 else None
    a1 = None if index is None else any(r["training_overlap"]["candidates"]["n_dup_games"] > 0 or r["training_overlap"]["questions"]["n_dup_games"] > 0 for r in results)
    shift_flags = [r["opening_shift"]["4"]["flag"] for r in results if r["opening_shift"]["4"]["flag"] is not None]
    flags = {"A1_training_overlap": a1, "A3i_opening_shift": (all(shift_flags) if shift_flags else None),
             "one_to_one_likely": {r["prefix"]: r["main_line"]["top1_collisions"]["one_to_one_likely"] for r in results}}
    training = None if index is None else {"train_dir": str(a.train_dir), "n_games": int(len(index.meta)),
                                           "lengths_black": checks.length_quantiles(*index.black_lengths()), "seconds": index.seconds}
    out = {"pools": results, "cross_pool_candidate_dups": cross, "training": training, "flags": flags, "fusion": doc,
           "args": {k: v for k, v in vars(a).items()}, "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"cross-pool candidate duplicates: {cross}", flush=True)
    print(f"flags: A1 training overlap {flags['A1_training_overlap']}; A3(i) opening shift {flags['A3i_opening_shift']}; one-to-one likely {flags['one_to_one_likely']}", flush=True)
    print(f"wrote {a.out} ({out['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
