"""The 11/04 official-file checks as pure functions (docs/records/specs/2026-11-03-pid-fork-plan-design.md section 3).

Every function takes in-memory inputs (hashes, parsed moves, style feature rows) and returns a JSON-serialisable dict:
duplicates against the training index and between files (checks 1-2), length quantiles (4), legality (5), the opening-
distribution shift with a split-half null (6), the within-library self-identification (7), Top-1 collisions of the main
line (8), num_games counts (9) and record properties (10).
"""
import math
from collections import Counter

import numpy as np

from aicup.sgf import handicap, properties
from aicup.submission import check_game
from aicup_pid import pools
from aicup_pid.components import COMPONENTS, fit_from_parsed, score_questions
from aicup_pid.fusion import Fusion, topk
from aicup_pid.opening import canonical

QUANTILES = {"p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}
ONE_TO_ONE_FRACTION = 0.875  # 350 of 400 distinct Top-1 candidates
SHIFT_FACTOR = 1.5
MAX_EXAMPLES = 20
MAX_PLAYERS_LISTED = 50


# ---------------------------------------------------------------- duplicates (checks 1-2)

def duplicates_against_index(hashes, index):
    """{key: [{player_id, rank, game_key}]} for the keys whose hash is in the training index."""
    out = {}
    for k, h in hashes.items():
        if h is None:
            continue
        rows = index.lookup(h)
        if rows:
            m = index.meta.iloc[rows]
            out[k] = [{"player_id": r.player_id, "rank": r.rank, "game_key": r.game_key} for r in m.itertuples()]
    return out


def summarise_overlap(dups, owner):
    """Per owning player (owner: {key: player_id}) how many games hit which training players."""
    by_player = {}
    for k, hits in dups.items():
        c = by_player.setdefault(owner[k], Counter())
        for h in hits:
            c[h["player_id"]] += 1
    single = sum(1 for c in by_player.values() if len(c) == 1)
    listed = sorted(by_player)[:MAX_PLAYERS_LISTED]
    return {"n_games_checked": len(owner), "n_dup_games": len(dups), "n_players_with_dups": len(by_player),
            "players_single_training_id": single, "players_multiple_training_ids": len(by_player) - single,
            "same_id_players": sum(1 for p, c in by_player.items() if p in c),
            "by_player": {p: dict(by_player[p]) for p in listed}, "players_not_listed": max(len(by_player) - MAX_PLAYERS_LISTED, 0)}


def duplicates_between(hashes_a, hashes_b):
    """[(key_a, key_b)] sharing a hash (None never matches), in hashes_a order."""
    by_hash = {}
    for k, h in hashes_b.items():
        if h is not None:
            by_hash.setdefault(h, []).append(k)
    return [(ka, kb) for ka, h in hashes_a.items() if h is not None for kb in by_hash.get(h, ())]


def duplicates_within(hashes):
    """[(key_i, key_j)] pairs of distinct keys sharing a hash, insertion order."""
    groups = {}
    for k, h in hashes.items():
        if h is not None:
            groups.setdefault(h, []).append(k)
    return [(ks[i], ks[j]) for ks in groups.values() if len(ks) > 1 for i in range(len(ks)) for j in range(i + 1, len(ks))]


# ---------------------------------------------------------------- lengths, legality (checks 4-5)

def _quantiles(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    out = {k: float(np.percentile(x, q)) for k, q in QUANTILES.items()}
    out["mean"] = float(x.mean())
    return out


def length_quantiles(n_moves, n_black):
    n_moves, n_black = np.asarray(n_moves), np.asarray(n_black)
    return {"n": int(len(n_moves)), "moves": _quantiles(n_moves), "black_moves": _quantiles(n_black)}


def legality(sgfs, colour="B"):
    """check_game over every record: counts, the first MAX_EXAMPLES errors, HD counts and the usable keys in order."""
    n_bad, errors, hd, usable = 0, [], Counter(), []
    for k, sgf in sgfs.items():
        ok, info = check_game(sgf, colour)
        hd[str(info["handicap"]) if info["handicap"] is not None else "none"] += 1
        if ok:
            usable.append(k)
        else:
            n_bad += 1
            if len(errors) < MAX_EXAMPLES:
                errors.append({"key": k, "error": info["error"]})
    return {"n": len(sgfs), "n_bad": n_bad, "errors": errors, "handicap_counts": dict(hd), "usable_keys": usable}


# ---------------------------------------------------------------- opening-distribution shift (check 6)

def black_prefix(moves, k):
    """Canonical coordinates of Black's first k moves among the first 2k plies (the opening model's token path)."""
    cs, _ = canonical(moves, 2 * k)
    return tuple(cs[i] for i, (c, *_) in enumerate(moves[:2 * k]) if c == "B")[:k]


def js_divergence(count_p, count_q):
    """Jensen-Shannon divergence in bits (0-1) between two count dicts."""
    np_, nq = sum(count_p.values()), sum(count_q.values())
    if np_ == 0 or nq == 0:
        return float("nan")
    total = 0.0
    for key in set(count_p) | set(count_q):
        p, q = count_p.get(key, 0) / np_, count_q.get(key, 0) / nq
        m = 0.5 * (p + q)
        if p > 0:
            total += 0.5 * p * math.log2(p / m)
        if q > 0:
            total += 0.5 * q * math.log2(q / m)
    return float(total)


def opening_shift(q_moves, c_moves, k, n_null=20, seed=0, factor=SHIFT_FACTOR):
    """JS divergence between the query games' and the candidate games' Black-prefix distributions, against a null of
    random candidate splits of the same size; flag = observed > factor * max(null)."""
    tq = [black_prefix(m, k) for m in q_moves]
    tc = [black_prefix(m, k) for m in c_moves]
    out = {"k": int(k), "n_query": len(tq), "n_candidate": len(tc), "observed": None, "null": [], "null_mean": None, "null_max": None,
           "ratio_to_null_max": None, "flag": None}
    if not tq or len(tc) < 2:
        return out
    out["observed"] = js_divergence(Counter(tq), Counter(tc))
    nq = min(len(tq), len(tc) // 2)
    rng = np.random.RandomState(seed)
    for _ in range(n_null):
        perm = rng.permutation(len(tc))
        out["null"].append(js_divergence(Counter(tc[i] for i in perm[:nq]), Counter(tc[i] for i in perm[nq:])))
    out["null_mean"], out["null_max"] = float(np.mean(out["null"])), float(np.max(out["null"]))
    out["ratio_to_null_max"] = float(out["observed"] / out["null_max"]) if out["null_max"] > 0 else None
    out["flag"] = bool(out["observed"] > factor * out["null_max"])
    return out


# ---------------------------------------------------------------- within-library self-identification (check 7)

def self_identification(by_player, moves, feats, cfg, weights, seed=0, model_games=50, sizes=pools.SIZES):
    """Each candidate's games: model_games fit the models, the rest are cut into questions of `sizes` (ascending, while
    games remain); candidates with fewer than model_games + min(sizes) games are skipped. Returns the official score of
    every component and the fusion (weights = pid_fusion.json), the fusion margin quantiles and how many candidates have
    at least one question whose Top-1 is someone else."""
    sizes = sorted(int(m) for m in sizes)
    model_keys, q_rows, skipped = {}, [], 0
    for p in sorted(by_player):
        keys = sorted(by_player[p])
        if len(keys) < model_games + sizes[0]:
            skipped += 1
            continue
        keys = [keys[i] for i in pools._player_rng(seed, p).permutation(len(keys))]
        model_keys[p] = keys[:model_games]
        rest = keys[model_games:]
        for m in sizes:
            if len(rest) < m:
                break
            q_rows.append((p, m, rest[:m]))
            rest = rest[m:]
    out = {"n_candidates": len(model_keys), "n_skipped": skipped, "n_questions": len(q_rows), "model_games": int(model_games),
           "scores": None, "fusion_margin_quantiles": None, "candidates_misidentified": None}
    if not model_keys or not q_rows:
        return out
    opening, style = fit_from_parsed(model_keys, moves, feats, cfg)
    S = score_questions(opening, style, [ks for _, _, ks in q_rows], moves, feats)
    fus = Fusion()
    fus.w = np.asarray(weights, dtype=float)
    S_f = fus.scores([S[c] for c in COMPONENTS])
    truth = [p for p, _, _ in q_rows]
    n_games = np.array([m for _, m, _ in q_rows])
    scores = {}
    for name, M in (*S.items(), ("fusion", S_f)):
        tops = [[opening.players[j] for j in row] for row in topk(M, pools.TOP_K)]
        scores[name] = pools.summarise(tops, truth, n_games)
    srt = np.sort(S_f, axis=1)
    margins = srt[:, -1] - srt[:, -2] if S_f.shape[1] >= 2 else np.zeros(len(q_rows))
    top1 = [opening.players[j] for j in np.argmax(S_f, axis=1)]
    out.update({"scores": scores, "fusion_margin_quantiles": _quantiles(margins),
                "candidates_misidentified": len({p for p, t in zip(truth, top1) if p != t})})
    return out


# ---------------------------------------------------------------- Top-1 collisions (check 8), num_games (9), records (10)

CONFIDENT_FRACTION, CONFIDENT_DISTINCT = 0.5, 0.95  # rule v2 (METHOD_NOTES 2.27), fixed 11-10 from the birthday rate before its simulation


def confident_distinct(S_f, frac=CONFIDENT_FRACTION):
    """Share of distinct Top-1 candidates among the `frac` most confident questions (largest Top-1 margin). One question per
    candidate -> about 1 whatever the difficulty (confident picks are right, so they do not collide); questions drawn with
    replacement over C candidates -> at most the birthday rate C (1 - (1 - 1/C)^m) / m (0.79 for m = 200, C = 400)."""
    S = np.asarray(S_f, dtype=float)
    if len(S) < 2 or S.shape[1] < 2:
        return 1.0
    top = np.sort(S, axis=1)
    keep = np.argsort(-(top[:, -1] - top[:, -2]), kind="stable")[:int(np.ceil(frac * len(S)))]
    return float(len(set(S[keep].argmax(axis=1).tolist())) / len(keep))


def top1_collisions(S_f, n_games, fraction=ONE_TO_ONE_FRACTION):
    """How many distinct candidates the questions' Top-1 picks name, overall and per size, against the random expectation
    C (1 - (1 - 1/C)^Q); one_to_one_likely when distinct >= fraction * Q; one_to_one_likely_v2 adds the confident-half clause
    (predict_pid.py --assign auto uses v2; the file-check flags keep the first rule)."""
    S = np.asarray(S_f, dtype=float)
    n_games = np.asarray(n_games)
    Q, C = S.shape
    if Q == 0:
        return {"n_questions": 0, "n_candidates": int(C), "distinct_top1": 0, "expected_random": 0.0, "by_n": {}, "one_to_one_likely": None,
                "confident_distinct": None, "one_to_one_likely_v2": None}
    t1 = np.argmax(S, axis=1)
    distinct = len(set(t1.tolist()))
    return {"n_questions": int(Q), "n_candidates": int(C), "distinct_top1": int(distinct),
            "expected_random": float(C * (1 - (1 - 1 / C) ** Q)) if C > 0 else 0.0,
            "by_n": {str(n): len(set(t1[n_games == n].tolist())) for n in sorted(set(n_games.tolist()))},
            "one_to_one_likely": bool(distinct >= fraction * Q), "confident_distinct": (conf := confident_distinct(S)),
            # v2: the 87.5% rule, or no more questions than candidates and >= 95% distinct Top-1 among the more confident half (hard blocks)
            "one_to_one_likely_v2": bool(distinct >= fraction * Q or (Q <= C and conf >= CONFIDENT_DISTINCT))}


def num_games_counts(questions):
    c = Counter(int(q["declared_games"]) for q in questions)
    return {str(n): int(c[n]) for n in sorted(c)}


def record_properties(sgfs, game_ids):
    """SGF property identifiers (games containing each), game_id examples, whether the ids are all numeric, sorted and unique."""
    props = Counter()
    for sgf in sgfs.values():
        props.update(set(properties(sgf)))
    ids = list(game_ids)
    numeric = bool(ids) and all(str(g).isdigit() for g in ids)
    if not ids:
        sorted_flag = None
    elif numeric:
        vals = [int(g) for g in ids]
        sorted_flag = bool(all(a <= b for a, b in zip(vals, vals[1:])))
    else:
        sorted_flag = bool(all(str(a) <= str(b) for a, b in zip(ids, ids[1:])))
    return {"properties": dict(props), "game_id_examples": [str(g) for g in ids[:5]], "game_id_all_numeric": numeric, "game_id_sorted_in_file": sorted_flag,
            "game_id_unique": bool(len(set(map(str, ids))) == len(ids)), "n_game_ids": len(ids)}
