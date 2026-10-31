#!/usr/bin/env python
"""Is the simulated-pool score of the opening model real? A leak audit of pid_cv.py's pools (report only, nothing is selected).

Per pool, the same make_pool as pid_cv.py: (1) duplicates: query games whose whole move sequence, up to the 8 board
symmetries, also appears among the same player's candidates, among other players' candidates, or among the same player's
other query games; (2) the opening and style scores with those query games removed; (3) the opening score by depth (how
many of the first moves it reads); (4) negative control: candidate games shuffled between players must give chance level
(1 + e^-1 + ... + e^-4) / 400 = 0.0039; (5) how often a query game's first 12 moves appear in the same player's candidates
against a random other candidate's. Fold 0 is refused.

  python -I scripts/pid_audit.py --pools 1,2 3,4 --workers 12 --out results/pid_audit.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import pools  # noqa: E402
from aicup_pid.components import features_all, parse_all, score_questions  # noqa: E402
from aicup_pid.opening import OpeningModel, canonical  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.style import StyleModel  # noqa: E402
from pid_cv import evaluate, parse_pools  # noqa: E402

DEPTHS = (1, 2, 4, 6, 8, 12, 20, 30)


def whole_game(moves):
    return tuple(canonical(moves, len(moves))[0])


def head(moves, depth=12):
    return tuple(canonical(moves, depth)[0][:depth])


def score(model, qs_keys, moves, qs, players, truth):
    S = np.stack([model.log_scores([moves[k] for k in keys]) for keys in qs_keys])
    return evaluate(S, players, qs, truth)[0]


def audit_pool(games, folds_df, folds, a):
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed,
                                      pool_id="p" + "".join(map(str, folds)))
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
    moves = parse_all(sgfs, a.workers)
    feats = features_all(sgfs, a.workers)
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()
    whole = {k: whole_game(moves[k]) for k in keys}
    cand_owner = {}
    for p, ks in by_player.items():
        for k in ks:
            cand_owner.setdefault(whole[k], set()).add(p)
    q_keys = [list(k) for k in qs["game_keys"]]
    n_query = same = other = 0
    drop = set()
    for p, ks in zip(qs["player_id"], q_keys):
        for k in ks:
            n_query += 1
            owners = cand_owner.get(whole[k], set())
            same += p in owners
            other += bool(owners - {p})
            if owners:
                drop.add(k)
    per_player = {}
    for p, ks in zip(qs["player_id"], q_keys):
        per_player.setdefault(p, []).extend(ks)
    q_repeat = sum(len(ks) - len({whole[k] for k in ks}) for ks in per_player.values())
    uniq_cands = len({whole[k] for k in cands["game_key"]})
    # the two models as pid_cv.py fits them
    opening = OpeningModel(depth=a.depth).fit({p: [moves[k] for k in ks] for p, ks in by_player.items()})
    style = StyleModel().fit({p: np.stack([feats[k] for k in ks]) for p, ks in by_player.items()})
    players = opening.players
    truth = np.array([opening.index[p] for p in qs["player_id"]])
    # (2) without the duplicated query games
    kept = [[k for k in ks if k not in drop] for ks in q_keys]
    ok = np.array([len(ks) > 0 for ks in kept])
    S = score_questions(opening, style, [ks for ks, o in zip(kept, ok) if o], moves, feats)
    qs_ok = qs[ok].reset_index(drop=True)
    nodup = {c: evaluate(S[c], players, qs_ok, truth[ok])[0] for c in S}
    # (3) by depth
    by_depth = {}
    for d in DEPTHS:
        m = OpeningModel(depth=d).fit({p: [moves[k] for k in ks] for p, ks in by_player.items()})
        by_depth[str(d)] = score(m, q_keys, moves, qs, players, truth)
        print(f"pool {folds} depth {d}: score {by_depth[str(d)]['score']:.4f} top1 {by_depth[str(d)]['top1']:.3f}", flush=True)
    # (4) negative control: the same games, candidates shuffled between players
    rng = np.random.RandomState(a.seed)
    all_keys = cands["game_key"].to_numpy(dtype=str)[rng.permutation(len(cands))]
    shuffled = {p: list(all_keys[i * a.cand_games:(i + 1) * a.cand_games]) for i, p in enumerate(players)}
    control = score(OpeningModel(depth=a.depth).fit({p: [moves[k] for k in ks] for p, ks in shuffled.items()}), q_keys, moves, qs, players, truth)
    # (5) first 12 moves seen in the player's own candidates vs a random other candidate's
    heads = {p: {head(moves[k]) for k in ks} for p, ks in by_player.items()}
    hit_own = hit_other = n = 0
    for p, ks in zip(qs["player_id"], q_keys):
        others = [o for o in players if o != p]
        o = others[rng.randint(len(others))]
        for k in ks:
            h = head(moves[k])
            hit_own += h in heads[p]
            hit_other += h in heads[o]
            n += 1
    out = {"folds": list(folds), "pool": info, "questions": int(len(qs)), "query_games": n_query, "unique_candidate_games": uniq_cands, "candidate_games": int(len(cands)),
           "query_dup_in_own_candidates": int(same), "query_dup_in_other_candidates": int(other), "query_repeats_within_player": int(q_repeat),
           "without_duplicates": nodup, "questions_without_duplicates": int(ok.sum()), "opening_by_depth": by_depth, "negative_control": control,
           "first12_in_own_candidates": hit_own / max(n, 1), "first12_in_random_other_candidates": hit_other / max(n, 1)}
    print(f"pool {folds}: {n_query} query games; duplicates in own candidates {same}, other candidates {other}, within player {q_repeat}; "
          f"candidates unique {uniq_cands}/{len(cands)}", flush=True)
    print(f"pool {folds}: without duplicates opening {nodup['opening']['score']:.4f} style {nodup['style']['score']:.4f}; shuffled candidates {control['score']:.4f}; "
          f"first 12 moves in own candidates {out['first12_in_own_candidates']:.3f}, random other {out['first12_in_random_other_candidates']:.3f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    specs = parse_pools(a.pools)
    if any(0 in f for f in specs):
        ap.error("fold 0 is kept for the final check")
    folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
    games = pools.black_games(aicup_pid.TRAIN_DIR)
    res = [audit_pool(games, folds_df, f, a) for f in specs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json({"pools": res, "seed": a.seed, "depth": a.depth, "n_players": a.n_players, "cand_games": a.cand_games,
                                    **environment("cpu")}, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
