#!/usr/bin/env python
"""Stress test: harder candidate pools (same-rank bands; the 400 players with the most similar opening repertoires).

All eligible players of --folds (>= 105 Black games) are split as make_pool does (seed, 100 candidate games, the rest
cut into 5/10/15/20-game questions). Pools are subsets of these players: 'random' (the reference, --n-players at random),
one pool per --bands entry (every eligible player of those ranks, at most --n-players), and 'nearest' (greedy: start from the
two players whose opening repertoires -- L2-normalised frequencies of the canonical first --nearest-k Black moves of their
candidate games -- are most similar, keep adding the player most similar on average to the members). Each pool is scored
with the production models and the FIXED fusion weights of --fusion / --fusion3 (no refit): opening, style, local, fusion,
fusion3, by n. Report only; nothing is selected.

  python -I scripts/pid_hard_pools.py --folds 1 2 3 4 --workers 12 --out results/hard/hard_pools.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # noqa: E402  (AI_CUP_2026-1 on sys.path)
from aicup.ladder import to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup_pid import checks, pools  # noqa: E402
from aicup_pid.components import features_all, fit_from_parsed, fit_local, parse_all, score_components  # noqa: E402
from aicup_pid.fusion import Fusion, topk  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

THREE = ("opening", "style", "local")
DEFAULT_BANDS = ["D,C,B", "A,1D,2D", "3D,4D", "5D,6D"]


def repertoire_similarity(players, by_player, moves, k):
    """[P, P] cosine similarity of the players' opening-repertoire frequency vectors (first k Black moves, canonical)."""
    counts = [Counter(checks.black_prefix(moves[key], k) for key in by_player[p]) for p in players]
    vocab = {tok: i for i, tok in enumerate(sorted({t for c in counts for t in c}))}
    M = np.zeros((len(players), len(vocab)))
    for i, c in enumerate(counts):
        for tok, n in c.items():
            M[i, vocab[tok]] = n
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M /= norms
    return M @ M.T


def nearest_pool(sim, players, n):
    """Greedy: the most similar pair, then repeatedly the player most similar on average to the members."""
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    members = [int(i), int(j)]
    while len(members) < min(n, len(players)):
        mean_sim = sim[:, members].mean(axis=1)
        mean_sim[members] = -np.inf
        members.append(int(np.argmax(mean_sim)))
    return [players[m] for m in members]


def score_pool(name, member_ids, cands, qs, moves, feats, cfg, w2, w3, sim, players, log=print):
    t0 = time.time()
    members = set(member_ids)
    c = cands[cands["player_id"].isin(members)]
    q = qs[qs["player_id"].isin(members)].reset_index(drop=True)
    by_player = c.groupby("player_id")["game_key"].apply(list).to_dict()
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)
    local = fit_local(by_player, moves, cfg)
    q_keys = [list(k) for k in q["game_keys"]]
    S = score_components(THREE, opening, style, q_keys, moves, feats, local)
    fus2, fus3 = Fusion(), Fusion()
    fus2.w, fus3.w = np.asarray(w2, float), np.asarray(w3, float)
    S["fusion"] = fus2.scores([S["opening"], S["style"]])
    S["fusion3"] = fus3.scores([S[c_] for c_ in THREE])
    truth = [p for p in q["player_id"]]
    scores = {}
    for arm, M in S.items():
        tops = [[opening.players[j] for j in row] for row in topk(M, pools.TOP_K)]
        scores[arm] = pools.summarise(tops, truth, q["n_games"].to_numpy())
    idx = [players.index(p) for p in opening.players]
    sub = sim[np.ix_(idx, idx)]
    off = sub[~np.eye(len(idx), dtype=bool)]
    out = {"name": name, "players": len(by_player), "questions": int(len(q)), "questions_by_n": {str(m): int((q["n_games"] == m).sum()) for m in pools.SIZES},
           "ranks": {lv: int((c.drop_duplicates("player_id")["rank"] == lv).sum()) for lv in LEVELS},
           "mean_pairwise_similarity": float(off.mean()) if len(off) else None, "scores": scores, "seconds": time.time() - t0}
    log(f"{name}: {out['players']} players, {out['questions']} questions, mean similarity {out['mean_pairwise_similarity']:.3f}; "
        + "  ".join(f"{arm} {scores[arm]['score']:.4f}" for arm in ("opening", "style", "local", "fusion", "fusion3"))
        + f"  fusion3 n=5 {scores['fusion3']['by_n'].get('5', {}).get('score', float('nan')):.3f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands", nargs="*", default=DEFAULT_BANDS, help="rank bands, e.g. D,C,B A,1D,2D")
    ap.add_argument("--nearest", type=int, default=pools.N_CANDIDATES, help="size of the most-similar-repertoire pool (0 = skip)")
    ap.add_argument("--nearest-k", type=int, default=4)
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion.json"))
    ap.add_argument("--fusion3", default=str(ROOT / "results" / "pid_fusion3.json"))
    ap.add_argument("--max-players", type=int, default=0, help="smoke tests: only the first N eligible players")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        if 0 in a.folds:
            raise ValueError("fold 0 is kept for the final check")
        doc2, doc3 = json.loads(Path(a.fusion).read_text()), json.loads(Path(a.fusion3).read_text())
        if list(doc2["components"]) != ["opening", "style"] or list(doc3["components"]) != list(THREE):
            raise ValueError("--fusion must be the two-component document and --fusion3 the three-component one")
        cfg = {**doc3.get("cfg", {})}
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        games = pools.black_games(aicup_pid.TRAIN_DIR)
        elig = pools.eligible_players(games, folds_df, a.folds, a.cand_games + min(pools.SIZES))
        n_all = len(elig) if not a.max_players else min(a.max_players, len(elig))
        cands, qs, info = pools.make_pool(games, folds_df, a.folds, n_players=n_all, cand_games=a.cand_games, seed=a.seed, pool_id="hard")
        players = sorted(cands["player_id"].unique())
        keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})
        print(f"{len(players)} eligible players of folds {a.folds}, {len(qs)} questions; loading {len(keys)} games", flush=True)
        sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}
        moves, feats = parse_all(sgfs, a.workers), features_all(sgfs, a.workers)
        by_all = cands.groupby("player_id")["game_key"].apply(list).to_dict()
        sim = repertoire_similarity(players, by_all, moves, a.nearest_k)
        rank_of = dict(zip(cands["player_id"], cands["rank"]))
        rng = np.random.RandomState(a.seed)
        pool_sets = [("random", [players[i] for i in np.sort(rng.permutation(len(players))[:a.n_players])])]
        for band in a.bands:
            ranks = [r.strip() for r in band.split(",") if r.strip()]
            unknown = [r for r in ranks if r not in LEVELS]
            if unknown:
                raise ValueError(f"unknown ranks {unknown} in band {band!r}")
            members = [p for p in players if rank_of[p] in ranks]
            if len(members) > a.n_players:
                members = [members[i] for i in np.sort(rng.permutation(len(members))[:a.n_players])]
            if len(members) >= 2:
                pool_sets.append((f"band:{band}", members))
        if a.nearest and len(players) >= 2:
            pool_sets.append((f"nearest:{min(a.nearest, len(players))}", nearest_pool(sim, players, a.nearest)))
        results = [score_pool(name, members, cands, qs, moves, feats, cfg, doc2["weights"], doc3["weights"], sim, players) for name, members in pool_sets]
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    out = {"pools": results, "eligible": int(len(elig)), "players_used": len(players), "folds": a.folds, "seed": a.seed, "cand_games": a.cand_games,
           "nearest_k": a.nearest_k, "fusion": doc2["weights"], "fusion3": doc3["weights"], "cfg": cfg, "seconds": time.time() - t0, **pid_environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
