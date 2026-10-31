#!/usr/bin/env python
"""Are the rank test's players training players? Identify a question's player among all training players (AI_CUP_2026-1 issue #22).

Candidates: every training player, one opening model per colour (aicup_pid.opening.OpeningModel(color=...)) fitted on the
player's training games of that colour minus the games held out as queries. Queries: pseudo-questions of one player
(5-20 games with colours) from --questions CSVs (AI_CUP_2026-1 splits format: question_id, rank, player_id, n_games,
game_keys joined by ';') or rebuilt from scored folds (--scores DIR --calib-fold K --eval-fold K: aicup.pilot.pilot_questions
over aicup.l1bcv.players_from_scores, the fold-0 set behind results/l1b/fold0_fused_opp_proba.csv). A question's score per
candidate is the sum over its games of the colour-matched model's log likelihood; margin = top-1 minus top-2.

The calibration questions choose the margin threshold tau: the lowest margin such that the questions at or above it are
identified with precision >= --precision. The evaluation questions are scored once: Top-1 / Top-5 by question size, the official
rank score of the top-1 candidate's rank (id_rank) and, with --fused-proba, the combined decision (margin >= tau -> the identified
player's rank, else fused_opp_avg's expected-score decision) against fused_opp alone, with a player bootstrap. --siblings: per
question size, a small model per evaluation question; how often the most similar other question of that size belongs to the same
player (the pooling fallback if the test players are not training players). --test-csv: official rank-test files instead of
pseudo-questions, scored against all training games: per question the top-1 player, its rank and margin, and the share at or
above --tau. Fold 0 is evaluated once, in the role the user chose.

  cd ~/AICUP_2026-1 && python -I ../AICUP_2026-2/scripts/pid_rank_overlap.py --scores results/scores --calib-fold 1 --eval-fold 0 \\
      --fused-proba results/l1b/fold0_fused_opp_proba.csv --siblings --workers 24 --out ../AICUP_2026-2/results/rank_overlap.json
"""
import argparse
import multiprocessing as mp
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
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import to_json  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX, expected_score_decision, score as official_score  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.submission import read_test  # noqa: E402
from aicup_pid.opening import OpeningModel, parse_prefix  # noqa: E402
from aicup_pid.paths import FOLDS_CSV  # noqa: E402
from aicup_pid.provenance import pid_environment  # noqa: E402

COLORS = ("B", "W")
_MODELS = {}  # set before forking the scoring workers


def load_training(train_dir, depth, workers=1):
    """player_id, rank, game_key, color, moves (the first `depth` moves) of every training game."""
    frames = []
    for p in sorted(Path(train_dir).glob("train_*.csv")):
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False)
        df["game_key"] = df["rank"] + ":" + df["game_id"]
        frames.append(df[["player_id", "rank", "game_key", "color", "sgf_content"]])
    games = pd.concat(frames, ignore_index=True)
    sgfs = games["sgf_content"].tolist()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            moves = pool.starmap(parse_prefix, ((s, depth) for s in sgfs), chunksize=2048)
    else:
        moves = [parse_prefix(s, depth) for s in sgfs]
    games = games.drop(columns=["sgf_content"])
    games["moves"] = moves
    return games


def fold_of(questions, default=None):
    """The fold named by the question ids' 'f<k>_' prefix when they all agree, else `default`."""
    folds = set(questions["question_id"].str.extract(r"^f(\d+)_", expand=False).dropna())
    return int(next(iter(folds))) if len(folds) == 1 else default


def read_questions(paths):
    """The splits-format question CSVs as one table with game_keys as lists."""
    frames = []
    for p in paths:
        q = pd.read_csv(p, dtype=str, keep_default_na=False)
        q["game_keys"] = q["game_keys"].str.split(";")
        q["n_games"] = q["n_games"].astype(int)
        frames.append(q[["question_id", "rank", "player_id", "n_games", "game_keys"]])
    return pd.concat(frames, ignore_index=True)


def questions_from_scores(scores_dir, fold, folds_df):
    """The pilot questions (1 x 20, 2 x 10, 4 x 5 per player) of one scored fold, as the fused checks use them."""
    hsl = read_scored_games(Path(scores_dir) / f"fold{fold}")
    return pilot_questions(players_from_scores(hsl, folds_df))


def fit_candidates(games, held_out, depth, players):
    """{colour: OpeningModel over `players`} on the training games not in held_out."""
    keep = games[~games["game_key"].isin(held_out)]
    models = {}
    for c in COLORS:
        sub = keep[keep["color"] == c]
        by_player = {p: [] for p in players}
        for pid, mv in zip(sub["player_id"], sub["moves"]):
            by_player[pid].append(mv)
        models[c] = OpeningModel(depth=depth, color=c).fit(by_player)
    return models


def _score_game(args):
    color, moves = args
    return _MODELS[color].log_scores([moves]).astype(np.float32)


def score_games(models, games, workers=1):
    """[(colour, moves)] -> [G, C] log likelihoods under the colour-matched candidate models."""
    global _MODELS
    _MODELS = models
    if not games:
        return np.zeros((0, len(next(iter(models.values())).players)), dtype=np.float32)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_score_game, games, chunksize=16)
    else:
        rows = [_score_game(g) for g in games]
    return np.stack(rows)


def question_scores(models, questions, game_of, workers=1):
    """[Q, C]: per question the sum of its games' log likelihoods. game_of: game_key -> (colour, moves)."""
    keys = sorted({k for ks in questions["game_keys"] for k in ks})
    G = score_games(models, [game_of[k] for k in keys], workers)
    row = {k: i for i, k in enumerate(keys)}
    return np.stack([G[[row[k] for k in ks]].sum(axis=0) for ks in questions["game_keys"]]) if len(questions) else np.zeros((0, G.shape[1]))


def identify(S, players):
    """top1 index, top5 indices, margin (top-1 minus top-2 log likelihood) per question."""
    order = np.argsort(-S, axis=1)
    top1, top5 = order[:, 0], order[:, :5]
    margin = S[np.arange(len(S)), top1] - S[np.arange(len(S)), order[:, 1]] if S.shape[1] > 1 else np.full(len(S), np.inf)
    return top1, top5, margin


def choose_tau(margin, correct, precision):
    """The lowest margin at which the questions with margin >= tau are identified with precision >= `precision`
    (the largest such prefix of the questions sorted by margin); +inf when no prefix qualifies. Also the curve."""
    order = np.argsort(-margin)
    hits = np.cumsum(correct[order])
    k = np.arange(1, len(order) + 1)
    prec = hits / k
    ok = np.flatnonzero(prec >= precision)
    tau = float(margin[order[ok[-1]]]) if len(ok) else float("inf")
    curve = [{"tau": float(margin[order[i]]), "precision": float(prec[i]), "recall": float(hits[i] / max(correct.sum(), 1)), "questions": int(k[i])}
             for i in sorted(set(np.linspace(0, len(order) - 1, num=min(25, len(order))).astype(int)))] if len(order) else []
    return tau, curve


def by_size(values, n, sizes=SIZES):
    return {str(m): float(np.mean(values[n == m])) for m in sizes if np.any(n == m)}


def sizes_present(n):
    """The official question sizes that occur in n (the balanced means skip absent sizes instead of averaging NaN)."""
    return tuple(m for m in SIZES if np.any(n == m)) or SIZES


def evaluate_questions(S, questions, players, rank_of, tau):
    """Identification metrics of one question set (true players known)."""
    top1, top5, margin = identify(S, players)
    truth = np.array([players.index(p) if p in players else -1 for p in questions["player_id"]])
    n = questions["n_games"].to_numpy()
    y = questions["rank"].map(RANK_TO_IDX).to_numpy()
    correct = top1 == truth
    in5 = np.any(top5 == truth[:, None], axis=1)
    id_rank = np.array([rank_of[players[i]] for i in top1])
    id_score = np.array([official_score(p, t) for p, t in zip(id_rank, y)])
    above = margin >= tau
    out = {"questions": int(len(S)), "top1": float(correct.mean()), "top5": float(in5.mean()), "top1_by_n": by_size(correct, n), "top5_by_n": by_size(in5, n),
           "id_rank_score": balanced_mean_over_n(id_score, y, n, sizes_present(n)), "id_rank_by_n": by_size(id_score, n),
           "coverage_at_tau": float(above.mean()), "precision_at_tau": float(correct[above].mean()) if above.any() else None,
           "coverage_by_n": by_size(above.astype(float), n), "margin_quantiles_correct": _quantiles(margin[correct]), "margin_quantiles_wrong": _quantiles(margin[~correct])}
    return out, {"top1": top1, "margin": margin, "correct": correct, "id_rank": id_rank, "y": y, "n": n}


def _quantiles(x):
    return {q: float(np.quantile(x, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")} if len(x) else {}


def combine(questions, ident, proba, tau, n_boot):
    """Combined decision against fused_opp alone on the questions present in the proba table (player bootstrap)."""
    key = questions["question_id"].str.replace(r"^f\d+_", "", regex=True)
    pt = proba.set_index("question_id")
    have = key.isin(pt.index).to_numpy()
    if not have.any():
        raise ValueError("no evaluation question is in the --fused-proba table")
    prefix = next((pre for pre in ("fused_opp_avg_p_", "fused_opp_p_") if f"{pre}{LEVELS[0]}" in pt.columns), None)
    if prefix is None:
        raise ValueError("--fused-proba needs fused_opp_avg_p_<rank> (fold0_fused_opp_proba.csv) or fused_opp_p_<rank> (oof_fused_opp_proba.csv) columns")
    P = pt.loc[key[have], [f"{prefix}{lv}" for lv in LEVELS]].to_numpy(dtype=float)
    y, n = ident["y"][have], ident["n"][have]
    fused = np.array([expected_score_decision(p) for p in P])
    use_id = ident["margin"][have] >= tau
    comb = np.where(use_id, ident["id_rank"][have], fused)
    s_f = np.array([official_score(p, t) for p, t in zip(fused, y)])
    s_c = np.array([official_score(p, t) for p, t in zip(comb, y)])
    pid = questions["player_id"].to_numpy(dtype=str)[have]
    sizes = sizes_present(n)
    d = paired_diff(s_c, s_f, y, pid, n, sizes=sizes, n=n_boot)
    return {"questions": int(have.sum()), "fused_opp": balanced_mean_over_n(s_f, y, n, sizes), "combined": balanced_mean_over_n(s_c, y, n, sizes),
            "fused_opp_by_n": by_size(s_f, n), "combined_by_n": by_size(s_c, n), "share_identified": float(use_id.mean()),
            "combined_vs_fused_opp": d, "pass": bool(d["ci95"][0] > 0 and d["diff"] >= 0.005)}


def siblings(questions, game_of, depth, workers=1):
    """Per question size: a small model per question; the most similar other question of that size is the same player's."""
    out = {}
    for m in SIZES:
        qs = questions[questions["n_games"] == m].reset_index(drop=True)
        has_sib = qs["player_id"].map(qs["player_id"].value_counts()) > 1
        if len(qs) < 2 or not has_sib.any():
            continue
        models = {}
        for c in COLORS:
            models[c] = OpeningModel(depth=depth, color=c).fit({q: [game_of[k][1] for k in ks if game_of[k][0] == c] for q, ks in zip(qs["question_id"], qs["game_keys"])})
        S = question_scores(models, qs, game_of, workers)
        np.fill_diagonal(S, -np.inf)
        top1, _, margin = identify(S, list(qs["question_id"]))
        same = (qs["player_id"].to_numpy()[top1] == qs["player_id"].to_numpy()) & has_sib.to_numpy()
        out[str(m)] = {"questions": int(len(qs)), "with_sibling": int(has_sib.sum()), "top1_same_player": float(same[has_sib].mean()),
                       "margin_quantiles_same": _quantiles(margin[has_sib & same]), "margin_quantiles_other": _quantiles(margin[has_sib & ~same])}
    return out


def test_mode(a, games, models, players, rank_of):
    rows = []
    for path in a.test_csv:
        for q in read_test(path):
            moves = [(c, parse_prefix(s, a.depth)) for s, c in q["games"]]
            rows.append({"question_id": q["question_id"], "n_games": len(moves), "games": moves})
    if not rows:
        raise ValueError("no questions in --test-csv")
    flat = [g for r in rows for g in r["games"]]
    G = score_games(models, flat, a.workers)
    S, i = [], 0
    for r in rows:
        S.append(G[i:i + len(r["games"])].sum(axis=0))
        i += len(r["games"])
    S = np.stack(S)
    top1, top5, margin = identify(S, players)
    table = pd.DataFrame({"question_id": [r["question_id"] for r in rows], "n_games": [r["n_games"] for r in rows],
                          "top1_player": [players[j] for j in top1], "top1_rank": [LEVELS[rank_of[players[j]]] for j in top1],
                          "margin": margin, "above_tau": margin >= a.tau, "top5_players": [";".join(players[j] for j in row) for row in top5]})
    n = table["n_games"].to_numpy()
    summary = {"questions": int(len(table)), "tau": a.tau, "share_above_tau": float(table["above_tau"].mean()),
               "share_above_tau_by_n": {str(m): float(table["above_tau"][n == m].mean()) for m in sorted(set(n.tolist()))},
               "margin_quantiles": _quantiles(margin), "distinct_top1_players": int(table["top1_player"].nunique())}
    print(f"test questions {len(table)}: margin >= tau {a.tau:.2f} in {summary['share_above_tau']:.3f}; margin quantiles {summary['margin_quantiles']}")
    return table, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", default=str(aicup_pid.TRAIN_DIR))
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--questions", nargs="*", default=[], help="splits-format question CSVs used for calibration (first) and evaluation (second)")
    ap.add_argument("--scores", default=None, help="AI_CUP_2026-1 results/scores: rebuild the pilot questions of --calib-fold and --eval-fold")
    ap.add_argument("--calib-fold", type=int, default=1)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--fused-proba", default=None, help="fold0_fused_opp_proba.csv of the evaluation fold (combined decision)")
    ap.add_argument("--siblings", action="store_true", help="also the same-player question matching on the evaluation questions")
    ap.add_argument("--test-csv", nargs="*", default=[], help="official rank-test CSVs: score their questions against all training players")
    ap.add_argument("--tau", type=float, default=None, help="margin threshold for --test-csv (from a previous run's json)")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--precision", type=float, default=0.98)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-table", default=None, help="--test-csv: per-question CSV")
    a = ap.parse_args(argv)
    t0 = time.time()
    folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
    try:
        games = load_training(a.train_dir, a.depth, a.workers)
        players = sorted(games["player_id"].unique())
        rank_of = dict(zip(games["player_id"], games["rank"].map(RANK_TO_IDX)))
        print(f"training games {len(games)}, players {len(players)} ({time.time() - t0:.0f} s)", flush=True)
        if a.test_csv:
            if a.tau is None:
                raise ValueError("--test-csv needs --tau")
            models = fit_candidates(games, set(), a.depth, players)
            table, summary = test_mode(a, games, models, players, rank_of)
            if a.out_table:
                Path(a.out_table).parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(a.out_table, index=False)
            out = {"mode": "test", "test_csv": a.test_csv, "candidates": {"players": len(players), "games": int(len(games))}, "depth": a.depth, **summary, **pid_environment("cpu")}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(to_json(out, indent=1) + "\n")
            print(f"wrote {a.out}")
            return 0
        if a.scores:
            calib = questions_from_scores(a.scores, a.calib_fold, folds_df)
            evalq = questions_from_scores(a.scores, a.eval_fold, folds_df)
            calib_fold, eval_fold = a.calib_fold, a.eval_fold
        elif len(a.questions) == 2:
            calib, evalq = read_questions([a.questions[0]]), read_questions([a.questions[1]])
            calib_fold, eval_fold = fold_of(calib), fold_of(evalq)
        else:
            raise ValueError("give --scores with --calib-fold/--eval-fold, or exactly two --questions CSVs (calibration, evaluation)")
        if set(calib["player_id"]) & set(evalq["player_id"]):
            raise ValueError("calibration and evaluation questions share players")
        held_out = {k for q in (calib, evalq) for ks in q["game_keys"] for k in ks}
        game_of = {k: (c, m) for k, c, m in zip(games["game_key"], games["color"], games["moves"]) if k in held_out}
        missing = held_out - set(game_of)
        if missing:
            raise ValueError(f"{len(missing)} question games are not in the training files, e.g. {sorted(missing)[:3]}")
        models = fit_candidates(games, held_out, a.depth, players)
        print(f"candidate models fitted without {len(held_out)} held-out games ({time.time() - t0:.0f} s)", flush=True)
        S_c = question_scores(models, calib, game_of, a.workers)
        res_c, ident_c = evaluate_questions(S_c, calib, players, rank_of, np.inf)
        tau, curve = choose_tau(ident_c["margin"], ident_c["correct"], a.precision)
        res_c.update({"tau": tau, "precision_target": a.precision, "curve": curve})
        print(f"calibration (fold {calib_fold}): top1 {res_c['top1']:.3f} top5 {res_c['top5']:.3f} by n {res_c['top1_by_n']}; tau {tau:.2f} ({time.time() - t0:.0f} s)", flush=True)
        S_e = question_scores(models, evalq, game_of, a.workers)
        res_e, ident_e = evaluate_questions(S_e, evalq, players, rank_of, tau)
        print(f"evaluation (fold {eval_fold}): top1 {res_e['top1']:.3f} top5 {res_e['top5']:.3f} by n {res_e['top1_by_n']}; id_rank {res_e['id_rank_score']:.4f}; "
              f"margin >= tau in {res_e['coverage_at_tau']:.3f} with precision {res_e['precision_at_tau']}", flush=True)
        out = {"candidates": {"players": len(players), "games": int(len(games)), "held_out_games": len(held_out)}, "depth": a.depth,
               "calibration": {"fold": calib_fold, **res_c}, "evaluation": {"fold": eval_fold, **res_e}}
        if a.fused_proba:
            proba = pd.read_csv(a.fused_proba, dtype={"question_id": str, "player_id": str})
            out["combined"] = combine(evalq, ident_e, proba, tau, a.n_boot)
            c = out["combined"]
            print(f"combined {c['combined']:.4f} vs fused_opp {c['fused_opp']:.4f}: {c['combined_vs_fused_opp']['diff']:+.4f} "
                  f"[{c['combined_vs_fused_opp']['ci95'][0]:+.4f}, {c['combined_vs_fused_opp']['ci95'][1]:+.4f}] on {c['questions']} questions, "
                  f"identified share {c['share_identified']:.3f}", flush=True)
        if a.siblings:
            out["siblings"] = siblings(evalq, game_of, a.depth, a.workers)
            for m, r in out["siblings"].items():
                print(f"siblings n={m}: top1 same player {r['top1_same_player']:.3f} over {r['with_sibling']} questions with a sibling", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    out.update({"seconds": time.time() - t0, **pid_environment("cpu")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
