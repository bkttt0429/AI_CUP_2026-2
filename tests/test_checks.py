import numpy as np

from aicup.sgf import parse_moves
from aicup_pid import checks, opening
from aicup_pid.components import features_all, fit_from_parsed, fit_models, parse_all
from aicup_pid.dedup import build_training_index, game_hash, hash_games
from tests.synth import black_table, habit_of, random_game, sgf_lookup, sgf_of, training_frame
from tests.test_dedup import _apply_symmetry, _write_training


def test_fit_from_parsed_matches_fit_models():
    frame, _ = training_frame(folds=(1,), players_per_fold=4, games_per_player=12, seed=1)
    sgfs = {k: s for k, (s, c) in sgf_lookup(frame).items() if c == "B"}
    by_player = black_table(frame).groupby("player_id")["game_key"].apply(list).to_dict()
    o1, s1, moves, feats = fit_models(by_player, sgfs, {"depth": 8})
    o2, s2 = fit_from_parsed(by_player, moves, feats, {"depth": 8})
    assert o1.players == o2.players and o1.state() == o2.state() and np.allclose(s1.means, s2.means)


def test_overlap_summary_maps_candidates_to_training_players(tmp_path):
    frame, _ = training_frame(folds=(1,), players_per_fold=3, games_per_player=10, seed=3)
    _write_training(frame, tmp_path)
    idx = build_training_index(tmp_path, workers=1, log=lambda *a, **k: None)
    black = frame[frame["color"] == "B"]
    a = black["player_id"].iloc[0]
    games_a = black[black["player_id"] == a]["sgf_content"].tolist()
    mirrored = sgf_of(_apply_symmetry(parse_moves(games_a[1]), 5))
    novel = random_game(np.random.RandomState(99), habit_of(7))
    hashes = {k: v[0] for k, v in hash_games({"c1": games_a[0], "c2": mirrored, "c3": novel}).items()}
    dups = checks.duplicates_against_index(hashes, idx)
    assert set(dups) == {"c1", "c2"} and dups["c2"][0]["player_id"] == a and {"player_id", "rank", "game_key"} <= set(dups["c2"][0])
    s = checks.summarise_overlap(dups, {"c1": "pub_p_A", "c2": "pub_p_A", "c3": "pub_p_B"})
    assert s["n_games_checked"] == 3 and s["n_dup_games"] == 2 and s["n_players_with_dups"] == 1
    assert s["players_single_training_id"] == 1 and s["players_multiple_training_ids"] == 0 and s["same_id_players"] == 0
    assert s["by_player"] == {"pub_p_A": {a: 2}}
    assert checks.summarise_overlap(dups, {"c1": a, "c2": a, "c3": "x"})["same_id_players"] == 1


def test_duplicates_between_and_within():
    h1, h2, h3 = b"1" * 16, b"2" * 16, b"3" * 16
    assert checks.duplicates_between({"q1": h1, "q2": h2, "q3": None}, {"c1": h2, "c2": h3, "c3": None}) == [("q2", "c1")]
    assert checks.duplicates_within({"a": h1, "b": h1, "c": None, "d": h2}) == [("a", "b")]


def test_length_quantiles_and_legality():
    q = checks.length_quantiles(np.array([10, 20, 30, 40]), np.array([5, 10, 15, 20]))
    assert q["n"] == 4 and q["moves"]["p50"] == 25.0 and q["black_moves"]["mean"] == 12.5 and set(q["moves"]) == {"p05", "p25", "p50", "p75", "p95", "mean"}
    assert checks.length_quantiles(np.array([]), np.array([]))["n"] == 0
    leg = checks.legality({"ok": sgf_of([("B", 3, 3), ("W", 15, 15)]), "bad": sgf_of([("W", 3, 3), ("B", 15, 15)]), "junk": "(;SZ[9];B[aa])"})
    assert leg["n"] == 3 and leg["n_bad"] == 2 and {e["key"] for e in leg["errors"]} == {"bad", "junk"}
    assert leg["handicap_counts"] == {"none": 3} and leg["usable_keys"] == ["ok"]


def _habit_games(habits, per_habit, seed):
    rng = np.random.RandomState(seed)
    return [parse_moves(random_game(rng, habit_of(h))) for h in habits for _ in range(per_habit)]


def test_opening_shift_flags_a_different_habit_set():
    cands = _habit_games(range(8), 30, 0)
    same = _habit_games(range(8), 10, 1)
    other = _habit_games(range(8, 16), 10, 2)  # a different second corner for every player
    r_same = checks.opening_shift(same, cands, k=2, n_null=20, seed=0)
    r_other = checks.opening_shift(other, cands, k=2, n_null=20, seed=0)
    assert r_same["k"] == 2 and r_same["n_query"] == 80 and r_same["n_candidate"] == 240 and len(r_same["null"]) == 20
    assert r_same["flag"] is False and r_same["observed"] <= 1.5 * r_same["null_max"]
    assert r_other["flag"] is True and r_other["observed"] > 1.5 * r_other["null_max"]
    assert 0.0 <= checks.js_divergence({"a": 1}, {"a": 1}) < 1e-12 and abs(checks.js_divergence({"a": 1}, {"b": 1}) - 1.0) < 1e-12


def test_self_identification_recovers_planted_habits_and_skips_short_candidates():
    frame, _ = training_frame(folds=(1,), players_per_fold=6, games_per_player=60, seed=4)  # 30 Black games each
    sgfs = {k: s for k, (s, c) in sgf_lookup(frame).items() if c == "B"}
    by_player = black_table(frame).groupby("player_id")["game_key"].apply(list).to_dict()
    short = sorted(by_player)[0]
    by_player[short] = by_player[short][:12]
    moves, feats = parse_all(sgfs), features_all(sgfs)
    r = checks.self_identification(by_player, moves, feats, {"depth": 8}, [1.0, 0.5], seed=0, model_games=10, sizes=(5, 10))
    assert r["n_candidates"] == 5 and r["n_skipped"] == 1 and r["n_questions"] == 10
    assert set(r["scores"]) == {"opening", "style", "fusion"} and r["scores"]["fusion"]["by_n"].keys() == {"5", "10"}
    assert r["scores"]["fusion"]["top1"] >= 0.5
    assert {"p05", "p25", "p50", "p75", "p95", "mean"} <= set(r["fusion_margin_quantiles"]) and 0 <= r["candidates_misidentified"] <= 5


def test_top1_collisions():
    S = np.eye(6)[[0, 1, 2, 2, 3, 3]]
    r = checks.top1_collisions(S, np.array([5, 5, 10, 10, 20, 20]))
    assert r["n_questions"] == 6 and r["n_candidates"] == 6 and r["distinct_top1"] == 4
    assert r["by_n"] == {"5": 2, "10": 1, "20": 1} and r["one_to_one_likely"] is False
    assert abs(r["expected_random"] - 6 * (1 - (1 - 1 / 6) ** 6)) < 1e-9
    assert checks.top1_collisions(np.eye(6), np.full(6, 5))["one_to_one_likely"] is True
    assert checks.top1_collisions(np.zeros((0, 6)), np.array([]))["distinct_top1"] == 0


def test_num_games_and_record_properties():
    assert checks.num_games_counts([{"declared_games": 5}, {"declared_games": 5}, {"declared_games": 20}]) == {"5": 2, "20": 1}
    p = checks.record_properties({"a": "(;SZ[19]HD[2];B[pd];W[dp])", "b": "(;SZ[19];B[pd])"}, ["g_000002", "g_000001"])
    assert p["properties"] == {"SZ": 2, "HD": 1, "B": 2, "W": 1} and p["game_id_all_numeric"] is False and p["game_id_sorted_in_file"] is False
    assert p["game_id_examples"] == ["g_000002", "g_000001"]
    assert checks.record_properties({}, ["10", "9", "100"])["game_id_sorted_in_file"] is False
    assert checks.record_properties({}, ["9", "10", "100"])["game_id_sorted_in_file"] is True
    assert checks.record_properties({}, [])["game_id_sorted_in_file"] is None
    assert checks.record_properties({}, ["a", "a", "b"])["game_id_unique"] is False and p["game_id_unique"] is True and p["n_game_ids"] == 2
