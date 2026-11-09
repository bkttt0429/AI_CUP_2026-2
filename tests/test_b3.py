import numpy as np
import pandas as pd
import pytest

from aicup_pid.b3 import RANK_COLUMNS, chunk_slices, rank_inputs, score_b3


def test_chunk_slices_cover_the_games_in_near_equal_chunks():
    assert [len(c) for c in chunk_slices(100)] == [20] * 5
    assert [len(c) for c in chunk_slices(45)] == [15, 15, 15]
    assert [len(c) for c in chunk_slices(7)] == [7]
    assert chunk_slices(0) == []
    assert np.concatenate(chunk_slices(101)).tolist() == list(range(101)) and max(len(c) for c in chunk_slices(101)) <= 20


def planted(rng, n_players=30, chunks=5, n_q=30, dims=32, passes=3, noise=0.7):
    centres = rng.randn(n_players, dims)
    players = [f"pub_p_{i:04d}" for i in range(n_players)]
    ids, rows, chunks_by = [], [], {}
    for i, p in enumerate(players):
        for j in range(chunks):
            ids.append(f"pub_c{i + 1:04d}_k{j}")
            chunks_by.setdefault(p, []).append(ids[-1])
    truth = rng.randint(0, n_players, n_q)
    qids = [f"pub_q_{k:04d}" for k in range(n_q)]
    ids += qids
    base = np.concatenate([np.repeat(centres, chunks, axis=0), centres[truth]])
    Z = np.stack([base + noise * rng.randn(*base.shape) for _ in range(passes)])
    return Z, {q: r for r, q in enumerate(ids)}, chunks_by, players, qids, truth


def test_score_b3_finds_planted_players_and_averages_passes():
    Z, row, chunks_by, players, qids, truth = planted(np.random.RandomState(0))
    S, info = score_b3(Z, row, chunks_by, players, qids)
    assert S.shape == (len(qids), len(players)) and info == {"missing_questions": [], "missing_candidates": [], "passes": 3}
    assert (S.argmax(axis=1) == truth).mean() > 0.9
    one = np.mean([score_b3(Z[k:k + 1], row, chunks_by, players, qids)[0] for k in range(3)], axis=0)
    assert np.allclose(S, one)


def test_score_b3_missing_rows_are_neutral():
    Z, row, chunks_by, players, qids, truth = planted(np.random.RandomState(1))
    gone_q, gone_p = qids[3], players[5]
    row = {k: v for k, v in row.items() if k != gone_q and k not in chunks_by[gone_p]}
    S, info = score_b3(Z, row, chunks_by, players, qids)
    assert info["missing_questions"] == [gone_q] and info["missing_candidates"] == [gone_p]
    assert np.allclose(S[3], 0) and np.allclose(S[:, 5], 0) and np.isfinite(S).all()
    with pytest.raises(ValueError):
        score_b3(Z, {}, chunks_by, players, qids)


def test_rank_inputs_chunks_each_candidate_in_game_id_order():
    rows = []
    for p, n in (("pub_p_0002", 45), ("pub_p_0001", 100)):
        for g in range(n):
            rows.append({"player_id": p, "game_id": f"pub_g_{p[-1]}{n - g:04d}", "sgf_content": f"(;B[{p}-{n - g}])", "color": "B"})
    df, mp = rank_inputs(pd.DataFrame(rows).sample(frac=1, random_state=0), "pub")
    assert list(df.columns) == RANK_COLUMNS
    assert df.question_id.tolist() == [f"pub_c0001_k{j}" for j in range(5)] + [f"pub_c0002_k{j}" for j in range(3)]
    assert df.num_games.tolist() == ["20"] * 5 + ["15"] * 3
    assert mp.player_id.tolist() == ["pub_p_0001"] * 5 + ["pub_p_0002"] * 3
    first = df.iloc[0]
    assert first.sgf_1 == "(;B[pub_p_0001-1])" and first.sgf_20 == "(;B[pub_p_0001-20])" and first.color_20 == "B"
    assert df.iloc[5].sgf_16 == "" and mp.game_ids.str.split("|").map(len).sum() == 145
