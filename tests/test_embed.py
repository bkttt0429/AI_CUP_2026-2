import numpy as np
import pytest

from aicup_pid.embed import PlayerEmbedding, score_embed


BASIS = np.linalg.qr(np.random.RandomState(99).randn(256, 8))[0].T  # players differ in one shared 8-dim subspace


def planted(rng, n_players=40, games=30, noise=1.0, spread=3.0):
    centres = (spread * rng.randn(n_players, 8) @ BASIS).astype(np.float32)
    return {f"p{i}": (centres[i] + noise * rng.randn(games, 256)).astype(np.float32) for i in range(n_players)}, centres


def test_embedding_learns_planted_players_and_scores_held_out_ones():
    rng = np.random.RandomState(0)
    train, _ = planted(rng)
    m = PlayerEmbedding(hidden=64, d_out=32, steps=300, players_per_episode=32, support=10, query_sizes=(2, 5), lr=3e-3, device="cpu").fit(train)
    assert m.players_trained == 40 and m.history[-1]["loss"] < m.history[0]["loss"]
    # unseen players: candidates and questions from the same centres
    rng2 = np.random.RandomState(1)
    test, centres = planted(rng2, n_players=10, games=25)
    store, by_player, q_keys, truth = {}, {}, [], []
    for i, (p, rows) in enumerate(test.items()):
        for j, r in enumerate(rows):
            store[f"{p}:{j}"] = r
        by_player[p] = [f"{p}:{j}" for j in range(20)]
        q_keys.append([f"{p}:{j}" for j in range(20, 25)])
        truth.append(i)
    S = score_embed(m, by_player, q_keys, store)
    assert S.shape == (10, 10) and np.isfinite(S).all()
    assert (S.argmax(axis=1) == np.array(truth)).mean() >= 0.8
    v = m.embed(test["p0"][:3])
    assert v.shape == (3, 32) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    m2 = PlayerEmbedding.from_state(m.state_dict(), device="cpu")
    assert np.allclose(score_embed(m2, by_player, q_keys, store), S, atol=1e-5)
    assert score_embed(m, by_player, [], store).shape == (0, 10)


def test_augmentation_rows_and_minimum_games():
    rng = np.random.RandomState(2)
    train, _ = planted(rng, n_players=6, games=12)
    aug = {p: r + 0.01 for p, r in train.items()}
    m = PlayerEmbedding(hidden=32, d_out=16, steps=20, players_per_episode=6, support=5, query_sizes=(2,), min_games=12, device="cpu").fit(train, aug)
    assert m.players_trained == 6
    with pytest.raises(ValueError):
        PlayerEmbedding(min_games=13, steps=1, device="cpu").fit(train)


def test_score_components_passes_the_embedding_matrix():
    from aicup_pid import components
    assert "emb" in components.OPTIONAL and components.ALL_COMPONENTS[-1] == "emb"
