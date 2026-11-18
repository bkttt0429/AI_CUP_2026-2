import numpy as np
import torch

from aicup_pid.local_choice import (
    ConditionalChoice, action_basis, local_geometry, phase_permutation,
)


def empty_position():
    planes = torch.zeros(1, 22, 19, 19)
    planes[:, 0] = 1
    return planes


def test_board_geometry_comes_only_from_pre_move_planes():
    planes = empty_position()
    planes[0, 2, 9, 9] = planes[0, 9, 9, 9] = 1
    old_p = torch.ones(1, 362) / 362
    shape, previous, relative, context = local_geometry(planes, old_p)
    assert shape.shape == (1, 362, 45)
    assert previous.shape == (1, 45)
    torch.testing.assert_close(previous[0], shape[0, 180])
    assert shape[0, 0, 30:].sum() > shape[0, 180, 30:].sum()
    assert relative[0, 180, 0] > relative[0, 0, 0]
    assert shape[0, 361].count_nonzero() == relative[0, 361].count_nonzero() == 0
    no_previous = local_geometry(empty_position(), old_p)
    assert no_previous[1].count_nonzero() == no_previous[2].count_nonzero() == 0
    assert (context != no_previous[3]).any()
    prob2 = torch.zeros_like(old_p)
    prob2[:, 0] = 1
    assert not torch.equal(context, local_geometry(planes, prob2)[3])


def test_action_projection_preserves_exact_reference_residual_and_rotation():
    planes = empty_position()
    planes[0, 2, 2, 5] = planes[0, 9, 2, 5] = planes[0, 1, 9, 14] = 1
    p0 = torch.ones(1, 362) / 362
    geom = local_geometry(planes, p0)
    rotated = local_geometry(planes.rot90(1, (-2, -1)), p0)
    for col in (0, 2):
        restored = geom[col][:, :361].reshape(1, 19, 19, -1).rot90(1, (1, 2)).reshape(1, 361, -1)
        torch.testing.assert_close(rotated[col][:, :361], restored)
    torch.testing.assert_close(rotated[3], geom[3])
    trunk = torch.randn(1, 362, 8)
    x = action_basis(trunk, *geom[:3], products=True)
    additive = action_basis(trunk, *geom[:3], products=False)
    assert x.shape == additive.shape and not torch.equal(x, additive)
    w = torch.nn.Linear(x.size(-1), 5, bias=False)
    expected = (x * p0.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(w(x[:, 100] - expected), w(x[:, 100]) - (w(x) * p0.unsqueeze(-1)).sum(1))


def test_zero_memory_and_scoring_pass_do_not_reveal_the_action():
    model = ConditionalChoice(5, 8, hidden=4)
    ctx = torch.randn(2, 5)
    features = torch.randn(2, 4, 8)
    base = torch.tensor([[-2., -1., -torch.inf, -3.], [-1., -3., -torch.inf, -2.]]).log_softmax(-1)
    keys = torch.randn(3, 2, 4, 5)
    vals = torch.zeros(3, 2, 4, 8)
    logp, null = model(ctx, features, base, keys, vals)
    torch.testing.assert_close(logp, base[:, None].expand(-1, 3, -1))
    vals.normal_()
    first, a = model(ctx, features, base, keys, vals)
    first[..., 0].exp()
    second, b = model(ctx, features, base, keys, vals)
    second[..., -1].exp()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(a, b)
    assert torch.isneginf(first[..., 2]).all() and first[..., -1].isfinite().all()
    torch.testing.assert_close(first.exp().sum(-1), torch.ones_like(null))


def test_replaying_a_different_recorded_query_move_leaves_every_prediction_unchanged():
    from pid_choice_data_audit import replay_choices_checked, te
    te.init_worker({"version": 15})
    histories = [replay_choices_checked("example", "(;SZ[19];B[dd];W[pp];B[" + move + "])", "B")
                 for move in ("qq", "")]
    assert histories[0]["played"][-1] != histories[1]["played"][-1] == 361
    contexts, local = [], []
    for h in histories:
        allowed = torch.from_numpy(h["legal"][-1:])
        baseline = torch.zeros(1, 362).masked_fill(~allowed, -torch.inf).log_softmax(-1)
        shape, previous, relative, key = local_geometry(torch.from_numpy(h["bins"][-1:]), baseline.exp())
        contexts.append(key)
        local.append(action_basis(torch.zeros(1, 362, 1), shape, previous, relative, products=True))
    for name in ("bins", "globs", "legal"):
        np.testing.assert_array_equal(histories[0][name][-1], histories[1][name][-1])
    torch.testing.assert_close(contexts[0], contexts[1])
    model = ConditionalChoice(contexts[0].size(-1), local[0].size(-1), hidden=4)
    keys = torch.randn(2, 1, 3, contexts[0].size(-1))
    vals = torch.randn(2, 1, 3, local[0].size(-1))
    pred, weight = model(contexts[0], local[0], baseline, keys, vals)
    alt, weight2 = model(contexts[1], local[1], baseline, keys, vals)
    torch.testing.assert_close(pred, alt)
    torch.testing.assert_close(weight, weight2)


def test_phase_permutation_keeps_each_game_and_stratum_intact():
    t = np.array([[[0, 3, 7, 60, 61, 70, 120, 220], [2, 6, 8, 52, 56, 80, 111, 240]]])
    order = phase_permutation(t, seed=405)
    assert order.shape == t.shape
    for phase, ix in zip(t[0], order[0]):
        assert sorted(ix) == list(range(len(phase)))
        assert np.array_equal(phase // 32, phase[ix] // 32)
        for cell in np.unique(phase // 32):
            idx = np.where(phase // 32 == cell)[0]
            if len(idx) > 1:
                assert (ix[idx] != idx).all()


def test_sampled_replay_matches_audited_replay_including_capture_and_pass():
    from pid_reference_policy import replay, te
    from pid_local_choice_features import replay_positions
    te.init_worker({"version": 15})
    # The corner white stone is captured on ply 3. Later passes span the sampling grid.
    game = {"key": "capture", "color": "B", "sgf": "(;SZ[19];B[ba];W[aa];B[ab];W[qp]" + ";B[];W[]" * 39 + ")"}
    original = replay(game)
    sampled = replay_positions(game)
    for name in sampled:
        np.testing.assert_array_equal(sampled[name], original[name])


def test_matching_context_resolves_identical_marginal_choices():
    torch.manual_seed(2)
    model = ConditionalChoice(2, 2, hidden=4)
    query = torch.eye(2)
    features = torch.eye(2).expand(2, -1, -1)
    base = torch.full((2, 2), -np.log(2), dtype=torch.float32)
    refctx = query.expand(2, 1, -1, -1)
    residual = (torch.eye(2) - .5)
    values = torch.stack([residual, -residual])[:, None]
    opt = torch.optim.Adam(model.parameters(), lr=.04)
    for _ in range(140):
        lp, _ = model(query, features, base, refctx, values)
        target = torch.stack([lp[0, 0, 0], lp[1, 0, 1], lp[0, 1, 1], lp[1, 1, 0]])
        loss = -target.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert target.exp().min() > .8
    uniform = ConditionalChoice(2, 2, hidden=4, uniform=True)
    lp, _ = uniform(query, features, base, refctx, values)
    torch.testing.assert_close(lp[:, 0], lp[:, 1])
