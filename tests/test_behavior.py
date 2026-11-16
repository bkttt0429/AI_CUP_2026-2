import importlib.util
import numpy as np
import pytest
import torch


def modules():
    assert importlib.util.find_spec('behavior_model'), 'conditional behavior model not implemented'
    assert importlib.util.find_spec('behavior_data'), 'behavior data boundaries not implemented'
    import behavior_model as m
    import behavior_data as d
    return m, d


def inputs():
    torch.manual_seed(8)
    q, c, mem, actions = 3, 2, 5, 7
    lp = torch.randn(q, actions)
    lp[:, 1] = -torch.inf
    lp = lp.log_softmax(-1)
    return [torch.randn(q, actions, 385), lp, torch.randn(q, 388), torch.randn(q, 8, 6),
            torch.randn(c, mem, 388), torch.randn(c, mem, 8, 6), torch.randn(c, mem, 386),
            torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)]


@pytest.mark.parametrize('mode', ['background', 'global', 'contextual'])
def test_legal_mask_pass_and_finite_gradient(mode):
    m, _ = modules()
    net = m.BehaviorPolicy(mode)
    x = inputs()
    out = net(*x)
    assert out.shape == (3, 2, 7)
    assert (out[:, :, 1] == -torch.inf).all()
    assert torch.isfinite(out[:, :, -1]).all()  # pass survives
    assert torch.allclose(out.exp().sum(-1), torch.ones(3, 2), atol=1e-6)
    (-out[:, :, -1].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_reference_permutation_and_padding_do_not_change_prediction():
    m, _ = modules()
    net = m.BehaviorPolicy('contextual').eval()
    x = inputs(); expected = net(*x)
    order = torch.tensor([4, 2, 0, 3, 1])
    y = x[:4] + [z[:, order] for z in x[4:]]
    assert torch.allclose(net(*y), expected, atol=2e-6)
    y = [z.clone() for z in x]
    for j in (4, 5, 6):
        y[j][~x[7]] = 10000
    assert torch.allclose(net(*y), expected, atol=2e-6)


def test_action_permutation_is_equivariant():
    m, _ = modules(); net = m.BehaviorPolicy('contextual').eval(); x = inputs()
    order = torch.tensor([6, 3, 2, 1, 0, 5, 4])
    y = [x[0][:, order], x[1][:, order]] + x[2:]
    assert torch.allclose(net(*y), net(*x)[:, :, order], atol=2e-6)


def test_background_cannot_use_reference_identity_and_no_actions_really_removes_actions():
    m, _ = modules(); x = inputs(); y = [z.clone() for z in x]; y[6] *= -5
    bg = m.BehaviorPolicy('background').eval()
    assert torch.equal(bg(*x), bg(*y))
    assert torch.equal(bg(*x)[:, 0], bg(*x)[:, 1])
    net = m.BehaviorPolicy('contextual').eval()
    assert torch.equal(net(*x, ablation='no_actions'), net(*y, ablation='no_actions'))
    assert not torch.allclose(net(*x), net(*y), atol=1e-7)


def test_attention_permutation_keeps_global_summary():
    m, _ = modules(); x = inputs(); net = m.BehaviorPolicy('contextual').eval()
    global_before, _, _ = net.reference(*x[4:])
    global_after, _, _ = net.reference(*x[4:], ablation='permute')
    assert torch.equal(global_before, global_after)
    net2 = m.BehaviorPolicy('global').eval()
    assert torch.equal(net2(*x), net2(*x, ablation='permute'))


def test_empty_reference_and_impossible_query_are_rejected():
    m, _ = modules(); x = inputs(); net = m.BehaviorPolicy('contextual')
    y = x.copy(); y[-1] = torch.zeros_like(x[-1])
    with pytest.raises(ValueError): net(*y)
    y = x.copy(); y[1] = torch.full_like(x[1], -torch.inf)
    with pytest.raises(ValueError): net(*y)


def test_prefix_excludes_current_and_future_action():
    _, d = modules()
    moves = [('B', 0, 0), ('W', None, None), ('B', 18, 18)]
    h = d.prefix_history(moves, 2, 'B')
    assert h.shape == (8, 6)
    assert np.array_equal(h[-2], [0, 0, 0, 0, 1, 1])
    assert np.array_equal(h[-1], [0, 0, 1, 1, 0, 1])
    assert np.array_equal(h, d.prefix_history(moves[:2] + [('W', 4, 5)] * 20, 2, 'B'))
    assert not d.prefix_history(moves, 0, 'B').any()


def test_player_split_and_episode_game_boundaries():
    _, d = modules()
    rows = [{'owner': str(i), 'key': f'{i}:{j}', 'sgf_sha256': f'{i}:{j}', 'fold': 3,
             'pool': 'calibration'} for i in range(100) for j in range(10)]
    split = d.player_split(rows)
    assert sorted(map(len, split.values())) == [16, 20, 64]
    assert len(set().union(*map(set, split.values()))) == 100
    assert split == d.player_split(list(reversed(rows)))
    groups = {p: [f'{p}:{j}' for j in range(10)] for p in split['train']}
    owners, refs, queries = d.episode_games(groups, np.random.default_rng(0))
    assert len(owners) == len(set(owners)) == 16
    assert all(len(r) == 3 and q not in r and len(set(r)) == 3 for r, q in zip(refs, queries))
    # Any repeated SGF is rejected, including duplicates under different owners.
    rows[-1]['sgf_sha256'] = rows[0]['sgf_sha256']
    with pytest.raises(ValueError): d.player_split(rows)


def test_short_game_position_sampling_and_zero_length():
    _, d = modules()
    assert d.position_indices(3).tolist() == [0, 1, 2]
    ix = d.position_indices(100)
    assert len(set(ix)) == 16 and ix[0] == 0 and ix[-1] == 99
    with pytest.raises(ValueError): d.position_indices(0)
