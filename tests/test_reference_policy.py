import importlib
import torch


def fixture():
    torch.manual_seed(4)
    m = importlib.import_module("aicup_pid.reference_policy")
    model = m.ReferencePolicy(3, 4, hidden=8)
    return m, model, (torch.randn(5, 3), torch.randn(5, 6, 4),
        torch.tensor([[0., -1., -2., -float("inf"), -1., -3.]]).expand(5, -1),
        torch.randn(2, 3, 7, 3), torch.randn(2, 3, 7, 4))


def test_legal_normalizer_and_zero_memory_fallback():
    _, model, args = fixture()
    lp, null = model(*args)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(5, 2), atol=1e-6)
    assert (lp.exp()[:, :, 3] == 0).all()
    zero, _ = model(*args[:-1], torch.zeros_like(args[-1]))
    assert torch.allclose(zero, torch.log_softmax(args[2], -1)[:, None].expand(-1, 2, -1))
    assert ((null >= 0) & (null <= 1)).all()


def test_query_actions_cannot_change_memory_alignment():
    _, model, args = fixture()
    lp, null = model(*args)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    changed, null2 = model(args[0], args[1][:, perm], args[2][:, perm], *args[3:])
    assert torch.allclose(null, null2, atol=1e-7)
    assert torch.allclose(lp[:, :, perm], changed, atol=1e-6)


def test_reference_candidate_and_game_order_are_equivariant():
    _, model, args = fixture()
    lp, _ = model(*args)
    other, _ = model(*args[:3], args[3].flip(0).flip(1), args[4].flip(0).flip(1))
    assert torch.allclose(lp.flip(1), other, atol=1e-6)


def test_learns_opposite_conditional_preferences_with_equal_marginals():
    m = importlib.import_module("aicup_pid.reference_policy")
    torch.manual_seed(17); torch.set_num_threads(1)
    model = m.ReferencePolicy(2, 2, hidden=12)
    context = torch.eye(2).repeat(2, 1)
    actions = torch.tensor([[[1., 0.], [-1., 0.]]]).expand(4, -1, -1)
    refs = torch.eye(2)[None, None].expand(2, 1, -1, -1)
    values = torch.tensor([[[[1., 0.], [-1., 0.]]], [[[-1., 0.], [1., 0.]]]])
    played, owners = torch.tensor([0, 1, 1, 0]), torch.tensor([0, 0, 1, 1])
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(250):
        lp, _ = model(context, actions, torch.zeros(4, 2), refs, values)
        loss = m.episodic_loss(lp, played, owners)
        opt.zero_grad(); loss.backward(); opt.step()
    lp, _ = model(context, actions, torch.zeros(4, 2), refs, values)
    assert (lp.exp()[torch.arange(4), owners, played] > .8).all()
    model.mode = "uniform"
    uniform, _ = model(context, actions, torch.zeros(4, 2), refs, values)
    assert torch.allclose(uniform[:, 0], uniform[:, 1], atol=1e-6)
