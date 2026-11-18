import numpy as np

from aicup_pid.tactical_duration import advance_streak, duration_tables, duration_probabilities, log_scores


def test_streak_is_history_only_and_pass_resets():
    assert advance_streak(None, (3, 3), 0) == 0
    assert advance_streak((3, 3), (4, 5), 0) == 1
    assert advance_streak((4, 5), (5, 6), 1) == 2
    assert advance_streak((5, 6), (6, 7), 2) == 2
    assert advance_streak((5, 6), (9, 5), 2) == 0
    assert advance_streak((5, 6), None, 2) == 0


def test_streak_control_keeps_all_five_context_rates():
    n = 180
    c = np.arange(n) % 5
    s = (np.arange(n) // 5) % 3
    y = s == 2
    phase = np.arange(n) // 60
    v, cp, sp = duration_tables(c, s, y, phase, 21)
    np.testing.assert_array_equal(v.sum((0, 1)), cp.sum((0, 1)))
    np.testing.assert_array_equal(v.sum(1), sp.sum(1))
    np.testing.assert_array_equal(v[:, :, 0], sp[:, :, 0])
    assert not np.array_equal(v, sp)
    other, _, _ = duration_tables(c, s, ~y, phase, 22)
    p = duration_probabilities(v, v + other)
    b = duration_probabilities(other, v + other)
    assert log_scores(v, p)[2] > log_scores(v, b)[2]
    assert np.isfinite(log_scores(np.zeros_like(v), p)).all()
