import numpy as np

from aicup_pid.assign import assign_topk, assignment_report
from aicup_pid.fusion import topk


def test_assign_fixes_a_collision_and_keeps_row_order():
    S = np.array([[0.9, 0.8, 0.0], [0.95, 0.1, 0.0], [0.0, 0.0, 1.0]])
    assert topk(S, 3)[:, 0].tolist() == [0, 0, 2]  # two questions claim candidate 0
    out = assign_topk(S, 3)
    assert out.tolist() == [[1, 0, 2], [0, 1, 2], [2, 0, 1]]  # the assignment maximises 0.8 + 0.95 + 1.0
    rep = assignment_report(S, out)
    assert rep["n_questions"] == 3 and rep["n_top1_changed"] == 1 and rep["distinct_top1_before"] == 2 and rep["distinct_top1_after"] == 3


def test_assign_more_questions_than_candidates_keeps_argmax_for_the_rest():
    S = np.array([[1.0, 0.0], [0.9, 0.2], [0.1, 0.8]])
    out = assign_topk(S, 5)
    assert out.shape == (3, 2) and out[:, 0].tolist() == [0, 0, 1]  # rows 0 and 2 assigned, row 1 keeps its argmax


def test_assign_edge_cases():
    assert assign_topk(np.zeros((0, 4)), 5).shape == (0, 4)
    S = np.eye(4)
    assert assign_topk(S, 5).tolist() == topk(S, 5).tolist()  # already one-to-one: nothing changes
    assert assignment_report(S, assign_topk(S, 5))["n_top1_changed"] == 0
