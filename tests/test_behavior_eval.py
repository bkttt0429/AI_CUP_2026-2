import importlib.util
import json
import numpy as np
import pytest
from behavior_data import immutable_json, validate_features


def runner():
    assert importlib.util.find_spec('pid_behavior_train'), 'training/evaluation runner not implemented'
    import pid_behavior_train as r
    return r


def feature_fixture():
    lp = np.full((2, 362), -np.inf, np.float32); lp[:, [0, 361]] = np.log(.5)
    phi = np.zeros((2, 362, 385), np.float16); phi[:, 361, -1] = 1
    return dict(phi=phi, logp=lp, state=np.zeros((2, 388)), history=np.zeros((2, 8, 6)),
                ref_value=np.zeros((2, 386)), played=np.array([0, 361]), t=np.array([2, 8]))


def test_feature_validation_rejects_bad_pass_and_illegal_label():
    a = feature_fixture(); validate_features(a)
    a['played'][0] = 1
    with pytest.raises(ValueError): validate_features(a)
    a = feature_fixture(); a['phi'][0, 361, -1] = 0
    with pytest.raises(ValueError): validate_features(a)


def test_immutable_protocol_does_not_overwrite(tmp_path):
    p = tmp_path/'protocol.json'; immutable_json(p, {'split': ['train', 'test']})
    old = p.read_bytes(); immutable_json(p, {'split': ['train', 'test']})
    with pytest.raises(ValueError): immutable_json(p, {'split': ['test', 'train']})
    assert p.read_bytes() == old


def test_true_candidate_score_and_pairing_from_hand_fixture():
    r = runner()
    # Two players, two query games each: each row contains game-mean log p.
    scores = np.array([[-1., -3.], [-4., -2.], [-4., -1.], [-2., -1.]])
    out = r.player_metrics(scores, np.array([0, 0, 1, 1]))
    assert np.allclose(out['nll'], [2.5, 1.])
    assert np.allclose(out['margin'], [0., 2.])
    assert np.allclose(out['top1'], [.5, 1.])


def test_bootstrap_keeps_seeds_paired_and_gate_requires_all_mechanisms():
    r = runner()
    assert r.paired_ci(np.ones((2, 20))) == {'mean': 1., 'ci95': [1., 1.]}
    assert r.paired_ci(np.stack([np.ones(20), -np.ones(20)])) == {'mean': 0., 'ci95': [0., 0.]}
    positive = {'mean': .1, 'ci95': [.01, .2]}
    assert r.mechanism_verdict(positive, positive, [.1, .1], [.1, .1]) == 'mechanism_supported_in_pilot'
    assert r.mechanism_verdict(positive, positive, [.1, -.1], [.1, .1]) == 'personalization_without_context_binding'
    assert r.mechanism_verdict({'mean': 0., 'ci95': [-.1, .1]}, positive, [.1, .1], [.1, .1]) == 'not_supported'


def test_toy_result_serializes_numpy_derived_gate():
    r = runner()
    result = r.toy_report([True]*8 + [False]*2, np.array([.1]*10))
    assert json.loads(json.dumps(result))['passed'] is True
    assert result['accuracy'] == .8
    assert r.toy_report([False]*10, np.array([.1]*10))['passed'] is False


def test_extraction_samples_beyond_old_256_ply_boundary():
    from pid_behavior_extract import init_worker, replay
    init_worker({'version': 15})
    # A legal long all-pass fixture isolates full-game traversal without captures.
    sgf = '(;GM[1]FF[4]SZ[19]' + ';B[];W[]'*150 + ')'
    r = replay({'sgf': sgf, 'color': 'W'})
    assert len(r['t']) == 16 and r['t'][-1] == 299
    assert (r['played'] == 361).all()


def test_budget_counts_failures_and_reserves_running_jobs():
    assert importlib.util.find_spec('pid_behavior_submit'), 'bounded launcher missing'
    from pid_behavior_submit import available_seconds
    # Failed allocations are charged; a running allocation reserves its full limit.
    assert available_seconds([('FAILED', 12, 300), ('COMPLETED', 48, 300)]) == 3540
    assert available_seconds([('FAILED', 12, 300), ('RUNNING', 10, 1200)]) == 2388
    with pytest.raises(ValueError): available_seconds([('UNKNOWN', 0, 0)])
