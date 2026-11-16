import importlib.util
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
