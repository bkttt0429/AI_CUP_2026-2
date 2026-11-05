"""Trunk-style component (fork plan line B4, trunk variant; METHOD_NOTES 2.21).

scripts/pid_trunk_extract.py stores one 256-dimensional row per game: mean and standard deviation of the humanv0 trunk's
board-pooled and played-point vectors (each projected to 64 dimensions) over the target player's positions. The component
is the style LDA of aicup_pid.style (shared shrunk within-player covariance, Gaussian log-likelihood of the question's
mean row) on those rows instead of the 32 hand-made statistics. `trunk_feats` picks the columns: all (256), mean (the two
means, 128), pooled (128), local (played point, 128).
"""
import glob

import numpy as np

from aicup_pid.style import StyleModel

COLUMNS = {"all": np.arange(256), "mean": np.r_[0:64, 128:192], "pooled": np.arange(0, 128), "local": np.arange(128, 256)}


def load_store(*dirs):
    """{game key: [256] float32} from every part file of the given directories."""
    store = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f, allow_pickle=False)
            for k, row in zip(z["keys"], z["feats"]):
                store[str(k)] = row
    return store


def rows_of(store, keys, cols):
    missing = [k for k in keys if k not in store]
    if missing:
        raise KeyError(f"{len(missing)} games have no trunk row, e.g. {missing[:3]}")
    return np.stack([store[k][cols] for k in keys]).astype(np.float64)


def fit_trunk(cand_keys_by_player, store, cfg):
    """StyleModel on the candidates' trunk rows (cfg keys trunk_feats, trunk_shrink)."""
    cols = COLUMNS[cfg.get("trunk_feats", "all")]
    model = StyleModel(shrink=cfg.get("trunk_shrink", 0.1)).fit({p: rows_of(store, keys, cols) for p, keys in cand_keys_by_player.items()})
    model.columns = cols
    return model


def score_trunk(model, question_keys, store):
    """[Q, C] trunk-style log scores for questions given as lists of game keys."""
    if not question_keys:
        return np.zeros((0, len(model.players)))
    return np.stack([model.log_scores(rows_of(store, keys, model.columns)) for keys in question_keys])
