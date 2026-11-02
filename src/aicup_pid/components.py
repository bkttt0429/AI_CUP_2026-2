"""Build the per-candidate score matrices of a pool for the ladder's components (shared by pid_cv.py and predict_pid.py)."""
import multiprocessing as mp

import numpy as np

from aicup_pid.local import LocalModel
from aicup_pid.opening import OpeningModel, parse
from aicup_pid.style import StyleModel, game_features

COMPONENTS = ("opening", "style")  # the production fusion
OPTIONAL = ("local",)  # line B2 (fusion3): fitted only when a fusion document names it
ALL_COMPONENTS = (*COMPONENTS, *OPTIONAL)


def _features(sgf):
    return game_features(parse(sgf))


def parse_all(sgfs, workers=1):
    """{game_key: parsed moves}."""
    keys = list(sgfs)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            moves = pool.map(parse, [sgfs[k] for k in keys], chunksize=256)
    else:
        moves = [parse(sgfs[k]) for k in keys]
    return dict(zip(keys, moves))


def features_all(sgfs, workers=1):
    """{game_key: style feature row}."""
    keys = list(sgfs)
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_features, [sgfs[k] for k in keys], chunksize=64)
    else:
        rows = [_features(sgfs[k]) for k in keys]
    return dict(zip(keys, rows))


def fit_from_parsed(cand_keys_by_player, moves, feats, cfg):
    """(OpeningModel, StyleModel) fitted on already parsed games: moves / feats cover every key of cand_keys_by_player."""
    opening = OpeningModel(depth=cfg.get("depth", 12), alpha=cfg.get("alpha", 2.0), beta=cfg.get("beta", 5.0), alpha_full=cfg.get("alpha_full", 1.0))
    opening.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})
    style = StyleModel(shrink=cfg.get("shrink", 0.1))
    style.fit({p: np.stack([feats[k] for k in keys]) for p, keys in cand_keys_by_player.items()})
    if opening.players != style.players:
        raise ValueError("the two models disagree on the candidate order")
    return opening, style


def fit_models(cand_keys_by_player, sgfs, cfg, workers=1):
    """(OpeningModel, StyleModel, parsed moves, feature rows) fitted on the candidates' games; sgfs covers every key used."""
    moves = parse_all(sgfs, workers)
    feats = features_all(sgfs, workers)
    opening, style = fit_from_parsed(cand_keys_by_player, moves, feats, cfg)
    return opening, style, moves, feats


def score_questions(opening, style, question_keys, moves, feats):
    """{component: [Q, C]} for questions given as lists of game keys."""
    S_o = np.stack([opening.log_scores([moves[k] for k in keys]) for keys in question_keys]) if question_keys else np.zeros((0, len(opening.players)))
    S_s = np.stack([style.log_scores(np.stack([feats[k] for k in keys])) for keys in question_keys]) if question_keys else np.zeros((0, len(style.players)))
    return {"opening": S_o, "style": S_s}


def fit_local(cand_keys_by_player, moves, cfg):
    """LocalModel on the candidates' parsed games (cfg keys local_alpha, local_beta, local_start)."""
    model = LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12))
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})


def score_local(model, question_keys, moves):
    """[Q, C] local-shape log scores for questions given as lists of game keys."""
    if not question_keys:
        return np.zeros((0, len(model.players)))
    return np.stack([model.log_scores([moves[k] for k in keys]) for keys in question_keys])


def score_components(comps, opening, style, question_keys, moves, feats, local=None):
    """{component: [Q, C]} for the named components (the two production ones plus 'local' when a model is given)."""
    S = score_questions(opening, style, question_keys, moves, feats)
    if "local" in comps:
        if local is None:
            raise ValueError("the fusion names 'local' but no local model was fitted")
        if local.players != opening.players:
            raise ValueError("the local model disagrees on the candidate order")
        S["local"] = score_local(local, question_keys, moves)
    return {c: S[c] for c in comps}

