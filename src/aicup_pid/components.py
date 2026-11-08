"""Build the per-candidate score matrices of a pool for the ladder's components (shared by pid_cv.py and predict_pid.py)."""
import multiprocessing as mp

import numpy as np

from aicup_pid.hd import HDModel
from aicup_pid.local import LocalModel
from aicup_pid.orient import OrientModel
from aicup_pid.ref import RefModel, cosine_scores
from aicup_pid.opening import OpeningModel, parse
from aicup_pid.style import StyleModel, game_features

COMPONENTS = ("opening", "style")  # the production fusion
OPTIONAL = ("local", "hd", "orient", "opening_raw", "ref", "trunk", "emb")  # lines B2 (fusion3), B5 (hd), B6 (orient, opening_raw), B7 (ref), B4 (trunk), B8 (emb): fitted only when a fusion document names them
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


def fit_hd(cand_keys_by_player, sgfs, cfg):
    """HDModel on the candidates' records (cfg keys hd_alpha, hd_beta); sgfs covers every candidate key."""
    model = HDModel(alpha=cfg.get("hd_alpha", 2.0), beta=cfg.get("hd_beta", 5.0))
    return model.fit({p: [sgfs[k] for k in keys] for p, keys in cand_keys_by_player.items()})


def score_hd(model, question_keys, sgfs):
    """[Q, C] handicap-habit log scores for questions given as lists of game keys."""
    if not question_keys:
        return np.zeros((0, len(model.players)))
    return np.stack([model.log_scores([sgfs[k] for k in keys]) for keys in question_keys])


def fit_orient(cand_keys_by_player, moves, cfg):
    """OrientModel on the candidates' parsed games (cfg keys orient_alpha, orient_beta, orient_depth, orient_raw)."""
    model = OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=cfg.get("orient_raw", 1))
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})


def fit_opening_raw(cand_keys_by_player, moves, cfg):
    """The opening back-off model on raw coordinates (no symmetry normalisation; same depth / alpha / beta / alpha_full)."""
    model = OpeningModel(depth=cfg.get("depth", 12), alpha=cfg.get("alpha", 2.0), beta=cfg.get("beta", 5.0), alpha_full=cfg.get("alpha_full", 1.0), symmetry=False)
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})


def fit_ref(ref_keys_by_player, moves, cfg):
    """RefModel on the reference players' parsed games (cfg keys ref_alpha, ref_beta, local_start, orient_depth, ref_orient_raw)."""
    model = RefModel(alpha=cfg.get("ref_alpha", 2.0), beta=cfg.get("ref_beta", 5.0), local_start=cfg.get("local_start", 12),
                     orient_depth=cfg.get("orient_depth", 12), orient_raw=cfg.get("ref_orient_raw", 2))
    return model.fit({p: [moves[k] for k in keys] for p, keys in ref_keys_by_player.items()})


def score_ref(model, cand_keys_by_player, question_keys, moves):
    """[Q, C] cosine between the questions' and the candidates' reference-affinity profiles (candidates in sorted id order,
    as the other models)."""
    players = sorted(cand_keys_by_player)
    C = np.stack([model.profile([moves[k] for k in cand_keys_by_player[p]]) for p in players])
    if not question_keys:
        return np.zeros((0, len(players)))
    Q = np.stack([model.profile([moves[k] for k in keys]) for keys in question_keys])
    return cosine_scores(Q, C)


def score_components(comps, opening, style, question_keys, moves, feats, local=None, hd=None, sgfs=None, orient=None, opening_raw=None, ref=None, trunk=None, emb=None):
    """{component: [Q, C]} for the named components: the two production ones, plus 'local' / 'orient' / 'opening_raw'
    (model given), 'hd' (model and the sgf strings given), 'ref' (the [Q, C] matrix of score_ref) and 'trunk' (the
    [Q, C] matrix of aicup_pid.trunkstyle.score_trunk) and 'emb' (aicup_pid.embed.score_embed); the last three are
    computed by the caller."""
    S = score_questions(opening, style, question_keys, moves, feats)
    if "ref" in comps:
        if ref is None:
            raise ValueError("the fusion names 'ref' but no reference-affinity matrix was given (score_ref)")
        S["ref"] = ref
    if "trunk" in comps:
        if trunk is None:
            raise ValueError("the fusion names 'trunk' but no trunk-style matrix was given (aicup_pid.trunkstyle.score_trunk)")
        S["trunk"] = trunk
    if "emb" in comps:
        if emb is None:
            raise ValueError("the fusion names 'emb' but no embedding matrix was given (aicup_pid.embed.score_embed)")
        S["emb"] = emb
    if "opening_raw" in comps:
        if opening_raw is None:
            raise ValueError("the fusion names 'opening_raw' but no raw opening model was fitted")
        if opening_raw.players != opening.players:
            raise ValueError("the raw opening model disagrees on the candidate order")
        S["opening_raw"] = score_local(opening_raw, question_keys, moves)
    if "orient" in comps:
        if orient is None:
            raise ValueError("the fusion names 'orient' but no orientation model was fitted")
        if orient.players != opening.players:
            raise ValueError("the orientation model disagrees on the candidate order")
        S["orient"] = score_local(orient, question_keys, moves)
    if "local" in comps:
        if local is None:
            raise ValueError("the fusion names 'local' but no local model was fitted")
        if local.players != opening.players:
            raise ValueError("the local model disagrees on the candidate order")
        S["local"] = score_local(local, question_keys, moves)
    if "hd" in comps:
        if hd is None or sgfs is None:
            raise ValueError("the fusion names 'hd' but no hd model (or no sgf strings) was given")
        if hd.players != opening.players:
            raise ValueError("the hd model disagrees on the candidate order")
        S["hd"] = score_hd(hd, question_keys, sgfs)
    return {c: S[c] for c in comps}

