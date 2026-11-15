"""The 11-08 v2 release's added pools (METHOD_NOTES 2.29): 600 candidates with 10 games each, questions of 1 / 2 / 3 / 5 games,
the player Black or White in any of them, several questions per player possible.

A game is seen from its target player: when the player is White the colours are swapped, so every model's "Black" is the
player. Opening lines and raw-coordinate habits differ between moving first and second, so those models are fitted per view
(the player's Black games / White games) and a question's score is the sum, over its games, of the log score under the model
of that game's view; a candidate without a game in a view scores the background there. Style statistics, local shapes and the
learnt embedding use all of a player's games together. Every function returns [Q, C] with the candidates in sorted id order.
"""
import numpy as np

from aicup_pid.hd import HDModel
from aicup_pid.local import LocalModel
from aicup_pid.opening import OpeningModel
from aicup_pid.orient import OrientModel
from aicup_pid.style import StyleModel, game_features

VIEWS = ("B", "W")
SPLIT = ("opening", "opening_raw", "orient4", "orient1", "local_split", "hd")
MERGED = ("local", "style", "orient4_merged", "emb", "seq")


def swap(moves):
    return [("W" if c == "B" else "B", *rest) for c, *rest in moves]


def view(moves, colour):
    """The parsed game with the target player's stones marked B."""
    return moves if colour == "B" else swap(moves)


def _split(make, cand_items, q_items, players):
    """Per-view models: cand_items {view: {player: [items]}}, q_items [[(item, view)]] -> [Q, C]."""
    models = {v: make().fit(cand_items[v]) for v in VIEWS}
    if any(list(m.players) != players for m in models.values()):
        raise ValueError("candidate order differs between the views")
    out = np.zeros((len(q_items), len(players)))
    for qi, games in enumerate(q_items):
        for v in VIEWS:
            items = [it for it, vv in games if vv == v]
            if items:
                out[qi] += models[v].log_scores(items)
    return out


def components(cands, questions, moves, sgfs, cfg, rows=None, emb=None, which=None, seq=None, rows_full=None, emb_full=None):
    """{name: [Q, C]}. cands {player: [(key, colour)]}, questions [[(key, colour)]], moves {key: parsed moves as recorded},
    sgfs {key: record}, rows {key: [256] trunk row of the player's side}, emb a fitted aicup_pid.embed.PlayerEmbedding,
    seq {key: [128] unit vector of the player's side from the sequence identity model (scripts/pid_v2_seq_embed.py)};
    rows_full {key: row of the UNPROJECTED trunk statistics (scripts/pid_trunk_extract.py --full)} and emb_full, one or several
    PlayerEmbedding fitted on such rows: the component 'embfull' (METHOD_NOTES 2.75)."""
    which = set(which or (*SPLIT, *MERGED))
    players = sorted(cands)
    vm = {k: view(moves[k], c) for games in (*cands.values(), *questions) for k, c in games}
    cand_m = {v: {p: [vm[k] for k, c in cands[p] if c == v] for p in players} for v in VIEWS}
    q_m = [[(vm[k], c) for k, c in q] for q in questions]
    all_m = {p: [vm[k] for k, _ in cands[p]] for p in players}
    a, b, depth = cfg.get("alpha", 2.0), cfg.get("beta", 5.0), cfg.get("depth", 12)
    S = {}
    if "opening" in which:
        S["opening"] = _split(lambda: OpeningModel(depth=depth, alpha=a, beta=b, alpha_full=cfg.get("alpha_full", 1.0)), cand_m, q_m, players)
    if "opening_raw" in which:
        S["opening_raw"] = _split(lambda: OpeningModel(depth=depth, alpha=a, beta=b, alpha_full=cfg.get("alpha_full", 1.0), symmetry=False), cand_m, q_m, players)
    for n_raw in (1, 4):
        if f"orient{n_raw}" in which:
            S[f"orient{n_raw}"] = _split(lambda: OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=n_raw),
                                         cand_m, q_m, players)
    if "local_split" in which:
        S["local_split"] = _split(lambda: LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12)), cand_m, q_m, players)
    if "hd" in which:
        S["hd"] = _split(lambda: HDModel(alpha=cfg.get("hd_alpha", 2.0), beta=cfg.get("hd_beta", 5.0)), {v: {p: [sgfs[k] for k, c in cands[p] if c == v] for p in players} for v in VIEWS},
                         [[(sgfs[k], c) for k, c in q] for q in questions], players)
    if "local" in which:
        m = LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12)).fit(all_m)
        S["local"] = np.stack([m.log_scores([vm[k] for k, _ in q]) for q in questions])
    if "orient4_merged" in which:
        m = OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=4).fit(all_m)
        S["orient4_merged"] = np.stack([m.log_scores([vm[k] for k, _ in q]) for q in questions])
    if "style" in which:
        feats = {k: game_features(g) for k, g in vm.items()}
        m = StyleModel(shrink=cfg.get("shrink", 0.1)).fit({p: np.stack([feats[k] for k, _ in cands[p]]) for p in players})
        if list(m.players) != players:
            raise ValueError("style: candidate order differs")
        S["style"] = np.stack([m.log_scores(np.stack([feats[k] for k, _ in q])) for q in questions])
    if "emb" in which and emb is not None:
        mats = []
        for one in (emb if isinstance(emb, (list, tuple)) else [emb]):  # several seeds: mean of their row-standardised matrices
            C = np.stack([one.set_vector(np.stack([rows[k] for k, _ in cands[p]])) for p in players])
            Q = np.stack([one.set_vector(np.stack([rows[k] for k, _ in q])) for q in questions])
            mats.append(one.scale * (Q @ C.T))
        if len(mats) == 1:
            S["emb"] = mats[0]
        else:
            from aicup_pid.fusion import standardise
            S["emb"] = np.mean([standardise(m) for m in mats], axis=0)
    if "embfull" in which and emb_full is not None:
        from aicup_pid.fusion import standardise
        mats = []
        for one in (emb_full if isinstance(emb_full, (list, tuple)) else [emb_full]):  # several seeds: mean of their row-standardised matrices
            C = np.stack([one.set_vector(np.stack([rows_full[k] for k, _ in cands[p]])) for p in players])
            Q = np.stack([one.set_vector(np.stack([rows_full[k] for k, _ in q])) for q in questions])
            mats.append(standardise(one.scale * (Q @ C.T)))
        S["embfull"] = np.mean(mats, axis=0)
    if "seq" in which and seq is not None:  # prototype = normalised sum of the game vectors, as the model was trained
        unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731
        mats = []
        # METHOD_NOTES 2.141: cfg["seq_near"] = {"k": 3, "weight": 0.5} adds, per network, the question's games against the candidate's
        # SINGLE reference games (mean cosine of the k most similar ones, averaged over the question's games), row-standardised
        near, nears, done = cfg.get("seq_near"), [], {}
        if near:
            c_games = [k for p in players for k, _ in cands[p]]
            c_len = np.array([len(cands[p]) for p in players])
            c_start = np.concatenate([[0], np.cumsum(c_len)[:-1]])
            q_games = [k for q in questions for k, _ in q]
            q_of = np.repeat(np.arange(len(questions)), [len(q) for q in questions])
            q_cnt = np.bincount(q_of, minlength=len(questions)).astype(np.float64)
            kk = int(near.get("k", 3))
        for one in (seq if isinstance(seq, (list, tuple)) else [seq]):  # several networks: mean of their row-standardised matrices
            C = unit(np.stack([np.sum([one[k] for k, _ in cands[p]], axis=0) for p in players]).astype(np.float64))
            Q = unit(np.stack([np.sum([one[k] for k, _ in q], axis=0) for q in questions]).astype(np.float64))
            mats.append(Q @ C.T)
            if near:
                G_ = np.stack([one[k] for k in c_games])
                mark = (G_[0].tobytes(), G_[-1].tobytes())  # a document lists a network once per unit of weight: compute it once
                if mark not in done:
                    X = unit(np.stack([one[k] for k in q_games]).astype(np.float64)) @ unit(G_.astype(np.float64)).T  # [question games, reference games]
                    if (c_len == c_len[0]).all():
                        m_ = min(kk, int(c_len[0]))
                        per = -np.partition(-X.reshape(len(q_games), len(players), int(c_len[0])), m_ - 1, axis=2)[:, :, :m_].mean(2)
                    else:
                        per = np.stack([np.sort(X[:, s_:s_ + n_], axis=1)[:, -min(kk, int(n_)):].mean(1) for s_, n_ in zip(c_start, c_len)], axis=1)
                    Nq = np.zeros((len(questions), len(players)))
                    np.add.at(Nq, q_of, per)
                    done[mark] = Nq / q_cnt[:, None]
                nears.append(done[mark])
        if len(mats) == 1:
            S["seq"] = mats[0]
        else:
            from aicup_pid.fusion import standardise
            S["seq"] = np.mean([standardise(m) for m in mats], axis=0)
            if near:
                S["seq"] = S["seq"] + float(near.get("weight", 0.5)) * np.mean([standardise(m) for m in nears], axis=0)
    return S


def official_pool(cand_path, q_path):
    """(cands {player: [(key, colour)]}, questions [[(key, colour)]], question ids, sgfs {key: record}) of one added pool; keys as
    scripts/pid_trunk_extract.py --official2 writes them (c:<row>:<game_id>, q:<question_id>:<i>)."""
    import pandas as pd
    from aicup_pid import pools
    c = pd.read_csv(cand_path, dtype=str, keep_default_na=False)
    cands, sgfs = {}, {}
    for i, r in enumerate(c.itertuples()):
        key = f"c:{i}:{r.game_id}"
        cands.setdefault(r.player_id, []).append((key, r.color))
        sgfs[key] = r.sgf_content
    questions, qids = [], []
    for q in pools.read_questions(q_path):
        games = []
        for i, (sgf, colour) in enumerate(q["games"]):
            key = f"q:{q['question_id']}:{i}"
            games.append((key, colour or "B"))
            sgfs[key] = sgf
        questions.append(games)
        qids.append(q["question_id"])
    return cands, questions, qids, sgfs


def candidate_holdout(cands, h, seed):
    """{player: [h held-out (key, colour)]}: the labelled pseudo-pool cut from an official pool's own candidates (METHOD_NOTES
    2.41). Per candidate (sorted ids, games in file order) h games drawn without replacement, in the drawn order; the same
    draw in scripts/pid_v2_seq_finetune.py --holdout (those games stay out of the fine-tune) and scripts/pid_v2_offsim.py."""
    rng = np.random.default_rng(4100 + seed)
    return {p: [cands[p][i] for i in rng.permutation(len(cands[p]))[:h]] for p in sorted(cands)}


def original_sim(cand_csv, keys, seed, n_ref=10, sizes=(1, 1, 1, 2, 2, 3, 5)):
    """A simulated pool cut from an ORIGINAL pool's candidates (METHOD_NOTES 2.55, item 7): official-population players no
    network has seen, 10-game candidate sets, questions of 1 / 2 / 3 / 5 games. Per candidate (sorted ids), among its games
    in `keys` (those with per-ply vectors), a random order by `seed`: the first n_ref are the candidate set, the next ones the
    questions of the given sizes. Returns (cands {player: [(key, 'B')]}, questions [[(key, 'B')]], truth [player ids],
    sgfs {key: record}). Keys c:<row in the file>:<game_id>. The original pools are Black only."""
    import pandas as pd
    c = pd.read_csv(cand_csv, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(5200 + seed)
    by, rec = {}, {}
    for i, r in enumerate(c.itertuples()):
        k = f"c:{i}:{r.game_id}"
        if k in keys:
            by.setdefault(r.player_id, []).append(k)
            rec[k] = r.sgf_content
    cands, questions, truth = {}, [], []
    for p in sorted(by):
        g = [by[p][i] for i in rng.permutation(len(by[p]))]
        if len(g) < n_ref + sum(sizes):
            continue
        cands[p] = [(k, "B") for k in g[:n_ref]]
        at = n_ref
        for m in sizes:
            questions.append([(k, "B") for k in g[at:at + m]])
            truth.append(p)
            at += m
    return cands, questions, truth, {k: rec[k] for games in (*cands.values(), *questions) for k, _ in games}


RANK_ROWS = ("results/m82/rows_train.npz", "results/m84_rowsp")  # under AI_CUP_2026-1, written by the rank session (read-only here)
_PERM = np.r_[0:64, 128:192, 64:128, 192:256]


def load_side_rows(aicup1_root, keep=None):
    """{training game key: [256] row of the game owner's side, either colour}, in competition 1's block order (mean and sd of
    the pooled vector, then of the played-point vector). The rank session stores the same features as [means, sds]; on the
    Black games both sessions have, the rows agree to 0.002 after the reorder (data/day_scripts/rows_compat.py, 2026-11-12)."""
    import glob
    from pathlib import Path
    root, out = Path(aicup1_root), {}
    z = np.load(root / RANK_ROWS[0])
    files = [(z["key"], z["T"], z["nT"])]
    for f in sorted(glob.glob(str(root / RANK_ROWS[1] / "*.npz"))):
        y = np.load(f)
        files.append((y["key"], y["T"], y["nT"]))
    for key, T, n in files:
        for k, row, m in zip(key.astype(str), T, n):
            if m > 0 and (keep is None or k in keep):
                out[k] = np.ascontiguousarray(row[_PERM], dtype=np.float32)
    return out


def training_games(train_dir):
    """(player_id, rank, game_key, color) of every training game, game_key 'RANK:game_id' as aicup.scoring."""
    import pandas as pd
    from pathlib import Path
    from aicup.metric import LEVELS
    frames = []
    for r in LEVELS:
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"], dtype=str)
        df["game_key"] = df["rank"] + ":" + df["game_id"]
        frames.append(df[["player_id", "rank", "game_key", "color"]])
    return pd.concat(frames, ignore_index=True)


def pool_normalised(vecs, cands, questions, shrink=0.3):
    """[Q, C] cosine of prototypes and question vectors after within-class covariance normalisation fitted on the pool's own
    candidates: W = covariance of the candidates' game vectors around their player's mean, shrunk towards (trace / d) I by
    `shrink`; every vector is multiplied by W^(-1/2) and renormalised. vecs {key: unit vector}; no question enters W."""
    players = sorted(cands)
    X = [np.stack([vecs[k] for k, _ in cands[p]]).astype(np.float64) for p in players]
    resid = np.vstack([x - x.mean(axis=0) for x in X])
    d = resid.shape[1]
    W = resid.T @ resid / max(len(resid) - len(X), 1)
    W = (1 - shrink) * W + shrink * (np.trace(W) / d) * np.eye(d)
    val, vec = np.linalg.eigh(W)
    B = vec @ np.diag(1.0 / np.sqrt(np.maximum(val, 1e-12))) @ vec.T
    unit = lambda M: M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)  # noqa: E731
    C = unit(np.stack([unit(x @ B).sum(axis=0) for x in X]))
    Q = unit(np.stack([unit(np.stack([vecs[k] for k, _ in q]).astype(np.float64) @ B).sum(axis=0) for q in questions]))
    return Q @ C.T


def load_phase_rows(aicup1_root, stored_dir, keep=None):
    """{training game key: [640]} = the 256-dim row of the owner's side (competition 1's block order) + the three phase means
    (plies t < 60, 60-150, >= 150; 128 dims each). Sources: the rank session's results/m84_rowsp (games without stored per-ply
    vectors) and scripts/pid_v2_phase_rows.py's stored_f<fold>.npz (the others); the same function computed both."""
    import glob
    from pathlib import Path
    out = {}
    files = sorted(glob.glob(str(Path(aicup1_root) / RANK_ROWS[1] / "*.npz"))) + sorted(glob.glob(str(Path(stored_dir) / "stored_f*.npz")))
    for f in files:
        z = np.load(f)
        n = z["nT"] if "nT" in z.files else np.ones(len(z["key"]))
        for k, row, ph, m in zip(z["key"].astype(str), z["T"], z["PT"], n):
            if m > 0 and (keep is None or k in keep):
                out[k] = np.concatenate([row[_PERM].astype(np.float32), ph.astype(np.float32)])
    return out


def load_store_phase(*dirs):
    """{official game key: [640]} from pid_trunk_extract.py parts that carry the phase means."""
    import glob
    store = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            if f.endswith(".tmp.npz"):
                continue
            z = np.load(f, allow_pickle=False)
            for k, row, ph in zip(z["keys"], z["feats"], z["phase"]):
                store[str(k)] = np.concatenate([row.astype(np.float32), ph.astype(np.float32)])
    return store
