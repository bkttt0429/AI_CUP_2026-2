"""Line B3 (METHOD_NOTES 2.25-2.26): the rank model ct9's 32-dim question embedding zq as an identification component.

Every candidate's games are cut into chunks of at most 20 (the rank model's question size). Each chunk and each question
goes through the rank session's test-side path (AI_CUP_2026-1 predict_submission.py -> trunk_predict.py extract /
predict, read-only) with a hook that keeps zq for every pass (checkpoint x L1b head); see scripts/pid_b3_pools.py
(leak-free simulated pools) and scripts/pid_b3_official.py (official files).
score_b3: candidate vector = mean of its chunks' zq; every dimension z-scored with the candidates' mean / sd; cosine with
the question's zq; the passes' [Q, C] matrices averaged. This is the component the 11-09 pool test passed with.
"""
import numpy as np
import pandas as pd

CHUNK = 20
RANK_COLUMNS = ["question_id", "num_games", *[f"sgf_{i}" for i in range(1, 21)], *[f"color_{i}" for i in range(1, 21)]]


def chunk_slices(n, size=CHUNK):
    """Index arrays of ceil(n / size) near-equal consecutive chunks (100 games -> 5 x 20, 45 -> 3 x 15)."""
    if n <= 0:
        return []
    return np.array_split(np.arange(n), -(-n // size))


def rank_inputs(cands, prefix, size=CHUNK):
    """Official candidates (player_id, game_id, sgf_content | sgf, color) -> (rank-test-format rows, map).

    Players in sorted id order, games in game_id order (as the rank session's 11-08 B1 build); chunk j of the i-th player
    (1-based) is question `<prefix>_c<i:04d>_k<j>`. map: question_id, player_id, chunk, game_ids ('|'-joined)."""
    sgf_col = "sgf_content" if "sgf_content" in cands.columns else "sgf"
    rows, mp = [], []
    for i, (pid, g) in enumerate(cands.groupby("player_id", sort=True), start=1):
        g = g.sort_values("game_id", kind="stable")
        sg, gid, col = g[sgf_col].tolist(), g["game_id"].tolist(), g["color"].tolist()
        for j, idx in enumerate(chunk_slices(len(sg), size)):
            qid = f"{prefix}_c{i:04d}_k{j}"
            row = {"question_id": qid, "num_games": str(len(idx))}
            for t, x in enumerate(idx):
                row[f"sgf_{t + 1}"] = sg[x]
                row[f"color_{t + 1}"] = col[x]
            rows.append(row)
            mp.append({"question_id": qid, "player_id": pid, "chunk": j, "game_ids": "|".join(gid[x] for x in idx)})
    return pd.DataFrame(rows).reindex(columns=RANK_COLUMNS).fillna(""), pd.DataFrame(mp, columns=["question_id", "player_id", "chunk", "game_ids"])


def score_b3(Z, row, chunks_by_player, players, question_ids):
    """([Q, C], info) from Z [passes, rows, dims] and row {captured id: row index}.

    chunks_by_player {player: [chunk ids]}; candidates in `players` order. A question without a captured row gets a zero
    row and a candidate without any captured chunk a zero column (both neutral after the fusion's row standardisation);
    info lists them."""
    Z = np.asarray(Z, dtype=np.float64)
    have = {p: [row[c] for c in chunks_by_player.get(p, []) if c in row] for p in players}
    missing_c = [p for p in players if not have[p]]
    missing_q = [q for q in question_ids if q not in row]
    if len(missing_c) == len(players):
        raise ValueError("no candidate has a captured chunk")
    sims = []
    for M in Z:
        C = np.stack([M[have[p]].mean(axis=0) if have[p] else np.full(M.shape[1], np.nan) for p in players])
        mu = np.nanmean(C, axis=0)
        C = np.where(np.isnan(C), mu, C)
        sd = C.std(axis=0) + 1e-6
        Q = np.stack([M[row[q]] if q in row else mu for q in question_ids]) if question_ids else np.zeros((0, M.shape[1]))
        Zc, Zq = (C - mu) / sd, (Q - mu) / sd
        Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12
        Zq /= np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-12
        sims.append(Zq @ Zc.T)
    return np.mean(sims, axis=0), {"missing_questions": missing_q, "missing_candidates": missing_c, "passes": int(len(Z))}
