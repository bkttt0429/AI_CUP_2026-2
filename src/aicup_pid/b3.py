"""Line B3 (METHOD_NOTES 2.25-2.26): the rank model ct9's 32-dim question embedding zq as an identification component.

Every candidate's games are cut into chunks of at most 20 (the rank model's question size). Each chunk and each question
goes through the rank session's test-side path (AI_CUP_2026-1 predict_submission.py -> trunk_predict.py extract /
predict, read-only) with a hook that keeps zq for every pass (checkpoint x L1b head); see experiments/pid_b3_pools.py
(leak-free simulated pools) and experiments/pid_b3_official.py (official files).
score_b3: candidate vector = mean of its chunks' zq; every dimension z-scored with the candidates' mean / sd; cosine with
the question's zq; the passes' [Q, C] matrices averaged. This is the component the 11-09 pool test passed with.
"""
import numpy as np  # 陣列運算
import pandas as pd  # 組出棋力測試格式的資料表

CHUNK = 20  # 每塊最多的局數（棋力模型的題目大小）
RANK_COLUMNS = ["question_id", "num_games", *[f"sgf_{i}" for i in range(1, 21)], *[f"color_{i}" for i in range(1, 21)]]  # 棋力測試檔的欄位


def chunk_slices(n, size=CHUNK):  # 把 n 局切成近乎等長的連續塊
    """Index arrays of ceil(n / size) near-equal consecutive chunks (100 games -> 5 x 20, 45 -> 3 x 15)."""
    if n <= 0:  # 沒有棋局
        return []  # 沒有任何塊
    return np.array_split(np.arange(n), -(-n // size))  # 塊數為 n / size 無條件進位


def rank_inputs(cands, prefix, size=CHUNK):  # 把候選人的棋局轉成棋力模型的題目格式
    """Official candidates (player_id, game_id, sgf_content | sgf, color) -> (rank-test-format rows, map).

    Players in sorted id order, games in game_id order (as the rank session's 11-08 B1 build); chunk j of the i-th player
    (1-based) is question `<prefix>_c<i:04d>_k<j>`. map: question_id, player_id, chunk, game_ids ('|'-joined)."""
    sgf_col = "sgf_content" if "sgf_content" in cands.columns else "sgf"  # 棋譜所在的欄位名稱
    rows, mp = [], []  # 輸出列與對照表
    for i, (pid, g) in enumerate(cands.groupby("player_id", sort=True), start=1):  # 候選人依 id 排序，從 1 編號
        g = g.sort_values("game_id", kind="stable")  # 棋局依 game_id 排序
        sg, gid, col = g[sgf_col].tolist(), g["game_id"].tolist(), g["color"].tolist()  # 棋譜、game_id 與執色
        for j, idx in enumerate(chunk_slices(len(sg), size)):  # 逐塊產生一題
            qid = f"{prefix}_c{i:04d}_k{j}"  # 這一塊的題目 id
            row = {"question_id": qid, "num_games": str(len(idx))}  # 題目列的開頭欄位
            for t, x in enumerate(idx):  # 逐局填入
                row[f"sgf_{t + 1}"] = sg[x]  # 棋譜
                row[f"color_{t + 1}"] = col[x]  # 執色
            rows.append(row)  # 加入輸出列
            mp.append({"question_id": qid, "player_id": pid, "chunk": j, "game_ids": "|".join(gid[x] for x in idx)})  # 記下題目對應的候選人與棋局
    return pd.DataFrame(rows).reindex(columns=RANK_COLUMNS).fillna(""), pd.DataFrame(mp, columns=["question_id", "player_id", "chunk", "game_ids"])  # 補齊欄位後回傳兩個表


def score_b3(Z, row, chunks_by_player, players, question_ids):  # B3 成分的分數矩陣
    """([Q, C], info) from Z [passes, rows, dims] and row {captured id: row index}.

    chunks_by_player {player: [chunk ids]}; candidates in `players` order. A question without a captured row gets a zero
    row and a candidate without any captured chunk a zero column (both neutral after the fusion's row standardisation);
    info lists them."""
    Z = np.asarray(Z, dtype=np.float64)  # [推論次數（檢查點 × L1b 等級頭）, 列, 維度]
    have = {p: [row[c] for c in chunks_by_player.get(p, []) if c in row] for p in players}  # 每位候選人已擷取到的塊的列索引
    missing_c = [p for p in players if not have[p]]  # 沒有任何塊的候選人
    missing_q = [q for q in question_ids if q not in row]  # 沒有擷取到的題目
    if len(missing_c) == len(players):  # 所有候選人都沒有塊
        raise ValueError("no candidate has a captured chunk")  # 無法計分
    sims = []  # 每次推論的 [Q, C]
    for M in Z:  # 逐次推論處理
        C = np.stack([M[have[p]].mean(axis=0) if have[p] else np.full(M.shape[1], np.nan) for p in players])  # 候選人向量（沒有塊時為 NaN）
        mu = np.nanmean(C, axis=0)  # 候選人向量的平均
        C = np.where(np.isnan(C), mu, C)  # 缺少的候選人以平均補上
        sd = C.std(axis=0) + 1e-6  # 候選人向量的標準差
        Q = np.stack([M[row[q]] if q in row else mu for q in question_ids]) if question_ids else np.zeros((0, M.shape[1]))  # 題目向量（缺少的以平均補上）
        Zc, Zq = (C - mu) / sd, (Q - mu) / sd  # 每一維以候選人的平均與標準差做 z 分數
        Zc /= np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12  # 候選人向量正規化
        Zq /= np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-12  # 題目向量正規化
        sims.append(Zq @ Zc.T)  # 餘弦相似度
    return np.mean(sims, axis=0), {"missing_questions": missing_q, "missing_candidates": missing_c, "passes": int(len(Z))}  # 各次推論的矩陣取平均，並附上缺漏資訊
