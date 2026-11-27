"""Player-level bootstrap of a paired difference of per-question scores (the candidates are the resampling unit)."""
import numpy as np  # 陣列運算


def paired_player_bootstrap(s_a, s_b, player_ids, n=2000, seed=42):  # 成對分數差的玩家層級 bootstrap
    """{diff, ci95, n_players, n_questions} of mean(s_a - s_b): players resampled with replacement, all their questions kept."""
    s_a, s_b, pid = np.asarray(s_a, float), np.asarray(s_b, float), np.asarray(player_ids).astype(str)  # 統一型別
    d = s_a - s_b  # 每題的分數差
    players = np.unique(pid)  # 所有玩家
    groups = [np.flatnonzero(pid == p) for p in players]  # 每位玩家的題目索引
    sums = np.array([d[g].sum() for g in groups])  # 每位玩家的分數差總和
    counts = np.array([len(g) for g in groups], dtype=float)  # 每位玩家的題數
    rng = np.random.RandomState(seed)  # 固定種子的亂數
    reps = np.empty(n)  # 每次重抽的平均差
    for i in range(n):  # 重抽 n 次
        pick = rng.randint(0, len(players), len(players))  # 以放回方式抽玩家
        reps[i] = sums[pick].sum() / counts[pick].sum()  # 抽中玩家所有題目的平均差
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],  # 平均差與 95% 信賴區間
            "n_players": int(len(players)), "n_questions": int(len(d))}  # 玩家數與題數
