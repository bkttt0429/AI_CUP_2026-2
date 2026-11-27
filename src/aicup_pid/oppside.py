"""Tail evidence from single reference games (METHOD_NOTES 2.117): the opponents of a question against the opponents of a
candidate, and its relatives.

A sequence network returns a vector for the side flagged as the subject of a game; with the other colour flagged it describes
the OPPONENT (experiments/pid_v2_opp_embed.py). A player meets some opponents again, so the opponent of a question game may also
be the opponent in one of the owner's reference games. On the official candidates (leave-one-game-out, experiments/pid_probe_opp.py)
the nearest opponent vector belongs to the same candidate in 3.4% of the games; 0.15% if nothing were there. The event is
rare and then decisive, so only the tail counts:
  per network cos(o_g, o_r) of a question game g and a reference game r; the networks' cosines are averaged;
  m[g, c] = the maximum over the reference games of candidate c;  z[g, .] = m standardised over the candidates;
  score[q, c] = sum over the games g of question q of max(z[g, c] - tau, 0).
The score is added to the fused score with its own weight. It is NOT standardised again: most of its entries are zero.
The same statistic on the subject's OWN vectors says that one reference game looks like the question game (the prototype
averages that away), and on the sum of both sides that the same pairing met again.
"""
import numpy as np  # 陣列運算


def nearest_z(cands, questions, sides):  # 每局題目棋對每位候選人的最佳單局相似度（標準化後）
    """(z [G, C], {question game key: row}): per question game and candidate, the best single reference game under the SUM
    over `sides` of the network-averaged cosines, standardised over the candidates. cands {player: [(key, colour)]},
    questions [[(key, colour)]], sides = [[{key: vector} per network] per side]; candidates in sorted id order. Games without
    a vector are skipped; a candidate without any takes the row mean (z = 0)."""
    players = sorted(cands)  # 候選人依 id 排序
    have = set(sides[0][0])  # 有向量的棋局鍵（以第一方第一個網路為準）
    ckeys = [k for p in players for k, _ in cands[p] if k in have]  # 有向量的參考棋局
    cidx = np.array([i for i, p in enumerate(players) for k, _ in cands[p] if k in have])  # 每局參考棋所屬的候選人索引
    qkeys = sorted({k for q in questions for k, _ in q if k in have})  # 有向量的題目棋局
    if set(qkeys) & set(ckeys):  # 題目棋不可同時是參考棋
        raise ValueError("a question game is among the reference games")  # 資料有誤就中止
    unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731  逐列正規化成單位向量
    cos = np.zeros((len(qkeys), len(ckeys)))  # [題目棋局, 參考棋局] 的餘弦
    for nets in sides:  # 逐一處理每一方（主體方或對手方）
        one_side = np.zeros_like(cos)  # 這一方各網路的餘弦和
        for one in nets:  # 逐個網路
            one_side += unit(np.stack([one[k] for k in qkeys]).astype(np.float64)) @ unit(np.stack([one[k] for k in ckeys]).astype(np.float64)).T  # 這個網路的餘弦
        cos += one_side / len(nets)  # 網路間取平均後，再把各方相加
    order = np.argsort(cidx, kind="stable")  # 依候選人排序參考棋局
    sorted_c = cidx[order]  # 排序後的候選人索引
    starts = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1]])  # 每位候選人區段的起點
    m = np.full((len(qkeys), len(players)), np.nan)  # 每位候選人的最大餘弦（沒有棋局時為 NaN）
    m[:, sorted_c[starts]] = np.maximum.reduceat(cos[:, order], starts, axis=1)  # 每個區段取最大值
    mu = np.nanmean(m, axis=1, keepdims=True)  # 每局題目棋對所有候選人的平均
    sd = np.nanstd(m, axis=1, keepdims=True)  # 每局題目棋對所有候選人的標準差
    sd[sd < 1e-12] = 1.0  # 避免除以 0
    z = np.where(np.isnan(m), 0.0, (m - mu) / sd)  # 在候選人之間標準化；沒有棋局的候選人取 0（列平均）
    return z, {k: i for i, k in enumerate(qkeys)}  # 回傳 z 與題目棋局鍵 -> 列號


def opp_z(cands, questions, opp):  # 只用對手方向量的 nearest_z
    """nearest_z for one side: opp = one or several {key: opponent-side vector}."""
    return nearest_z(cands, questions, [list(opp) if isinstance(opp, (list, tuple)) else [opp]])  # 包成只有一方的網路清單


def opp_scores(cands, questions, opp, tau=3.0, zq=None):  # 尾端證據的 [Q, C] 分數
    """[Q, C]: sum over a question's games of the part of z above tau. zq = opp_z(...) / nearest_z(...) when several tau are wanted."""
    z, row = zq if zq is not None else opp_z(cands, questions, opp)  # 重用已算好的 z，或現在計算
    h = np.maximum(z - tau, 0.0)  # 只保留超過 tau 的尾端
    out = np.zeros((len(questions), z.shape[1]))  # [Q, C] 分數
    for qi, q in enumerate(questions):  # 逐題
        for k, _ in q:  # 逐局題目棋
            if k in row:  # 有向量的棋局
                out[qi] += h[row[k]]  # 累加該局的尾端證據
    return out  # 回傳分數矩陣
