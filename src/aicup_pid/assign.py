"""Global one-to-one assignment of questions to candidates (fork plan line B3).

The official pools have 400 questions for 400 candidates, 100 per size, so very likely one question per candidate. When
that holds, the Top-1 picks should form a permutation: assign_topk replaces the per-question argmax by the assignment that
maximises the summed fused scores under the one-to-one constraint (Hungarian algorithm; a per-row constant such as the
log-softmax normaliser does not change it), and keeps the rest of each row's order for Top-2..k. With more questions than
candidates the unassigned rows keep their argmax. assignment_report counts what the assignment changed.
"""
import numpy as np  # 矩陣運算
from scipy.optimize import linear_sum_assignment  # 匈牙利演算法（線性指派）

from aicup_pid.fusion import topk  # 逐題取前 k 名


def assign_topk(S, k=5, capacity=1):  # 硬性一對一配對決定 Top-1
    """[Q, min(k, C)] candidate indices: Top-1 from the one-to-one assignment, then the row's remaining order.
    capacity > 1: every candidate may take up to `capacity` questions (the columns are tiled), for pools with several
    questions per candidate."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    base = topk(S, k)  # 逐題的前 k 名，作為其餘名次的順序
    Q, C = S.shape  # 題數與候選人數
    if Q == 0 or C == 0:  # 空矩陣無從配對
        return base  # 直接回傳逐題結果
    capacity = int(capacity)  # 每位候選人可接的題數
    if capacity < 1:  # 容量不合法
        raise ValueError("capacity must be >= 1")  # 拒絕執行
    rows, cols = linear_sum_assignment(np.tile(S, (1, capacity)) if capacity > 1 else S, maximize=True)  # 求融合分數總和最大的配對
    cols = cols % C  # 平鋪後的欄換回候選人索引
    out = base.copy()  # 從逐題結果開始修改
    width = base.shape[1]  # 每列輸出的名次數
    for q, c in zip(rows, cols):  # 逐一套用配對結果
        if out[q, 0] == c:  # 配對與逐題第一名相同
            continue  # 不需要修改
        rest = [x for x in base[q].tolist() if x != c]  # 其餘候選人保留原順序
        out[q] = np.array([c, *rest][:width])  # 配到的候選人放第一名
    return out  # 回傳 [Q, k] 索引


def assignment_report(S, assigned):  # 統計配對前後的差異
    """{n_questions, n_candidates, distinct_top1_before, distinct_top1_after, n_top1_changed}."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    Q, C = S.shape  # 題數與候選人數
    if Q == 0:  # 沒有題目
        return {"n_questions": 0, "n_candidates": int(C), "distinct_top1_before": 0, "distinct_top1_after": 0, "n_top1_changed": 0}  # 回傳全為 0 的報告
    before = np.argmax(S, axis=1)  # 配對前的逐題 Top-1
    after = np.asarray(assigned)[:, 0]  # 配對後的 Top-1
    return {"n_questions": int(Q), "n_candidates": int(C), "distinct_top1_before": int(len(set(before.tolist()))),  # 規模與配對前的不同 Top-1 數
            "distinct_top1_after": int(len(set(after.tolist()))), "n_top1_changed": int((before != after).sum())}  # 配對後的不同數與改變的題數


def assign_blocks(S, blocks, apply, k=5):  # 分區塊的一對一配對
    """assign_topk block by block (11-11 release: the original questions and the added ones are separate one-to-one sets
    over the same candidates). blocks [Q] labels; apply {label: bool}: a block with False keeps its per-question Top-k."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    blocks = np.asarray(blocks)  # 每題的區塊標籤
    out = topk(S, k)  # 預設為逐題前 k 名
    for b in sorted(set(blocks.tolist())):  # 逐一處理每個區塊
        if apply.get(b):  # 這個區塊要套用配對
            rows = np.flatnonzero(blocks == b)  # 該區塊的題目列
            out[rows] = assign_topk(S[rows], k)  # 只在區塊內做一對一配對
    return out  # 回傳 [Q, k] 索引


def assign_soft(S, lam, k=5, copies=3):  # 軟性配對：共用候選人要付懲罰
    """Top-k with a penalty for sharing a candidate: the j-th question a candidate takes costs (j - 1) * lam of fused score
    (assignment over `copies` penalised copies of every column). lam = 0 is the per-question Top-1 while collisions stay within
    `copies`; a large lam is assign_topk. For pools where most players have one question but some have several (the v2 added
    pools): a collision survives only when both questions prefer the candidate by more than lam over their next choice."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    base = topk(S, k)  # 逐題的前 k 名
    Q, C = S.shape  # 題數與候選人數
    if Q == 0 or C == 0:  # 空矩陣無從配對
        return base  # 直接回傳逐題結果
    big = np.concatenate([S - j * float(lam) for j in range(int(copies))], axis=1)  # 副本 j（從 0 起）整欄扣 j×lam：同一位候選人接第 j+1 題時要付的代價
    rows, cols = linear_sum_assignment(big, maximize=True)  # 在擴充後的矩陣上求最大配對
    cols = cols % C  # 副本的欄換回候選人索引
    out = base.copy()  # 從逐題結果開始修改
    width = base.shape[1]  # 每列輸出的名次數
    for q, c in zip(rows, cols):  # 逐一套用配對結果
        if out[q, 0] != c:  # 配對改變了第一名
            out[q] = np.array([c, *[x for x in base[q].tolist() if x != c]][:width])  # 配到的候選人放第一，其餘照原順序
    return out  # 回傳 [Q, k] 索引


def sinkhorn_prices(S, tau, iters=300):  # 每位候選人被爭搶的程度（價格）
    """log v [C] <= 0: the price of every candidate in the entropic assignment with row sums 1 and column sums <= 1
    (METHOD_NOTES 2.122). A candidate that several questions want gets a negative price; an uncontested one keeps 0."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    K = np.exp((S - S.max(axis=1, keepdims=True)) / float(tau))  # 以溫度 tau 換成核矩陣（先減每列最大值避免溢位）
    v = np.ones(S.shape[1])  # 欄的縮放因子，從 1 開始
    for _ in range(int(iters)):  # 固定次數的 Sinkhorn 交替縮放
        u = 1.0 / np.maximum(K @ v, 1e-300)  # 讓每列和為 1
        v = np.minimum(1.0, 1.0 / np.maximum(K.T @ u, 1e-300))  # 讓每欄和不超過 1（只縮小不放大）
    return np.log(np.maximum(v, 1e-300))  # 回傳對數價格（<= 0）


def assign_sinkhorn(S, tau, k=5, lam=None):  # 依價格調整後的前 k 名；最終流程以 λ 2、τ 0.5 用於多局題
    """Top-k of the scores minus tau x the Sinkhorn price of each candidate: unlike assign_soft / assign_topk it changes the
    whole order, not only the Top-1. With lam, the Top-1 is the one of assign_soft(S, lam) and the rest follows the priced
    order (a collision between two confident questions survives, as in assign_soft)."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    Q, C = S.shape  # 題數與候選人數
    if Q == 0 or C == 0:  # 空矩陣無從配對
        return topk(S, k)  # 直接回傳逐題結果
    order = topk(S + float(tau) * sinkhorn_prices(S, tau)[None, :], min(C, k + 3))  # 分數加上 tau 倍的對數價格（<= 0）後排序，取前 k+3 名（不超過候選人數）
    out = order[:, :min(k, C)].copy()  # 取前 k 名
    if lam is not None:  # 指定 lam 時 Top-1 改用軟性配對
        first = assign_soft(S, lam, k)[:, 0]  # 軟性配對的 Top-1
        for q, c in enumerate(first):  # 逐題替換第一名
            if out[q, 0] != c:  # 與價格排序的第一名不同
                out[q] = np.array([c, *[x for x in order[q].tolist() if x != c]][:out.shape[1]])  # 軟性 Top-1 放第一，其餘照價格排序
    return out  # 回傳 [Q, k] 索引
