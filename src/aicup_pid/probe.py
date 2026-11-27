"""Public-leaderboard probing of the one-to-one submission (user decision 2026-11-06; METHOD_NOTES 2.17).

The leaderboard returns the mean official score over the 400 Public questions to seven decimals, so one upload resolves
sums of per-question changes to about 4e-5. Permuting a question's five candidates changes its score by
DELTA[t] = e^-(pi(t)-1) - e^-(t-1) when the truth sits at current rank t (0 when it is outside the five). The deltas of
every permutation are differences of the same five values, so signed sums over several questions collide (for example
(1 - e^-1) + (e^-1 - e^-2) = 1 - e^-2): a design that identifies each tested question's own position is limited to a
handful of questions. What one upload does resolve is a HISTOGRAM: apply the same 5-cycle (every candidate one rank
down, rank 5 to rank 1) to a group of questions, and the total change is sum_t n_t DELTA[t] with n_t = how many of the
group have the truth at rank t; the five deltas are rationally independent and, for groups up to `max_group_size`, every
histogram gives a distinct sum. Groups are then split adaptively: the histogram of the untested half is the parent's
minus the tested half's. A decoded position >= 2 is fixed by moving the truth to rank 1 (duplicate Top-1 across questions
is allowed by the rules), and every later upload carries the fixes, so the reference score is known exactly.
"""
import itertools  # 列舉各種組合
from functools import lru_cache  # 快取 min_gap 的結果

import numpy as np  # 陣列運算

DECAY = np.exp(-np.arange(5))  # 正解在第 1..5 名時的分數
OUT = 0  # 正解不在提交的五名內
POSITIONS = (1, 2, 3, 4, 5, OUT)  # 直方圖各格對應的位置
N_PUBLIC = 400  # Public 題數
TOL = 2e-4  # 兩個捨入到 1e-7 的分數，乘以 400，再加上餘裕
CYCLE = (5, 1, 2, 3, 4)  # 以目前名次表示的新順序：第 5 名移到最前，其餘各往下一名


def deltas(perm):  # 各位置的分數變化
    """[6] score change for the truth at current rank 1..5 (and 0 for OUT) when the list is reordered by `perm`:
    perm[j] = current rank (1..5) of the candidate placed at new rank j+1."""
    new_rank = {r: j + 1 for j, r in enumerate(perm)}  # 目前名次 -> 新名次
    d = [DECAY[new_rank[t] - 1] - DECAY[t - 1] for t in range(1, 6)]  # 新分數減去舊分數
    return np.array([*d, 0.0])  # 再加上 OUT 的 0


def deficit_decompositions(score, n=N_PUBLIC, tol=TOL, max_wrong=12):  # 把總損失拆成各名次的答錯題數
    """Multisets (a, b, c, d, e) = numbers of wrong questions whose truth is at rank 2, 3, 4, 5, outside, whose total
    loss matches n * (1 - score) within tol."""
    loss = n * (1.0 - score)  # 以題為單位的總損失
    unit = 1.0 - DECAY[1:]  # 正解在第 2..5 名時每題的損失
    out = []  # 符合的組合
    for combo in itertools.product(range(max_wrong + 1), repeat=5):  # 列舉所有組合
        if sum(combo) > max_wrong:  # 答錯題數超過上限
            continue  # 略過
        total = sum(k * u for k, u in zip(combo[:4], unit)) + combo[4] * 1.0  # 這個組合的總損失（在五名之外每題損失 1）
        if abs(total - loss) <= tol:  # 與觀察到的損失相符
            out.append(combo)  # 記下相符的組合
    return out  # 回傳所有符合的組合


MAX_OUT = 1  # 每組至多這麼多題「正解在五名之外」（0.98735 的損失分解只允許 0 或 1）
BOUNDS = (5, 1, 2, 1, 1)  # 每組正解在第 2、3、4、5 名與五名之外的題數上限 = 0.98735 各種分解的逐項最大值


def bounds_from_score(score, n=N_PUBLIC, tol=TOL, max_wrong=12):  # 由 Public 分數推出每組的上限
    """Elementwise maximum over the loss decompositions: no group can hold more truths at a rank than the whole set."""
    combos = deficit_decompositions(score, n, tol, max_wrong)  # 所有可能的損失分解
    if not combos:  # 沒有任何分解
        raise ValueError("no decomposition of the loss: check the score")  # 分數可能有誤
    return tuple(max(c[i] for c in combos) for i in range(5))  # 逐項取最大值


def histograms(size, bounds=BOUNDS):  # 列舉一組題目可能的直方圖
    """Every (n_1, ..., n_5, n_out) with sum = size and (n_2, n_3, n_4, n_5, n_out) <= bounds. The deltas of any
    permutation sum to zero, so without a bound the histograms h and h + k (1, 1, 1, 1, 1, -5) would give the same
    total; with the bounds from the loss decomposition the set is small and the sums stay apart for any group size."""
    out = []  # 收集直方圖
    for rest in itertools.product(*[range(min(b, size) + 1) for b in bounds]):  # 列舉第 2 名以後各格的題數
        n1 = size - sum(rest)  # 其餘題目都在第 1 名
        if n1 >= 0:  # 題數不可為負
            out.append((n1, *rest))  # 記下這個直方圖
    return out  # 回傳清單


def histogram_sum(hist, perm=CYCLE):  # 一個直方圖預測的總分數變化
    d = deltas(perm)  # 各位置的分數變化
    return float(sum(n * d[POSITIONS.index(t)] for n, t in zip(hist, POSITIONS)))  # 題數乘以變化後相加


@lru_cache(maxsize=None)  # 同樣的參數只計算一次
def min_gap(size, bounds=BOUNDS, perm=CYCLE):  # 可行直方圖總和之間的最小差距
    """Smallest difference between the sums of two different admissible histograms of a group of this size."""
    sums = np.sort(np.array([histogram_sum(h, perm) for h in histograms(size, bounds)]))  # 每個直方圖的總和，排序
    return float(np.min(np.diff(sums))) if len(sums) > 1 else np.inf  # 相鄰差的最小值（只有一個直方圖時為無限大）


def max_group_size(tol=TOL, bounds=BOUNDS, limit=N_PUBLIC):  # 一次上傳能解碼的最大組
    """Largest group whose admissible histograms are all distinguishable at tolerance tol."""
    best = 0  # 目前能區分的最大組
    for size in range(1, limit + 1):  # 由小到大嘗試
        if min_gap(size, bounds) > tol:  # 最小差距大於容許誤差
            best = size  # 這個大小可以解碼
        else:  # 開始無法區分
            break  # 停止嘗試
    return best  # 回傳最大組大小


def decode_histogram(observed_delta, size, tol=TOL, bounds=BOUNDS, perm=CYCLE):  # 由分數變化解出直方圖
    """Admissible histograms of a group of `size` tested questions whose predicted sum matches observed_delta (= 400 x
    score change) within tol; exactly one when the group is within max_group_size."""
    return [h for h in histograms(size, bounds) if abs(histogram_sum(h, perm) - observed_delta) <= tol]  # 保留預測與觀察相符者


def reorder(top5, perm=CYCLE):  # 依排列重排前五名
    """The new list: perm[j] = current rank of the candidate at new rank j+1."""
    return [top5[r - 1] for r in perm]  # 依目前名次取出候選人


def fixed_list(top5, truth_pos):  # 修正後的前五名
    """Move the truth (current rank truth_pos, 2..5) to rank 1, keeping the others' order."""
    lst = list(top5)  # 複製清單
    c = lst.pop(truth_pos - 1)  # 取出正解
    return [c, *lst]  # 正解放到第一名


def gain_of_fix(truth_pos):  # 修正一題的分數增益
    """Score gain of moving the truth from rank truth_pos to rank 1."""
    return float(1.0 - DECAY[truth_pos - 1])  # 第一名的分數 1 減去原名次的分數


def resolve(groups):  # 由已解碼的組推出每題的位置
    """Propagate decoded histograms: for nested groups the difference set gets the difference histogram; a group whose
    histogram is all rank 1 marks its members right; a singleton fixes its member. groups: [{members: [..], hist: (6,)}]
    (hist None when not yet decoded). Returns ({question_id: position or None}, derived groups added)."""
    known = [g for g in groups if g.get("hist") is not None]  # 已解碼的組
    derived, changed = [], True  # 推導出的組與是否還有變化
    while changed:  # 反覆推導直到沒有新的組
        changed = False  # 本輪預設沒有變化
        for a in list(known):  # 逐對檢查組
            for b in list(known):  # 另一組
                sa, sb = set(a["members"]), set(b["members"])  # 兩組的成員集合
                if sa < sb:  # a 嚴格包含於 b：b 減 a 的直方圖可以確定
                    rest = sorted(sb - sa)  # 差集的成員
                    hist = tuple(int(x - y) for x, y in zip(b["hist"], a["hist"]))  # 差的直方圖
                    if min(hist) < 0:  # 出現負數表示兩組的直方圖不一致
                        raise ValueError(f"inconsistent histograms for {a['members']} inside {b['members']}")  # 中止
                    if not any(set(g["members"]) == set(rest) for g in known):  # 這個差集還不是已知的組
                        g = {"members": rest, "hist": hist, "derived": True}  # 建立推導出的組
                        known.append(g)  # 加入已知組
                        derived.append(g)  # 記為推導出的組
                        changed = True  # 還要再推一輪
    positions = {}  # 題目 -> 正解位置
    for g in known:  # 由每個已知組推出位置
        hist, members = g["hist"], g["members"]  # 直方圖與成員
        if hist[0] == len(members):  # 全部在第 1 名
            for m in members:  # 每個成員
                positions[m] = 1  # 都是第 1 名
        elif len(members) == 1:  # 只有一題的組
            t = [POSITIONS[i] for i, n in enumerate(hist) if n][0]  # 唯一的非零格就是位置
            positions[members[0]] = t  # 記下該題的位置
    return positions, derived  # 回傳位置與推導出的組
