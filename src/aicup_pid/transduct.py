"""Transductive self-training on a closed candidate pool (fork plan, robustness line).

After a first pass (fused scores, optionally the one-to-one assignment), the questions whose winning margin is at least
`tau` are treated as solved: their games are added to the matched candidate's library, the component models are
refitted on the enlarged libraries and every question is rescored with the same fusion weights. The pool is closed, so
every confident answer also sharpens the models the unconfident questions are compared against. Each round rebuilds the
libraries from the original candidate games plus the currently confident questions (no accumulation of stale answers).
"""
import numpy as np  # 分數矩陣與陣列運算

from aicup_pid.assign import assign_topk  # 一對一配對後的 Top-k
from aicup_pid.components import fit_from_parsed, fit_local, score_components  # 在擴大的棋譜庫上重新擬合並計分各成分模型
from aicup_pid.fusion import Fusion, topk  # 融合權重與逐題 Top-k

THREE = ("opening", "style", "local")  # fusion3 的三個成分：開局、風格、局部棋形


def margins_of(S_f, top1):  # 計算每題的勝出差距
    """Winning margin of each question: its Top-1's score minus the best other candidate's ([Q])."""
    S = np.asarray(S_f, dtype=float)  # 融合分數轉成浮點陣列
    top1 = np.asarray(top1)  # 每題 Top-1 的候選人索引
    if S.shape[0] == 0:  # 沒有題目時
        return np.zeros(0)  # 回傳空陣列
    rows = np.arange(len(top1))  # 每題的列索引
    best = S[rows, top1]  # 取出 Top-1 的分數
    masked = S.copy()  # 複製一份，避免改到傳入的分數
    masked[rows, top1] = -np.inf  # 遮掉 Top-1，只留其他候選人
    return best - masked.max(axis=1)  # Top-1 減去其餘最高分；配對換掉 argmax 時可能為負


def fuse(S, weights, comps=THREE):  # 用給定的融合權重合併各成分
    fus = Fusion()  # 只借用 Fusion 的計分方法，不呼叫 fit
    fus.w = np.asarray(weights, dtype=float)  # 直接套用權重，不重新擬合
    return fus.scores([S[c] for c in comps])  # 依成分順序逐列標準化後加權相加


def self_train(by_player, moves, feats, cfg, q_keys, S, weights, tau=3.0, rounds=1, assign=True, comps=THREE, k=5):  # 轉導式自我訓練的主流程
    """Self-training from the component scores S of the questions q_keys against the libraries by_player.

    Returns (Top-k indices [Q, k], final fused scores [Q, C], info) with info["rounds"] = per round {confident, games_added}
    and info["confident"] = the last round's confident question indices (for the precision of the confident set)."""
    players = sorted(by_player)  # 候選人依編號排序，對應分數矩陣的欄
    S_f = fuse(S, weights, comps)  # 自我訓練前第一遍的融合分數
    idx = assign_topk(S_f, k) if assign else topk(S_f, k)  # 第一遍的 Top-k：一對一配對或逐題排序
    history, confident = [], np.zeros(0, dtype=int)  # 各輪紀錄；rounds 為 0 時回傳空的有把握清單
    for r in range(rounds):  # 逐輪擴充棋譜庫並重新計分
        confident = np.flatnonzero(margins_of(S_f, idx[:, 0]) >= tau)  # 勝出差距達 tau 的題目視為已解
        history.append({"round": r + 1, "confident": int(len(confident)), "games_added": int(sum(len(q_keys[q]) for q in confident)),  # 記錄本輪有把握的題數與加入的局數
                        "margin_threshold": float(tau)})  # 以及本輪使用的門檻
        if len(confident) == 0:  # 沒有任何有把握的題目
            break  # 提前結束，保留目前的排序
        library = {p: list(keys) for p, keys in by_player.items()}  # 每輪都從原始候選人棋局重建，不累積舊答案
        for q in confident:  # 逐一處理有把握的題目
            library[players[idx[q, 0]]] = library[players[idx[q, 0]]] + list(q_keys[q])  # 題目棋局併入其 Top-1 候選人的棋譜庫
        opening, style = fit_from_parsed(library, moves, feats, cfg)  # 在擴大的棋譜庫上重新擬合開局與風格模型
        local = fit_local(library, moves, cfg) if "local" in comps else None  # 成分含局部棋形時一併重新擬合
        S_new = score_components(comps, opening, style, q_keys, moves, feats, local)  # 全部題目用新模型重新計分
        S_f = fuse(S_new, weights, comps)  # 以同一組權重重新融合
        idx = assign_topk(S_f, k) if assign else topk(S_f, k)  # 重新取 Top-k（需要時再配對）
    return idx, S_f, {"rounds": history, "players": players, "confident": confident.tolist()}  # 回傳 Top-k、最終融合分數，以及各輪紀錄、候選人順序與最後一輪的有把握題目
