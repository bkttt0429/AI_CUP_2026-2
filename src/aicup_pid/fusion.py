"""Fuse per-candidate log scores of several components into Top-5 lists (docs/records/ARCHITECTURE.md ladder (5)).

Each component gives a [Q, C] matrix of scores (questions x candidates). standardise centres and scales every row, so
components live on one scale; Fusion learns one weight per component by maximising the multinomial log-likelihood of
the true candidate under softmax(sum_j w_j z_j) (L-BFGS in float64, tiny L2); topk returns the indices of the best k.
"""
import numpy as np  # 陣列運算
import torch  # 以 L-BFGS 擬合權重


def standardise(S):  # 逐列標準化分數
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    mu = S.mean(axis=1, keepdims=True)  # 每列的平均
    sd = S.std(axis=1, keepdims=True)  # 每列的標準差
    sd[sd < 1e-12] = 1.0  # 常數列不縮放，避免除以 0
    return (S - mu) / sd  # 回傳標準化後的分數


def topk(S, k=5):  # 每題分數最高的 k 位候選人
    """[Q, k] candidate indices with the highest scores, best first."""
    S = np.asarray(S, dtype=float)  # 輸入可為清單，統一成浮點陣列
    k = min(k, S.shape[1])  # 候選人不足 k 位時取全部
    idx = np.argsort(-S, axis=1, kind="stable")[:, :k]  # 穩定排序，同分時保留原順序
    return idx  # 回傳索引


class Fusion:  # 各成分的線性融合
    """weights over components; fit(components [J x (Q, C)], truth [Q]) ; scores(components) -> [Q, C]."""

    def __init__(self, l2=1e-3, max_iter=200):  # 設定正則化與迭代上限
        self.l2, self.max_iter = l2, max_iter  # L2 係數與 L-BFGS 的最大迭代數
        self.w = None  # 權重：由 fit 擬合，或由融合文件直接指定

    def fit(self, components, truth):  # 以真實答案擬合權重
        Z = torch.tensor(np.stack([standardise(S) for S in components]), dtype=torch.float64)  # [J, Q, C]：成分 x 題目 x 候選人
        y = torch.tensor(np.asarray(truth, dtype=np.int64))  # 每題正解的候選人索引
        w = torch.ones(Z.shape[0], dtype=torch.float64, requires_grad=True)  # 權重初值皆為 1
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=self.max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)  # L-BFGS 最佳化器

        def closure():  # L-BFGS 每次需要重新計算損失
            opt.zero_grad()  # 清除梯度
            logits = torch.einsum("j,jqc->qc", w, Z)  # 各成分加權相加
            loss = torch.nn.functional.cross_entropy(logits, y) + self.l2 * (w ** 2).sum()  # 多項式負對數概似加上 L2
            loss.backward()  # 反向傳遞
            return loss  # 回傳損失

        opt.step(closure)  # 執行最佳化
        self.w = w.detach().numpy().copy()  # 存下權重
        return self  # 回傳自身以便串接

    def scores(self, components):  # 計算融合分數
        if self.w is None:  # 尚未擬合或設定權重
            raise ValueError("fit first")  # 要求先擬合
        return np.einsum("j,jqc->qc", self.w, np.stack([standardise(S) for S in components]))  # 各成分標準化後加權相加

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"weights": self.w.tolist(), "l2": self.l2}  # 權重與 L2 係數
