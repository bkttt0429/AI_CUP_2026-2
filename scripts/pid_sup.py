#!/usr/bin/env python
"""Supervised per-ply features (METHOD_NOTES 2.88, probe B).

The stored per-ply input of the sequence networks is the humanv0 trunk vector pushed through a RANDOM projection. A
supervised projection (the directions of the per-game means that separate training players best) keeps the identity
information in 32-64 dimensions. Here:
  fit(...)      the projection from per-game rows of unprojected statistics (scripts/pid_trunk_extract.py --full)
  load_sup(...) the per-ply parts of scripts/pid_sup_ply_extract.py, standardised per dimension over all their plies
  dense(...)    a padded [games, 256, K] array aligned with a list of game keys (zeros = the mean where a ply is not the
                subject's, beyond the game, or the game has no part)
  put(...)      writes the features into chosen input dimensions of a batch, in the units the network standardises
"""
import glob  # 列出各資料夾的分段檔

import numpy as np  # 陣列運算

C = 384  # 主幹向量（未投影）的通道數


def fit(rows, owners, k=64, ridge=1e-3):  # 由每局統計列求有監督投影
    """rows [G, 4C] (`full`: mean / std of the pooled vector, mean / std of the played-point vector), owners [G].
    Returns mu [2C], sd [2C], W [2C, k]: feature = ((x - mu) / sd) @ W for x = [pooled, played-point] (per game mean or per ply)."""
    from scipy.linalg import eigh  # 廣義特徵值分解
    X = rows[:, np.r_[0:C, 2 * C:3 * C]].astype(np.float64)  # 只取全盤向量與落點向量的平均兩個區塊（不用標準差區塊）
    mu, sd = X.mean(0), X.std(0) + 1e-6  # 標準化用的逐維度平均與標準差（加 1e-6 防止除以零）
    X = (X - mu) / sd  # 逐維度標準化
    _pl, inv = np.unique(owners, return_inverse=True)  # 每局所屬玩家的索引
    cnt = np.bincount(inv).astype(float)  # 每位玩家的局數
    cm = np.stack([np.bincount(inv, weights=X[:, j]) for j in range(X.shape[1])], axis=1) / cnt[:, None]  # 每位玩家的平均
    resid = X - cm[inv]  # 玩家內的偏差
    Sw = resid.T @ resid / max(len(X) - len(cnt), 1)  # 玩家內共變異
    Sb = np.cov(cm.T)  # 玩家間共變異
    lam, W = eigh(Sb, Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))  # 玩家間對玩家內變異比的廣義特徵向量（加脊正則）
    return mu.astype(np.float32), sd.astype(np.float32), W[:, ::-1][:, :k].astype(np.float32), lam[::-1][:k]  # 取變異比最大的 k 個方向與其比值


def fit_means(X, owners, k=64, ridge=1e-3):  # 任意多個每局平均區塊的有監督投影
    """fit for any stack of per-game mean blocks: X [G, d] (e.g. pooled, played-point and policy-expected vectors, METHOD_NOTES
    2.102). Returns mu [d], sd [d], W [d, k], ratios [k]: feature = ((x - mu) / sd) @ W for the same stack per game or per ply."""
    from scipy.linalg import eigh  # 廣義特徵值分解
    from scipy.sparse import csr_matrix  # 以稀疏指示矩陣求各玩家平均
    X = X.astype(np.float64)  # 以倍精度計算
    mu, sd = X.mean(0), X.std(0) + 1e-6  # 標準化用的逐維度平均與標準差（加 1e-6 防止除以零）
    X = (X - mu) / sd  # 逐維度標準化
    _pl, inv = np.unique(owners, return_inverse=True)  # 每局所屬玩家的索引
    cnt = np.bincount(inv).astype(float)  # 每位玩家的局數
    cm = np.asarray(csr_matrix((np.ones(len(X)), (inv, np.arange(len(X)))), shape=(len(cnt), len(X))) @ X) / cnt[:, None]  # 玩家 × 局的指示矩陣乘 X，得每位玩家的平均
    resid = X - cm[inv]  # 玩家內的偏差
    Sw = resid.T @ resid / max(len(X) - len(cnt), 1)  # 玩家內共變異
    lam, W = eigh(np.cov(cm.T), Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw)))  # 玩家間對玩家內變異比的廣義特徵向量（加脊正則）
    return mu.astype(np.float32), sd.astype(np.float32), W[:, ::-1][:, :k].astype(np.float32), lam[::-1][:k]  # 取變異比最大的 k 個方向與其比值


def load_sup(dirs, stats=None):  # 讀入有監督逐手特徵的分段檔
    """(index {key: (start, n)}, t [N] int16 ply numbers, Z [N, K] float16 standardised) over every part of the directories.
    Parts hold `z` (float16, unstandardised) or `zq` (int8: the standardised value x 20, written with a stats file). With
    `stats` (npz with mean, sd per dimension) the float parts are standardised by it, so that several stores share one scale;
    without it by their own mean and sd over all plies (the first probe)."""
    keys, ns, ts, zs, kinds = [], [], [], [], []  # 棋局鍵、每局存的手數（主體各手）、手數編號、特徵、分段檔種類
    for d in dirs:  # 每個特徵資料夾
        for f in sorted(glob.glob(f"{d}/*.npz")):  # 每個分段檔
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀入分段檔
            keys.append(z["keys"].astype(str)); ns.append(z["n"].astype(np.int64)); ts.append(z["t"])  # 收集棋局鍵、每局手數與手數編號
            if "zq" in z.files:  # int8 量化的分段檔
                zs.append(z["zq"]); kinds.append("q")  # 記為量化種類
            else:  # float16 的分段檔
                zs.append(z["z"]); kinds.append("f")  # 記為浮點種類
    if not keys:  # 一個分段檔都沒有
        raise FileNotFoundError(f"no supervised per-ply parts in {dirs}")  # 直接中止
    if stats is None and "q" in kinds:  # 量化檔需要當初寫出時的尺度
        raise ValueError("int8 parts need the stats file they were written with")  # 缺尺度檔就中止
    if stats is not None:  # 有共同尺度檔
        st = np.load(stats)  # 讀入尺度檔
        m, sd = st["mean"].astype(np.float32), st["sd"].astype(np.float32)  # 逐維度平均與標準差
    else:  # 沒有尺度檔
        allf = np.concatenate(zs).astype(np.float32)  # 全部手的特徵
        m, sd = allf.mean(0), allf.std(0) + 1e-6  # 用它們自己的統計
        del allf  # 釋放暫時的大陣列
    Z = np.concatenate([(x.astype(np.float32) / 20.0 if k == "q" else (x.astype(np.float32) - m) / sd).astype(np.float16) for x, k in zip(zs, kinds)])  # 量化檔除以 20 還原，浮點檔標準化
    keys, n, t = np.concatenate(keys), np.concatenate(ns), np.concatenate(ts)  # 合併各分段檔
    start = np.concatenate([[0], np.cumsum(n)[:-1]])  # 每局在 Z 中的起始列
    return {k: (int(s_), int(m_)) for k, s_, m_ in zip(keys.tolist(), start.tolist(), n.tolist())}, t, Z  # 棋局鍵 -> (起始列, 手數)，以及手數編號與特徵


def write_stats(dirs, out):  # 計算並寫出多個儲存共用的尺度檔
    """Mean and sd per dimension over all plies of the float parts of the directories -> npz (mean, sd)."""
    zs = []  # 收集浮點特徵
    for d in dirs:  # 每個資料夾
        for f in sorted(glob.glob(f"{d}/*.npz")):  # 每個分段檔
            if not f.endswith(".tmp.npz"):  # 略過寫到一半的暫存檔
                z = np.load(f)  # 讀入分段檔
                if "z" in z.files:  # 只用浮點分段檔
                    zs.append(z["z"].astype(np.float32))  # 收集其特徵
    allf = np.concatenate(zs)  # 全部手的特徵
    np.savez(out, mean=allf.mean(0), sd=allf.std(0) + 1e-6, plies=np.array(len(allf)))  # 寫出逐維度尺度與手數
    return len(allf)  # 回傳統計用到的手數


def dense(sup, keys, T=256, alloc=None):  # 依棋局鍵排成補齊的特徵陣列
    """idx [len(keys)] int32 (row of D; -1: the game has no supervised features), D [n_have, T, K] float16."""
    index, t, Z = sup  # load_sup 的結果
    have = [i for i, k in enumerate(keys) if k in index]  # 有特徵的棋局
    idx = np.full(len(keys), -1, np.int32)  # 預設為沒有特徵
    idx[have] = np.arange(len(have), dtype=np.int32)  # 有特徵的棋局依序編列
    shape = (max(len(have), 1), T, Z.shape[1])  # 至少一列，避免空陣列
    D = alloc(shape, np.float16) if alloc is not None else np.zeros(shape, np.float16)  # 可由呼叫端配置（例如共享記憶體）
    for j, i in enumerate(have):  # 逐局填入
        s, m = index[keys[i]]  # 該局的起始列與手數
        tt = t[s:s + m].astype(np.int64)  # 這些手的手數編號
        ok = tt < T  # 只留前 T 手
        D[j, tt[ok]] = Z[s:s + m][ok]  # 放到對應手的位置，其餘保持零
    return idx, D  # 回傳列索引與補齊陣列


def put(gl, z, mu, sd, dims, only=False):  # 把有監督特徵寫進輸入維度
    """gl [B, T, 128] and z [B, T, K] (torch tensors or numpy arrays of the same kind): the supervised features in the input
    dimensions dims = (d0, d1), scaled so that the network's own standardisation ((x - mu) / sd) gives z back; with
    only=True every other dimension is set to its mean (the network then sees nothing of the random projection)."""
    d0, d1 = dims  # 起訖維度
    assert d1 - d0 == z.shape[-1], (dims, z.shape)  # 維度數須等於特徵數
    if only:  # 只讓網路看有監督特徵
        gl = gl * 0 + mu  # 全部維度設為平均（標準化後為零）
    gl[..., d0:d1] = z * sd[d0:d1] + mu[d0:d1]  # 反標準化後寫入指定維度
    return gl  # 回傳寫好的輸入（only=False 時是就地修改傳入的 gl）
