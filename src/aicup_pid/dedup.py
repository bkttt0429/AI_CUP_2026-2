"""Whole-game duplicate detection (fork plan section 3, checks 1-2).

game_hash puts a game into the canonical orientation of aicup_pid.opening.canonical over all of its moves (so a game and
its 8 symmetric images hash alike), encodes every move as 3 bytes (colour B=0 / W=1, then the point x*19+y big-endian with
pass = 380) and takes a 16-byte blake2b. build_training_index streams the 10 train_<rank>.csv files in chunks, hashes
them in a fork pool and keeps, per game, (player_id, rank, game_key, color, n_moves, n_black) plus hash -> rows.
"""
import hashlib  # blake2b 雜湊
import multiprocessing as mp  # 平行計算雜湊
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列
import pandas as pd  # 分塊讀取訓練 CSV

from aicup.metric import LEVELS  # 10 個棋力等級（對應訓練檔名）
from aicup.sgf import parse_moves  # 解析 SGF 落子
from aicup_pid.opening import BOARD, canonical  # 棋盤大小與標準方向

PASS_CODE = BOARD * BOARD + BOARD  # 380：在每種對稱下都是 (19, 19)
COLUMNS = ["player_id", "rank", "game_key", "color", "n_moves", "n_black"]  # 索引中每局保留的欄位
USECOLS = ["player_id", "game_id", "rank", "color", "sgf_content"]  # 從訓練 CSV 讀取的欄位


def game_hash(moves):  # 不受棋盤對稱影響的整局雜湊
    """16 bytes identifying the game up to the 8 board symmetries (colours kept)."""
    if not moves:  # 沒有落子
        return hashlib.blake2b(b"", digest_size=16).digest()  # 空內容的雜湊
    cs, _ = canonical(moves, len(moves))  # 以整局決定標準方向後的座標
    buf = bytearray()  # 編碼緩衝區
    for (c, *_), (x, y) in zip(moves, cs):  # 逐手編碼
        code = PASS_CODE if x == BOARD else x * BOARD + y  # pass 用固定代碼，其餘為點位編號
        buf += bytes((0 if c == "B" else 1, code >> 8, code & 255))  # 顏色 1 位元組＋點位 2 位元組（大端序）
    return hashlib.blake2b(bytes(buf), digest_size=16).digest()  # 16 位元組的 blake2b


def _hash_one(sgf):  # 一局棋的雜湊與長度（供行程池呼叫）
    moves = parse_moves(sgf)  # 解析落子
    if not moves:  # 沒有落子的棋譜
        return None, 0, 0  # 沒有雜湊，長度為 0
    return game_hash(moves), len(moves), sum(1 for c, *_ in moves if c == "B")  # 雜湊、總手數、黑棋手數


def hash_games(sgfs, workers=1):  # 計算多局棋譜的雜湊
    """{key: (hash or None, n_moves, n_black)}; a record without moves gives (None, 0, 0)."""
    keys = list(sgfs)  # 固定鍵的順序
    texts = [sgfs[k] for k in keys]  # 對應的棋譜文字
    if workers > 1 and len(texts) > 1:  # 多行程且不只一局
        with mp.get_context("fork").Pool(workers) as pool:  # 以 fork 建立行程池
            res = pool.map(_hash_one, texts, chunksize=256)  # 平行計算
    else:  # 單一行程
        res = [_hash_one(s) for s in texts]  # 逐局計算
    return dict(zip(keys, res))  # 鍵對應結果


class TrainingIndex:  # 訓練集的重複查詢索引
    """meta: DataFrame with COLUMNS, one row per training game; by_hash: {hash: [row numbers]}."""

    def __init__(self, meta, by_hash, seconds=0.0):  # 保存索引內容
        self.meta, self.by_hash, self.seconds = meta, by_hash, seconds  # 每局資訊、雜湊對照、建立耗時

    def lookup(self, h):  # 查詢一個雜湊
        return list(self.by_hash.get(h, ()))  # 雜湊相同的列號（沒有則為空清單）

    def black_lengths(self):  # 執黑棋局的長度
        """(n_moves, n_black) arrays of the Black games."""
        m = self.meta[self.meta["color"] == "B"]  # 只取執黑的棋局
        return m["n_moves"].to_numpy(dtype=int), m["n_black"].to_numpy(dtype=int)  # 總手數與黑棋手數


def build_training_index(train_dir, workers=1, chunksize=20000, max_games=None, log=print):  # 建立訓練集的雜湊索引
    """Hash every game of train_<rank>.csv (LEVELS order, files that exist) in chunks; max_games stops early (smoke tests)."""
    t0 = time.time()  # 記下開始時間，最後回報建立索引的耗時
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None  # 多行程時建立 fork 行程池
    frames, by_hash, n = [], {}, 0  # 各塊的資訊表、雜湊對照、已處理的局數
    try:  # 確保最後會關閉行程池
        for r in LEVELS:  # 依等級逐檔處理
            path = Path(train_dir) / f"train_{r}.csv"  # 該等級的訓練檔
            if not path.exists():  # 檔案不存在
                continue  # 略過
            t1, n_file = time.time(), 0  # 該檔的計時與局數
            for chunk in pd.read_csv(path, usecols=USECOLS, dtype=str, keep_default_na=False, chunksize=chunksize):  # 分塊讀取
                if max_games is not None:  # 有局數上限時
                    chunk = chunk.iloc[:max(max_games - n, 0)]  # 截到剩餘的額度
                    if len(chunk) == 0:  # 額度已用完
                        break  # 停止讀這個檔
                texts = chunk["sgf_content"].tolist()  # 這一塊的棋譜文字
                res = pool.map(_hash_one, texts, chunksize=256) if pool else [_hash_one(s) for s in texts]  # 計算雜湊與長度
                frames.append(pd.DataFrame({"player_id": chunk["player_id"].to_numpy(), "rank": chunk["rank"].to_numpy(),  # 這一塊每局的資訊：玩家與等級
                                            "game_key": (chunk["rank"] + ":" + chunk["game_id"]).to_numpy(), "color": chunk["color"].to_numpy(),  # 棋局鍵 '等級:game_id' 與執色
                                            "n_moves": np.array([m for _, m, _ in res], dtype=np.int32),  # 總手數
                                            "n_black": np.array([b for _, _, b in res], dtype=np.int32)}))  # 黑棋手數
                for i, (h, _, _) in enumerate(res):  # 登記每局的雜湊
                    if h is not None:  # 有落子的棋局才有雜湊
                        by_hash.setdefault(h, []).append(n + i)  # 雜湊對應到全域列號
                n += len(chunk)  # 累計總局數
                n_file += len(chunk)  # 累計該檔的局數
                if max_games is not None and n >= max_games:  # 達到上限
                    break  # 停止讀這個檔
            log(f"training index: {path.name} {n_file} games in {time.time() - t1:.0f} s ({n} so far)", flush=True)  # 回報進度
            if max_games is not None and n >= max_games:  # 達到上限
                break  # 不再讀其他檔
    finally:  # 不論成功與否
        if pool is not None:  # 有建立行程池時
            pool.close()  # 停止接收工作
            pool.join()  # 等待行程結束
    meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)  # 合併所有塊（沒有資料時為空表）
    return TrainingIndex(meta[COLUMNS], by_hash, time.time() - t0)  # 回傳索引與耗時
