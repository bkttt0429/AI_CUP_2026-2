#!/usr/bin/env python
"""A sequence identity network trained for players it has NOT seen (METHOD_NOTES 2.39). Architecture, data and episode
objective are the rank session's M90 (AI_CUP_2026-1 scripts/m_series/m88_seqid.py --full; that module is imported read-only for
the model and the part readers). M90's accuracy on unseen players plateaus at 10-30k steps and falls afterwards while the
training players keep being learnt; the official players are all unseen. Changes offered here:
  --steps / --horizon   steps trained / the step where the cosine reaches zero (M90: 60,000; a 20k snapshot = --steps 20000
                        --horizon 60000)
  --drop                dropout (M90 0.1)
  --mask                ply masking in training: per game a share drawn from U(0, mask) of its plies has the trunk vector
                        replaced by the mean vector (zero after the model's standardisation)
  --ema                 decay of an exponential moving average of the weights, saved next to the raw state
Folds: net v trains on the players of folds 1-4 other than v; net -1 on folds 1-4. Fold 0 is not read into training or
evaluation (this repository never uses it); players without a fold train in every net, as in M90. The rank competition's
test games are not read.
  --fork <spec file>         several networks of one job on one shared copy of the games (see main_fork)
  --bank / --margin          cross-batch memory negatives, cosine margin (METHOD_NOTES 2.64)
  --cls-loss cosface         classification over all training players with a cosine margin (METHOD_NOTES 2.56)
  --rank-ep / --rank-ep-share   episodes of players of one rank class or band: hard negatives (METHOD_NOTES 2.55)
  --snap                     further checkpoints at given steps of one long run (METHOD_NOTES 2.51)
  --ext-ply / --ext-select   external players (METHOD_NOTES 2.46): per-ply parts of scripts/pid_ext_ply_extract.py and the
                        select.csv of scripts/ext_fox_select.py (game_key -> account). They train in every net (no fold),
                        are never in the held-out log, and fill --ext-share of every episode's players (default: drawn
                        like any other player).
  --sym-aug             every episode is seen under one random board symmetry (the played points are remapped; the trunk
                        vectors are kept): the network cannot tie a player to absolute corners, only to what is consistent
                        between the games of one episode (METHOD_NOTES 2.48)
  --p-ep                players per episode (M90: 256)
  --ext-share-end       the external share falls linearly from --ext-share to this value over the run (finish on the
                        competition's own players)
  --no-state            no resume file (saves 0.5 GB per network while it trains)
  --all-folds           fold 0 is read and trained on as well: net -1 then sees every training player, as the rank session's
                        net -1 (METHOD_NOTES 2.40). ONLY WITH THE USER'S CONSENT for fold 0; for official files only (no
                        simulation can score such a network). Held-out own games (results/m82/held_keys.json) are never trained on; for net v the ones of fold v
(unseen players) and of the training folds (seen players) are scored every --eval-every steps, as a log only: nothing is
selected on them (fixed step).
Outputs in --out: ckpt_<v>_<tag>.pt (raw state at the last step) and ckpt_<v>_<tag>e.pt (EMA state), both with the state
under "best_state" so that pid_v2_seq_embed.py / pid_v2_seq_finetune.py load them with --ckpt-suffix _<tag> / _<tag>e;
train_<v>_<tag>.json (history); state_<v>_<tag>.pt (resume point, removed at the end).
Run from the AI_CUP_2026-1 checkout: python -I <this> --net 1 --tag r1 ...
"""
import argparse  # 解析命令列參數
import copy  # 深複製網路以建立 EMA 副本
import json  # 讀留出清單、寫訓練歷程
import math  # 餘弦學習率排程
import multiprocessing as mp  # 以 fork 的工作行程平行讀取分段檔
import os  # fork、檔案描述子與系統資訊
import sys  # 調整模組搜尋路徑與結束碼
import time  # 計時與記錄經過秒數
from pathlib import Path  # 路徑操作

import numpy as np  # 陣列運算
import pandas as pd  # 讀取玩家分折表與訓練集 CSV

AICUP1 = Path.cwd()  # 須從 AI_CUP_2026-1 的根目錄執行
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "scripts/m_series")]  # 讓 m88_seqid 與 o26_parallel_forward 可被匯入
import m88_seqid as M  # noqa: E402  棋力 session 的 M90 模型與分段檔讀取函式（唯讀匯入）

T = M.T_MAX  # 每局最多讀前 256 手
PROTO_GAMES = 16  # 評估用原型：每位玩家最多取這麼多局訓練棋局（固定的選擇）


def _windows(pool, fn, files, w):  # 分批讀取分段檔，限制同時等待的數量
    """The parts in order, at most w of them waiting in this process at a time. With pool.imap the workers ran ahead of the copy
    into the big arrays and the waiting parts piled up: the processes of 2026-11-13 held 2.3 times the size of their arrays."""
    for i in range(0, len(files), w):  # 每次取 w 個檔案
        yield from pool.map(fn, files[i:i + w], chunksize=1)  # 這一批讀完才交出，避免讀好的分段檔在記憶體中堆積


def _trim():  # 把已釋放的記憶體還給系統
    try:  # 非 glibc 環境可能沒有 libc.so.6
        import ctypes  # 呼叫 C 函式庫
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # 把已釋放的分段檔緩衝區還給系統
    except OSError:  # 找不到 libc 時
        pass  # 略過，不影響結果


def _rss_gb():  # 目前行程的常駐記憶體（GB）
    with open("/proc/self/statm") as fh:  # Linux 的行程記憶體統計
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9  # 第二欄為常駐頁數，乘以頁大小


SHARED = ("GL", "POS", "MOV", "LEN", "SUBJ")  # 放在共享記憶體、各網路共用的大陣列
SHARED_ANON = False  # 由 main_fork 設定：大陣列放在匿名共享記憶體
PRELOADED = None  # 由 main_fork 在每個 fork 出的訓練行程設定：父行程讀過一次的棋局
_KEEP = []  # 保留匿名 mmap 物件的參照，避免被回收


def load_games(smoke, log, all_folds=False, ext_ply=None, ext_select=None, share_dir=None, ext_second=None, sup_ply=None, sup_stats=None, ext_deep=None):  # 讀入全部訓練棋局
    """M90's training games as padded arrays (m88_seqid.load_games without the test part). The worker pool starts before any
    large array exists and every part is written straight into preallocated arrays: the compute nodes run with strict
    overcommit, and a pool forked from a large parent exhausted the node's commit limit in the rank session's first run."""
    share = Path(share_dir) if share_dir else None  # 共享記憶體資料夾（沒指定則不用）
    if share is not None and (share / "DONE").exists():  # 每個工作只在共享記憶體準備一次陣列（experiments/nano4/pid_seq_train_shared.sh）
        G = {k: np.load(share / f"{k}.npy", mmap_mode="r") for k in SHARED}  # 以唯讀記憶體映射接上大陣列
        G.update({k: np.load(share / f"{k}.npy") for k in ("key", "pid", "fold")})  # 小陣列直接讀入
        log(f"games attached from {share}: {len(G['key'])} ({G['GL'].nbytes / 1e9:.1f} GB shared; process memory {_rss_gb():.1f} GB); games of fold 0: {int((G['fold'] == 0).sum())}")  # 記錄接上的局數與記憶體
        return G  # 已準備好，不必重讀
    if share is not None:  # 由本行程建立共享陣列
        share.mkdir(parents=True, exist_ok=True)  # 先建立資料夾

    def alloc(name, shape, dtype, fill=0):  # 有要求時放在共享記憶體：該工作的每個網路映射同一批記憶體頁
        if share is None and SHARED_ANON:  # 匿名共享映射：fork 出的訓練行程（main_fork）讀同一批頁面，只計一次
            import mmap  # 建立匿名共享記憶體
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize  # 陣列所需位元組數
            mm = mmap.mmap(-1, max(nbytes, 1), flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)  # fork 後父子行程共用的可讀寫映射
            _KEEP.append(mm)  # 保留參照，避免映射被釋放
            arr = np.frombuffer(mm, dtype=dtype, count=int(np.prod(shape))).reshape(shape)  # 在映射上建立 numpy 陣列
            if fill != 0:  # 需要非零初值時
                arr[:] = fill  # 填入初值
            return arr  # 回傳共享陣列
        if share is None:  # 不共享：一般記憶體
            return np.zeros(shape, dtype) if fill == 0 else np.full(shape, fill, dtype)  # 依初值配置
        arr = np.lib.format.open_memmap(share / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)  # 在共享資料夾建立 .npy 映射檔
        if fill != 0:  # 需要非零初值時
            arr[:] = fill  # 填入初值
        return arr  # 回傳映射陣列

    files, folds = [], []  # 已存分段檔的路徑與各自的折
    for d in M.DIRS:  # 棋力 session 已存的三個主幹向量資料夾
        for f in range(0 if all_folds else 1, 5):  # 除非 --all-folds，否則不開第 0 折的已存分段檔
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv"))  # 該折的分段檔清單
            fs = fs[:12] if smoke else fs  # 冒煙測試只取前 12 個
            files += fs  # 加入要讀的分段檔清單
            folds += [f] * len(fs)  # 記下每個檔案的折
    files2 = [f for k in (0, 1, 2) for f in sorted((AICUP1 / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]  # M90 另外抽取的逐手分段檔（略過暫存檔）
    files2 = files2[:20] if smoke else files2  # 冒煙測試只取前 20 個
    ext_own = {}  # 外部棋局鍵 -> 帳號（加上 x: 前綴）
    if ext_ply:  # 與 M90 相同的分段檔格式（key、LEN、SUBJ、POS、MOV、GLQ；相同的量化統計）
        empty = [str(d) for d in ext_ply if not any(Path(d).glob("ply_x_*.npz"))]  # 找出沒有分段檔的外部資料夾
        if empty:  # 缺資料夾以前會默默少讀資料（METHOD_NOTES 2.148）
            raise FileNotFoundError(f"external ply folders without parts: {empty}")  # 直接中止
        fx = sorted(f for d in ext_ply for f in Path(d).glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz"))  # 全部外部分段檔（略過暫存檔）
        files2 += fx[:5] if smoke else fx  # 接在 M90 分段檔之後；冒煙測試只取 5 個
        import csv  # 讀取 select.csv
        csv.field_size_limit(10 ** 9)  # 欄位含整份棋譜，放寬長度上限
        for sel in ext_select:  # 每個挑選檔
            with open(sel, newline="") as fh:  # 開啟 CSV
                ext_own.update((r["game_key"], "x:" + r["account"]) for r in csv.DictReader(fh))  # 棋局鍵對應到外部帳號
    deep_keys = set()  # METHOD_NOTES 2.126：為先前已選帳號再加更多棋局的挑選（也列在 --ext-select）；訓練時可用 --no-deep 排除
    if ext_ply and ext_deep:  # 有指定較深的挑選
        for sel in ext_deep:  # 每個較深的挑選檔
            with open(sel, newline="") as fh:  # 開啟 CSV
                deep_keys.update(r["game_key"] for r in csv.DictReader(fh))  # 收集其棋局鍵
    sec = {}  # METHOD_NOTES 2.73：外部棋局的另一位玩家（該帳號也參與訓練時）：棋局鍵 -> (玩家, 顏色 0／1)
    if ext_ply and ext_second:  # 有指定第二主體的清單
        for f in ext_second:  # 每個 second.csv
            with open(f, newline="") as fh:  # 開啟 CSV
                sec.update((r["game_key"], ("x:" + r["account"], 0 if r["color"] == "B" else 1)) for r in csv.DictReader(fh))  # 棋局鍵 -> (另一方帳號, 其顏色)
        sec = {k: v for k, v in sec.items() if k in ext_own}  # 只留已載入的外部棋局
    with mp.get_context("fork").Pool(12) as pool:  # 在配置大陣列之前先啟動 12 個工作行程
        n1 = pool.map(M.count_part, files, chunksize=8)  # 每個已存分段檔的局數
        n2 = pool.map(M.count_m90, files2, chunksize=8)  # 每個逐手分段檔（M90 與外部）的局數
        n_old, N = int(sum(n1)), int(sum(n1)) + int(sum(n2))  # 已存局數與總局數
        NA = N + len(sec)  # 預留第二主體複本的空間（上限；陣列最後裁成實際用量）
        GL, POS, MOV = alloc("GL", (NA, T, 128), np.float16), alloc("POS", (NA, T), np.int16, 362), alloc("MOV", (NA, T), np.int8)  # 逐手主幹向量、落點（362 = 補齊）、執色
        LEN, SUBJ, keys, pids, a = alloc("LEN", (NA,), np.int16), alloc("SUBJ", (NA,), np.int8), [], [], 0  # 手數、主體顏色、鍵與玩家清單、寫入位置
        for x in _windows(pool, M.read_part, files, 12):  # 依序讀已存分段檔，最多 12 個同時等待
            b = a + len(x[0])  # 這個分段檔的結束位置
            keys.append(x[0]); pids.append(x[1])  # 記下棋局鍵與玩家
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[2], x[3], x[4], x[5], x[6]  # 直接寫進預先配置的陣列
            a = b  # 前進寫入位置
        assert a == n_old, (a, n_old)  # 已存分段檔的局數須與計數一致
        for x in _windows(pool, M.read_m90, files2, 48):  # 再讀逐手分段檔（M90 與外部），最多 48 個同時等待
            b = a + len(x[0])  # 這個分段檔的結束位置
            keys.append(x[0])  # 記下棋局鍵（玩家稍後由對照表補上）
            GL[a:b], POS[a:b], MOV[a:b], LEN[a:b], SUBJ[a:b] = x[1], x[2], x[3], x[4], x[5]  # 直接寫進預先配置的陣列
            a = b  # 前進寫入位置
        assert a == N, (a, N)  # 總局數須與計數一致
    _trim()  # 歸還讀取時的暫存緩衝區
    key, pid = np.concatenate(keys), np.concatenate(pids)  # 合併棋局鍵與已存分段檔的玩家
    fold = np.concatenate([np.full(n, f, np.int8) for n, f in zip(n1, folds)])  # 已存分段檔各局的折
    tab = pd.read_csv(AICUP1 / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]  # 玩家 -> 折的對照表
    own = {}  # 棋局鍵 -> 玩家
    for f_ in sorted((AICUP1 / "data/training").glob("train_*.csv")):  # 官方訓練集的各段位檔
        c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str)  # 只讀需要的欄位
        own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))  # 鍵為「段位:棋局編號」
    own.update(ext_own)  # 加入外部棋局的帳號
    p2 = np.array([own[k] for k in key[n_old:]])  # 逐手分段檔各局的玩家
    pid = np.concatenate([pid, p2])  # 接在已存分段檔的玩家之後
    fold = np.concatenate([fold, np.array([int(tab.get(p, -9)) for p in p2], np.int8)])  # 折表裡沒有的玩家（例如外部帳號）記為 -9
    second = np.zeros(N, bool)  # 標記第二主體的列
    if sec:  # 同一盤已存棋局再從另一位玩家那一側讀一次（主體顏色翻轉、換成該玩家）
        src = np.array([i for i in range(n_old, N) if key[i] in sec], np.int64)  # 有第二主體的來源列
        n2_ = len(src)  # 第二主體的局數
        want = np.array([sec[k][1] for k in key[src]], np.int8)  # 另一方的顏色
        bad = int((want != 1 - SUBJ[src]).sum())  # 顏色須是已存主體的另一側
        if bad:  # 有不一致的列
            raise ValueError(f"{bad} second-side rows whose colour is not the other side of the stored subject")  # 資料有誤，中止
        for j in range(0, n2_, 20000):  # 分批複製，避免一次產生大暫存
            r = src[j:j + 20000]  # 這一批的來源列
            GL[N + j:N + j + len(r)], POS[N + j:N + j + len(r)], MOV[N + j:N + j + len(r)], LEN[N + j:N + j + len(r)] = GL[r], POS[r], MOV[r], LEN[r]  # 複製到陣列尾端的預留空間
        SUBJ[N:N + n2_] = want  # 主體換成另一方的顏色
        key = np.concatenate([key, key[src]])  # 複本沿用同一個棋局鍵
        pid = np.concatenate([pid, np.array([sec[k][0] for k in key[src]])])  # 複本屬於另一方帳號
        fold = np.concatenate([fold, np.full(n2_, -9, np.int8)])  # 外部帳號沒有折
        second = np.concatenate([second, np.ones(n2_, bool)])  # 標記為第二主體
        log(f"second side: {n2_} external games are read once more from the other player's side ({len(set(pid[N:].tolist()))} players)")  # 記錄第二主體的局數與帳號數
        N += n2_  # 更新總局數
    GL, POS, MOV, LEN, SUBJ = GL[:N], POS[:N], MOV[:N], LEN[:N], SUBJ[:N]  # 裁掉沒用到的預留空間
    log(f"games loaded: {N} ({n_old} stored of folds {'0-4' if all_folds else '1-4'} + {N - n_old} extracted; {GL.nbytes / 1e9:.1f} GB; process memory {_rss_gb():.1f} GB); "  # 記錄載入的局數、大小與記憶體
        f"games of fold 0: {int((fold == 0).sum())}, {'TRAINED ON (--all-folds)' if all_folds else 'left out everywhere'}")  # 並註明第 0 折是否參與訓練
    if share is not None:  # 由本行程建立共享陣列時
        for arr in (GL, POS, MOV, LEN, SUBJ):  # 每個映射陣列
            arr.flush()  # 寫回共享檔案
        np.save(share / "key.npy", key); np.save(share / "pid.npy", pid); np.save(share / "fold.npy", fold)  # 小陣列另存，供同一工作的其他網路讀取
        (share / "DONE").write_text("ok\n")  # 標記已準備完成
        log(f"arrays shared in {share}")  # 記錄共享位置
    sup_idx, SUPD = np.full(N, -1, np.int32), None  # 預設沒有有監督特徵
    if sup_ply:  # METHOD_NOTES 2.88（探針 B）：部分棋局的有監督逐手特徵，與 GL 一樣補齊並以相同方式共享
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的 pid_sup 可被匯入
        import pid_sup  # 有監督特徵的讀取工具
        sup_idx, SUPD = pid_sup.dense(pid_sup.load_sup(sup_ply, sup_stats), key.tolist(), T, alloc=lambda shape, dtype: alloc("SUP", shape, dtype))  # 依棋局鍵排成補齊陣列（同樣可放共享記憶體）
        log(f"supervised per-ply features: {int((sup_idx >= 0).sum())} of {N} games, {SUPD.shape[2]} dimensions ({SUPD.nbytes / 1e9:.1f} GB)")  # 記錄有特徵的局數與維度
    deep = np.fromiter((k in deep_keys for k in key.tolist()), bool, len(key)) if deep_keys else np.zeros(len(key), bool)  # 標記較深挑選的棋局
    if deep_keys:  # 有較深的挑選
        log(f"deeper selections: {int(deep.sum())} of the loaded games ({len(set(pid[deep].tolist()))} accounts)")  # 記錄其局數與帳號數
    return dict(key=key, pid=pid, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, fold=fold, second=second, sup_idx=sup_idx, SUP=SUPD, deep=deep)  # 全部棋局資料


def main(argv=None):  # 訓練一個網路
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--net", type=int, required=True, help="1-4: the fold left out (never sees its players); -1: folds 1-4")  # 留出的折（1–4，不看其玩家）；-1 = 不留出，用第 1–4 折訓練（--all-folds 時含第 0 折）
    ap.add_argument("--tag", required=True)  # 檢查點與紀錄的標記
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seqnet")  # 檢查點輸出資料夾
    ap.add_argument("--steps", type=int, default=25000)  # 訓練步數
    ap.add_argument("--horizon", type=int, default=None, help="cosine horizon (default: --steps)")  # 餘弦排程降到零的步數
    ap.add_argument("--drop", type=float, default=0.1)  # dropout 比例
    ap.add_argument("--mask", type=float, default=0.0)  # 遮手比例的上限
    ap.add_argument("--ema", type=float, default=0.999)  # 權重 EMA 的衰減率
    ap.add_argument("--seed", type=int, default=0)  # 亂數種子
    ap.add_argument("--eval-every", type=int, default=5000)  # 留出評估的間隔步數
    ap.add_argument("--smoke", action="store_true")  # 冒煙測試：少量資料、少步數
    ap.add_argument("--ext-ply", nargs="*", default=None, help="one or more directories of external per-ply parts")  # 外部帳號的逐手分段檔資料夾
    ap.add_argument("--ext-select", nargs="*", default=None, help="the select.csv files that name their accounts")  # 外部帳號的挑選檔
    ap.add_argument("--sup-ply", nargs="*", default=None, help="directories of scripts/pid_sup_ply_extract.py parts (supervised per-ply features; METHOD_NOTES 2.88)")  # 有監督逐手特徵的資料夾
    ap.add_argument("--sup-stats", default=None, help="the stats file of the supervised stores (pid_sup.write_stats): one scale for all of them")  # 有監督特徵的共同尺度檔
    ap.add_argument("--sup-dims", type=int, nargs=2, default=None, metavar=("FROM", "TO"), help="put the supervised features into these input dimensions (0 64: in place of the pooled half, 64 128: of the played-point half)")  # 有監督特徵放入的輸入維度
    ap.add_argument("--sup-only", action="store_true", help="with --sup-dims: every other input dimension is set to its mean")  # 其餘維度設為平均
    ap.add_argument("--sup-require", action="store_true", help="train only on games that have supervised features (the same games for every arm of the probe)")  # 只用具備有監督特徵的棋局（探針各臂用同一批棋局）
    ap.add_argument("--sup-name", default=None, help="name of the supervised feature set, recorded in the checkpoint so that the pool fine-tune reads the matching pool features (None: the first set, METHOD_NOTES 2.88; js: the joint projection with the policy-expected vector + 4 scalars, 2.110)")  # 特徵集名稱，供題池微調讀對應的特徵
    ap.add_argument("--ext-second", nargs="*", default=None, help="second.csv files of experiments/ext_fox_census.py: external games are also read from the other player's side")  # 外部棋局也從另一方讀
    ap.add_argument("--no-second", action="store_true", help="leave the second-side rows out (a control arm inside a job that loaded them)")  # 對照臂：排除第二主體的列
    ap.add_argument("--ext-deep", nargs="*", default=None, help="select.csv files (also given in --ext-select) of DEEPER selections: more games of accounts selected before (METHOD_NOTES 2.126)")  # 較深挑選的清單
    ap.add_argument("--no-deep", action="store_true", help="leave the games of the deeper selections out (a control arm inside a job that loaded them)")  # 對照臂：排除較深挑選的棋局
    ap.add_argument("--ext-share", type=float, default=None, help="share of an episode's players that are external (default: no quota)")  # 情節中外部帳號的比例
    ap.add_argument("--shape-win", type=int, default=0, help="METHOD_NOTES 2.170 (probe W): the stones in an N x N window around every ply's point as one more input, read by a small convolutional encoder trained with the network (scripts/pid_win.py); stored in the checkpoint under win, the loaders rebuild it")  # 棋形視窗的邊長（0 = 不用）
    ap.add_argument("--arch", default=None, help="D,LAYERS,HEADS,FFN of the transformer (M90: 384,8,8,1536); stored in the checkpoint, the loaders rebuild the same network")  # Transformer 的寬度、層數、注意力頭、FFN
    ap.add_argument("--sym-aug", action="store_true")  # 每個情節隨機一種棋盤對稱
    ap.add_argument("--drop-open", type=float, default=0.0, help="share of the games of a batch whose opening is hidden (board vectors and played points of the first K plies, "  # 部分棋局隱藏開局
                                                                 "K drawn in 20 .. --drop-open-max): the network must also tell players apart from the middle game (METHOD_NOTES 2.60)")  # 迫使網路也從中盤辨識玩家
    ap.add_argument("--drop-open-max", type=int, default=100)  # 隱藏開局的最多手數
    ap.add_argument("--sym-game", action="store_true", help="with --sym-aug: every GAME gets its own random board symmetry (orientation carries no identity then) instead of one per episode")  # 每局各自隨機對稱
    ap.add_argument("--p-ep", type=int, default=None)  # 每個情節的玩家數
    ap.add_argument("--ext-share-end", type=float, default=None)  # 訓練結束時的外部比例
    ap.add_argument("--no-state", action="store_true")  # 不寫續跑檔
    ap.add_argument("--bank", action="store_true", help="cross-batch memory (METHOD_NOTES 2.64): every training player's prototype from its last visit is a negative "  # 跨批次記憶的負例
                                                        "for the games of later episodes, so the most similar players are present at every step, not only the 255 of the episode")  # 讓最相似的玩家每一步都在
    ap.add_argument("--bank-start", type=int, default=1000, help="first step that uses the memory")  # 開始使用記憶的步數
    ap.add_argument("--bank-ema", action="store_true", help="the memory holds prototypes computed by the EMA network (slowly moving, as a momentum encoder) instead of the online one")  # 記憶改存 EMA 網路的原型
    ap.add_argument("--bank-age", type=int, default=0, help="only memory entries written in the last this many steps count (0: all)")  # 記憶條目的有效步數
    ap.add_argument("--bank-ramp", type=int, default=0, help="steps over which the memory's share of the loss rises from 0 to 1 after --bank-start")  # 記憶損失比重的漸增步數
    ap.add_argument("--win", type=float, default=0.0, help="share of the games of a batch shown only through a random window of plies (true ply positions kept; the rest hidden); "  # 只露出一段手數的棋局比例
                                                           "evaluation always reads whole games (METHOD_NOTES 2.66; the rank session's idea)")  # 評估時一律讀整局
    ap.add_argument("--win-len", type=int, nargs="+", default=[64, 128])  # 手數視窗長度的候選值
    ap.add_argument("--bank-topk", type=int, default=0, help="only the k highest memory negatives of each game (0: all)")  # 每局只取最高的 k 個記憶負例
    ap.add_argument("--margin", type=float, default=0.0, help="cosine margin taken off a game's score against its own player")  # 自己玩家分數扣掉的餘弦邊界
    ap.add_argument("--cls-loss", choices=("none", "cosface"), default="none",  # 對全部訓練玩家的分類損失
                    help="classification over ALL training players with an additive cosine margin (face / speaker recognition; METHOD_NOTES 2.56): "  # 說明：加性餘弦邊界
                         "every step each game is told apart from every player's centre, not only from the 255 others of its episode")  # 每步都與所有玩家中心區分
    ap.add_argument("--cls-margin", type=float, default=0.2)  # 分類損失的邊界
    ap.add_argument("--cls-scale", type=float, default=30.0)  # 分類損失的尺度
    ap.add_argument("--cls-weight", type=float, default=1.0)  # 分類損失的權重
    ap.add_argument("--no-episode-loss", action="store_true", help="train with the classification loss alone")  # 只用分類損失
    ap.add_argument("--share-dir", default=None, help="a directory in shared memory (/dev/shm/...): the game arrays are built there once and mapped by every network of the job")  # 共享記憶體資料夾
    ap.add_argument("--prepare-only", action="store_true", help="build the shared arrays and stop")  # 只建立共享陣列
    ap.add_argument("--rank-ep", choices=("class", "band"), default=None,  # 同等級玩家的情節（困難負例）
                    help="hard negatives (METHOD_NOTES 2.55): an episode's competition players come from ONE rank class (class) or from a class and its two neighbours (band), "  # 說明：同等級或含相鄰等級
                         "its external accounts from one rank folder; classes and folders drawn in proportion to their players")  # 外部帳號取自同一段位資料夾
    ap.add_argument("--rank-ep-share", type=float, default=1.0, help="share of the steps drawn that way; the others are drawn as before")  # 依等級抽的步數比例
    ap.add_argument("--fix-chan", type=float, default=0.0, help="this network never sees a fixed random share of the 128 input dimensions (its own subset, from --seed): their scale is set so large that they standardise to zero, in training and wherever the checkpoint is used (METHOD_NOTES 2.97)")  # 固定隱藏一部分輸入維度
    ap.add_argument("--ext-bag", type=float, default=0.0, help="this network trains on a random share of the external accounts (its own subset, from --seed)")  # 只用一部分外部帳號
    ap.add_argument("--ema-only", action="store_true", help="write only the EMA checkpoints (official networks: the raw ones are never used)")  # 只寫 EMA 檢查點
    ap.add_argument("--snap", nargs="*", type=int, default=[], help="also write the checkpoints at these steps, as ckpt_<net>_<tag>k<step/1000>[e].pt (snapshots of one long run)")  # 另存快照的步數
    ap.add_argument("--m-ep", type=int, default=None, help="games per player in an episode (default 8; METHOD_NOTES 2.69)")  # 情節中每位玩家的局數
    ap.add_argument("--min-games", type=int, default=None, help="an EXTERNAL account trains with at least this many usable games (default 8); competition players always need 8")  # 外部帳號至少需要的局數
    ap.add_argument("--mask-pos", type=float, default=0.0, help="per game a random share up to this of the plies have their played POINT hidden (the board vectors stay): the complement of --mask (METHOD_NOTES 2.80)")  # 隱藏落點的比例上限
    ap.add_argument("--mask-both", action="store_true", help="--mask hides the played point of the masked plies as well (whole plies dropped at random)")  # 遮手時也隱藏落點
    ap.add_argument("--chan-drop", type=float, default=0.0, help="per game every one of the 128 input dimensions is replaced by its mean with this probability")  # 逐維度換成平均的機率
    ap.add_argument("--drop-tail", type=float, default=0.0, help="share of the games of a batch whose plies from a random ply K on are hidden (K uniform in --drop-tail-min .. 220)")  # 隱藏後段的棋局比例
    ap.add_argument("--drop-tail-min", type=int, default=100)  # 隱藏後段的最早手數
    ap.add_argument("--jitter", type=float, default=0.0, help="Gaussian noise on the per-ply trunk vectors in training, in units of each dimension's standard deviation (METHOD_NOTES 2.72)")  # 主幹向量的高斯雜訊
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing of the episode loss")  # 情節損失的標籤平滑
    ap.add_argument("--ext-max-plies", type=int, default=0, help="external games are shown only up to this ply (0: all; as if they had been stored that short; METHOD_NOTES 2.70)")  # 外部棋局只顯示到這一手
    ap.add_argument("--ext-cap", type=int, default=0, help="use only the first this many games of every external account (0: all)")  # 每個外部帳號最多用幾局
    ap.add_argument("--ext-accounts", type=int, default=0, help="use only this many external accounts: the first N of one fixed random order of the accounts with 8 or more usable games, "  # 只用前 N 個外部帳號
                                                               "so the subsets of different N are nested (0: all)")  # 不同 N 的子集互相包含
    ap.add_argument("--own-by-games", choices=("linear", "sqrt"), default=None,  # 競賽玩家依局數抽樣
                    help="with --ext-share: the competition players of an episode are drawn with probability proportional to their number of usable games (linear) or to its "  # 說明：線性或平方根
                         "square root, instead of uniformly; the external accounts stay uniform (METHOD_NOTES 2.124; the rank session's M161)")  # 外部帳號仍等機率
    ap.add_argument("--ext-by-games", choices=("linear", "sqrt"), default=None,  # 外部帳號也依局數抽樣
                    help="with --ext-share: the external accounts of an episode are drawn by their number of usable games as well (METHOD_NOTES 2.126; "  # 說明的前半
                         "their depths differ once --ext-second or deeper selections are loaded)")  # 載入第二主體或較深挑選後局數才有差別
    ap.add_argument("--nn-ep", type=float, default=0.0, help="share of the steps whose competition players are LOOK-ALIKE clusters (METHOD_NOTES 2.140): seeds drawn as usual, "  # 相像玩家群集的步數比例
                                                             "each joined by its nearest competition players by the prototypes of their last visit; the external accounts stay random")  # 依上次的原型找最近的玩家
    ap.add_argument("--nn-k", type=int, default=8, help="--nn-ep: players per cluster (the seed and its k - 1 nearest)")  # 每個群集的玩家數
    ap.add_argument("--nn-start", type=int, default=4000, help="--nn-ep: first step that may draw clusters (the prototypes are collected from step 0)")  # 開始抽群集的步數
    ap.add_argument("--drop-fold", type=int, nargs="*", default=[], help="competition folds left out of the training in addition to the held-out one "  # 另外排除的競賽折
                                                                         "(a measurement of what one more fold of competition players is worth; METHOD_NOTES 2.131)")  # 量測多一折玩家的價值
    ap.add_argument("--all-folds", action="store_true", help="read and train on fold 0 as well (needs the user's consent; METHOD_NOTES 2.40)")  # 連第 0 折也讀入並訓練（須使用者同意）
    a = ap.parse_args(argv)  # 解析參數
    import torch  # 在函式內匯入：main_fork 的父行程不載入 torch
    import torch.nn.functional as F  # 損失與正規化函式
    t0 = time.time()  # 計時起點
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  印出訊息並附上經過秒數
    v, steps = a.net, (300 if a.smoke else a.steps)  # 網路編號與訓練步數（冒煙測試 300 步）
    horizon = a.horizon or steps  # 餘弦排程長度預設等於步數
    eval_every = 100 if a.smoke else a.eval_every  # 評估間隔
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 確保存在
    name = f"{v}_{a.tag}{'_smoke' if a.smoke else ''}"  # 紀錄與續跑檔的名稱
    dev = "cuda"  # 在 GPU 上訓練
    log(f"net {name}: steps {steps} (cosine horizon {horizon}), dropout {a.drop}, ply mask up to {a.mask}, EMA {a.ema}, seed {a.seed}; GPU {torch.cuda.get_device_name(0)}")  # 記錄設定與 GPU 型號
    G = PRELOADED if PRELOADED is not None else load_games(a.smoke, log, a.all_folds, a.ext_ply, a.ext_select, a.share_dir, a.ext_second, a.sup_ply, a.sup_stats, a.ext_deep)  # fork 模式沿用父行程讀好的棋局，否則自己讀
    if PRELOADED is not None:  # fork 模式
        log(f"games shared by the job: {len(G['key'])} ({G['GL'].nbytes / 1e9:.1f} GB, read once; this process holds {_rss_gb():.1f} GB)")  # 記錄共用的局數與本行程記憶體
    if a.prepare_only:  # 只要建立共享陣列
        return 0  # 不訓練
    held = set(json.load(open(M.M82 / "held_keys.json")))  # 留出的自有棋局，從不用於訓練
    players, owner = np.unique(G["pid"], return_inverse=True)  # 玩家清單與每局的玩家索引
    C, fold = len(players), G["fold"]  # 玩家數與每局的折
    is_held = np.array([k in held for k in G["key"]])  # 標記留出棋局
    allowed = np.ones(len(fold), bool) if a.all_folds else fold != 0  # 除非 --all-folds，第 0 折一律排除
    is_ext = np.char.startswith(players.astype(str), "x:")  # 每位玩家是否為外部帳號
    use = ~is_held & allowed & (fold != v)  # 可訓練的棋局：非留出、允許的折、不屬於留出折 v
    if a.drop_fold:  # 另外排除的折
        use &= ~np.isin(fold, a.drop_fold)  # 從訓練中拿掉
    sec_row = G.get("second", np.zeros(len(fold), bool))  # 第二主體的列
    if a.no_second:  # 對照臂
        use &= ~sec_row  # 排除第二主體的列
    if a.no_deep:  # 對照臂
        use &= ~G.get("deep", np.zeros(len(fold), bool))  # 排除較深挑選的棋局
    sup_idx = G.get("sup_idx")  # 每局在有監督特徵陣列中的列（-1 = 沒有）
    if a.sup_dims:  # 要把有監督特徵放進輸入
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的 pid_sup 可被匯入
        global pid_sup  # 匯入的模組綁在模組層級名稱，供 sup_in 使用
        import pid_sup  # 把有監督特徵寫入輸入維度的工具
    if a.sup_require:  # 只用有特徵的棋局
        use &= sup_idx >= 0  # 排除沒有有監督特徵的棋局
    if a.sup_dims or a.sup_require:  # 這兩個選項需要已載入的特徵
        assert sup_idx is not None and G.get("SUP") is not None, "--sup-dims / --sup-require need --sup-ply"  # 缺特徵時中止
    if a.m_ep:  # 指定每位玩家的局數
        M.M_EP = a.m_ep  # 覆寫 M90 的預設 8 局
    if a.ext_cap or a.ext_accounts:  # 減少外部帳號數及／或每個帳號的局數（METHOD_NOTES 2.69）
        e_idx = np.flatnonzero(use & is_ext[owner])  # 可用的外部棋局
        eo = owner[e_idx]  # 其帳號
        od = np.argsort(eo, kind="stable")  # 依帳號穩定排序
        e_idx, eo = e_idx[od], eo[od]  # 同帳號的棋局排在一起
        rk = np.arange(len(eo)) - np.searchsorted(eo, eo, side="left")  # 棋局在其帳號可用棋局中的序位
        out_ = np.zeros(len(eo), bool)  # 要排除的外部棋局
        if a.ext_accounts:  # 限制帳號數
            n1_ = np.bincount(owner[use & is_ext[owner] & ~sec_row], minlength=C)  # 只數自己（第一主體）的棋局：有無第二主體時子集相同
            full = np.flatnonzero(n1_ >= M.MIN_GAMES)  # 自己至少有 8 局可用棋局的帳號，依名稱順序
            out_ |= ~np.isin(eo, np.random.default_rng(20261114).permutation(full)[:a.ext_accounts])  # 固定亂數順序的前 N 個以外都排除
        if a.ext_cap:  # 限制每個帳號的局數
            out_ |= rk >= a.ext_cap  # 超過上限的棋局排除
        use[e_idx[out_]] = False  # 套用到訓練遮罩
    if a.ext_bag:  # 每個網路用外部帳號中不同的隨機一部分：成員的錯誤彼此不同
        ext_p = np.flatnonzero(is_ext)  # 全部外部帳號
        out_p = ext_p[np.random.default_rng(777 + a.seed).random(len(ext_p)) >= a.ext_bag]  # 依種子抽出不用的帳號
        use &= ~np.isin(owner, out_p)  # 排除它們的棋局
    cnt = np.bincount(owner[use], minlength=C)  # 每位玩家的可用局數
    tp = np.flatnonzero(np.where(is_ext, cnt >= max(a.min_games or M.MIN_GAMES, M.M_EP), cnt >= max(M.MIN_GAMES, M.M_EP)))  # 局數足夠、可參與訓練的玩家
    W = int(cnt.max())  # 單一玩家的最多局數
    idx = np.flatnonzero(use)  # 可用棋局的列
    o = owner[idx]  # 其玩家
    order = np.argsort(o, kind="stable")  # 依玩家穩定排序
    idx, o = idx[order], o[order]  # 同玩家的棋局排在一起
    tab = np.full((C, W), -1, np.int64)  # 玩家 × 局序 -> 棋局列
    tab[o, np.arange(len(o)) - np.searchsorted(o, o, side="left")] = idx  # 填入每位玩家的棋局
    P = (a.p_ep or M.P_EP) if not a.smoke else min(M.P_EP, len(tp))  # 每個情節的玩家數
    tp_ext, tp_own = tp[is_ext[tp]], tp[~is_ext[tp]]  # 外部帳號與競賽玩家
    n_ext = 0 if a.ext_share is None or not len(tp_ext) else min(int(round(P * a.ext_share)), len(tp_ext))  # 每個情節的外部帳號數
    p_own = None  # 競賽玩家的抽樣機率（None = 等機率）
    if a.own_by_games:  # 依局數抽競賽玩家
        assert a.ext_share is not None and not a.rank_ep, "--own-by-games needs --ext-share and excludes --rank-ep"  # 檢查選項組合
        w_own = cnt[tp_own].astype(np.float64)  # 每位玩家的局數
        w_own = w_own if a.own_by_games == "linear" else np.sqrt(w_own)  # 線性或平方根
        p_own = w_own / w_own.sum()  # 正規化成機率
        top = np.sort(p_own)[::-1]  # 由大到小，供紀錄用
        log(f"competition players drawn by games ({a.own_by_games}): games per player 10/50/90/99% {np.percentile(cnt[tp_own], [10, 50, 90, 99]).round().astype(int).tolist()}; "  # 記錄局數的分位數
            f"the busiest 10% of the players get {top[:max(len(top) // 10, 1)].sum():.2f} of the draws (uniform: 0.10)")  # 與局數最多的一成玩家所佔的抽樣比例
    p_ext = None  # 外部帳號的抽樣機率（None = 等機率）
    if a.ext_by_games:  # 依局數抽外部帳號
        assert a.ext_share is not None and not a.rank_ep and len(tp_ext), "--ext-by-games needs --ext-share and external accounts, and excludes --rank-ep"  # 檢查選項組合
        w_ext = cnt[tp_ext].astype(np.float64)  # 每個帳號的局數
        w_ext = w_ext if a.ext_by_games == "linear" else np.sqrt(w_ext)  # 線性或平方根
        p_ext = w_ext / w_ext.sum()  # 正規化成機率
        top = np.sort(p_ext)[::-1]  # 由大到小，供紀錄用
        log(f"external accounts drawn by games ({a.ext_by_games}): games per account 10/50/90/99% {np.percentile(cnt[tp_ext], [10, 50, 90, 99]).round().astype(int).tolist()}; "  # 記錄局數的分位數
            f"the busiest 10% of the accounts get {top[:max(len(top) // 10, 1)].sum():.2f} of the draws (uniform: 0.10)")  # 與局數最多的一成帳號所佔的抽樣比例
    log(f"training on {int(use.sum())} games of {len(tp)} players (folds {sorted(set(fold[use].tolist()))}; external players {len(tp_ext)} with {int((use & is_ext[owner]).sum())} games"  # 記錄訓練的局數、玩家與外部帳號
        f"{'' if a.ext_share is None else f', {n_ext} of every episode'}); episodes of {P} players x {M.M_EP} games")  # 以及情節的形狀
    strata = None  # 依等級分層的玩家（--rank-ep 時才用）
    if a.rank_ep:  # 困難負例：同等級的情節
        grp = np.array([k.rsplit(":", 1)[0] for k in G["key"]])  # 競賽棋局的段位檔（例如 5D），外部棋局則為 x:<資料夾>
        names, gi = np.unique(grp, return_inverse=True)  # 群組名稱與每局的群組索引
        tally = np.zeros((C, len(names)), np.int32)  # 玩家 × 群組的局數
        np.add.at(tally, (owner[use], gi[use]), 1)  # 累計可用棋局
        pg = tally.argmax(1)  # 玩家的群組：其訓練棋局最多歸檔的地方

        def level(nm):  # 各族群內的強弱順序
            if nm.startswith("x:"):  # 外部帳號的段位資料夾
                return (-1 if nm.endswith("k") else 1) * int(nm[2:-1])  # 級為負、段為正
            return ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"].index(nm)  # 競賽的等級由弱到強

        strata = {}  # 族群 -> (玩家, 等級, 各等級的比例)
        for pop, ids in (("own", tp_own if a.ext_share is not None else tp), ("ext", tp_ext)):  # 競賽玩家與外部帳號分開分層
            if len(ids):  # 族群非空
                lv = np.array([level(names[g]) for g in pg[ids]])  # 每位玩家的等級
                lv = np.searchsorted(np.unique(lv), lv)  # 0 .. L-1，相鄰值即實際存在的相鄰等級
                strata[pop] = (ids, lv, np.bincount(lv) / len(lv))  # 等級依玩家數比例抽取
                log(f"rank episodes ({a.rank_ep}, share {a.rank_ep_share}): {pop} players by level " + ", ".join(f"{names[g]} {int((pg[ids] == g).sum())}" for g in sorted(set(pg[ids].tolist()), key=lambda g: level(names[g]))))  # 記錄各等級的玩家數

    def rank_draw(pop, k):  # 從一個等級（或等級帶）抽 k 位玩家
        """k players of one level (or band) of a population; widened to the nearest levels when the level is short."""
        ids, lv, share = strata[pop]  # 該族群的玩家、等級與比例
        c = rng.choice(len(share), p=share)  # 依玩家數比例抽一個等級
        w = 0 if a.rank_ep == "class" else 1  # class 只取該等級；band 含兩側相鄰等級
        while (np.abs(lv - c) <= w).sum() < k:  # 人數不足
            w += 1  # 擴大範圍
        return rng.choice(ids[np.abs(lv - c) <= w], k, replace=False)  # 不重複抽 k 位

    rng = np.random.default_rng(990 + v + 1000 * a.seed)  # 抽樣亂數，依網路與種子固定
    torch.manual_seed(990 + v + 1000 * a.seed)  # 初始權重與 GPU 亂數的種子
    M.DROP = a.drop  # 先設定 dropout 再建模型
    if a.arch:  # 指定不同大小的 Transformer
        M.D, M.LAYERS, M.HEADS, M.FFN = (int(x) for x in a.arch.split(","))  # 覆寫 M90 的預設架構
    arch = [int(M.D), int(M.LAYERS), int(M.HEADS), int(M.FFN)]  # 存進檢查點，讓載入端重建相同網路
    net = M.build_model().to(dev)  # 建立 M90 的序列網路
    win_cfg = None  # 棋形視窗設定（不用時為 None）
    if a.shape_win:  # 加入棋形視窗輸入
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的 pid_win 可被匯入
        import pid_win  # 棋形視窗編碼器
        win_cfg = {"size": int(a.shape_win), "ch": 64}  # 視窗邊長；編碼器固定 64 通道
        pid_win.wrap(net, win_cfg)  # 就地把視窗編碼器加到網路上
        log(f"window input: {a.shape_win} x {a.shape_win} cells around every ply's point, {sum(p.numel() for p in net.win.parameters())} parameters (pid_win.py)")  # 記錄視窗大小與參數量
    sample = G["GL"][rng.choice(idx, min(4000, len(idx)), replace=False)].astype(np.float32).reshape(-1, 128)  # 抽至多 4000 局的逐手向量估計標準化參數
    sample = sample[np.abs(sample).sum(1) > 0]  # 去掉補齊的全零手
    net.mu.copy_(torch.as_tensor(sample.mean(0)))  # 輸入標準化的平均
    net.sd.copy_(torch.as_tensor(sample.std(0) + 1e-6))  # 輸入標準化的標準差
    del sample  # 釋放樣本
    if a.fix_chan:  # 固定隱藏部分輸入維度
        hidden_dims = np.sort(np.random.default_rng(4242 + a.seed).choice(128, int(round(128 * a.fix_chan)), replace=False))  # 依種子選出此網路看不到的維度
        net.sd[torch.as_tensor(hidden_dims, device=net.sd.device)] = 1e6  # 標準差設得極大，標準化後為零（隨檢查點儲存）
        log(f"input dimensions never seen by this network: {len(hidden_dims)} of 128 ({hidden_dims[:8].tolist()} ...)")  # 記錄隱藏的維度
    ema = copy.deepcopy(net).eval()  # 權重的指數移動平均副本
    for p_ in ema.parameters():  # EMA 副本不做反向傳播
        p_.requires_grad_(False)  # 關閉梯度
    bank = bank_ok = bcol = None  # 跨批次記憶（不用時為 None）
    if a.bank:  # 沒有參數：原型直接複製自網路自己的輸出
        bank = torch.zeros(len(tp), M.D_OUT, device=dev)  # 每位訓練玩家一個原型槽
        bank_ok = torch.zeros(len(tp), dtype=torch.bool, device=dev)  # 槽是否已寫入
        bank_step = torch.full((len(tp),), -10 ** 9, dtype=torch.long, device=dev)  # 槽最後寫入的步數
        bcol = torch.full((C,), -1, dtype=torch.long, device=dev)  # 玩家索引 -> 槽位
        bcol[torch.as_tensor(tp, device=dev)] = torch.arange(len(tp), device=dev)  # 訓練玩家依序對應槽位
        log(f"cross-batch memory: {len(tp)} slots, used from step {a.bank_start}{f', top {a.bank_topk} per game' if a.bank_topk else ''}"  # 記錄記憶的設定
            f"{', entries from the EMA network' if a.bank_ema else ''}{f', entries up to {a.bank_age} steps old' if a.bank_age else ''}{f', ramp over {a.bank_ramp} steps' if a.bank_ramp else ''}; margin {a.margin}")  # 以及餘弦邊界
    nn_proto = nn_ok = None  # 相像群集用的原型（不用時為 None）
    if a.nn_ep > 0:  # 每位競賽玩家最近一次的原型，用來找誰像誰
        assert a.ext_share is not None and not a.rank_ep, "--nn-ep needs --ext-share and excludes --rank-ep"  # 檢查選項組合
        nn_proto = torch.zeros(C, M.D_OUT, device=dev)  # 每位玩家的原型
        nn_ok = np.zeros(C, bool)  # 原型是否已有
        log(f"look-alike episodes: {a.nn_ep:g} of the steps from step {a.nn_start}, clusters of {a.nn_k} among {len(tp_own)} competition players")  # 記錄相像情節的設定
    nn_steps = 0  # 相像情節的步數計數（只累加，之後沒有讀取）
    centres = col = None  # 分類損失的中心（不用時為 None）
    if a.cls_loss != "none":  # 每位訓練玩家一個中心；中心不屬於存檔的網路
        centres = torch.nn.Parameter(torch.randn(len(tp), M.D_OUT, device=dev) * 0.01)  # 以小隨機值初始化的可學習中心
        col = torch.full((C,), -1, dtype=torch.long, device=dev)  # 玩家索引 -> 中心編號
        col[torch.as_tensor(tp, device=dev)] = torch.arange(len(tp), device=dev)  # 訓練玩家依序對應中心
        log(f"classification loss {a.cls_loss}: {len(tp)} centres, margin {a.cls_margin}, scale {a.cls_scale}, weight {a.cls_weight}{', episode loss off' if a.no_episode_loss else ''}")  # 記錄分類損失的設定
    opt = torch.optim.AdamW([{"params": net.parameters(), "weight_decay": M.WD}] + ([{"params": [centres], "weight_decay": 0.0}] if centres is not None else []), lr=M.LR)  # AdamW（M90 的學習率與權重衰減；中心不衰減）
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / M.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, horizon) / horizon)))  # 前 M.WARM 步的線性暖身乘上餘弦衰減，在 horizon 步降到零
    state_file, start, hist = out / f"state_{name}.pt", 0, []  # 續跑檔、起始步與評估歷程
    if state_file.exists():  # 有續跑檔就接著訓練
        st = torch.load(state_file, map_location=dev, weights_only=False)  # 讀入續跑狀態
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])  # 還原網路、EMA、最佳化器與排程
        start, hist = st["step"], st["hist"]  # 還原步數與歷程
        rng = np.random.default_rng(990 + v + 1000 * a.seed + 7 * start)  # 續跑時改用依起始步而定的亂數序列
        log(f"resumed at step {start}")  # 記錄續跑點
    tb = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731  numpy 陣列轉成 GPU 張量

    def sup_in(rows_, gl_):  # 把有監督特徵放進批次輸入
        """The batch's input with the supervised features in their dimensions (--sup-dims); unchanged without the option."""
        if not a.sup_dims:  # 不放有監督特徵
            return gl_  # 原樣回傳
        si = sup_idx[rows_]  # 每局在特徵陣列中的列
        z = torch.zeros(len(rows_), T, G["SUP"].shape[2], device=dev)  # 沒有特徵的局保持零（= 平均）
        has = np.flatnonzero(si >= 0)  # 有特徵的局
        if len(has):  # 至少一局有特徵
            z[torch.as_tensor(has, device=dev)] = tb(G["SUP"][si[has]]).float()  # 填入其特徵
        return pid_sup.put(gl_.float(), z, net.mu, net.sd, a.sup_dims, a.sup_only)  # 依網路的標準化尺度寫入指定維度

    def embed(model, rows):  # 計算多局的輸出向量（評估用）
        model.eval()  # 評估模式
        res = np.zeros((len(rows), M.D_OUT), np.float32)  # 輸出向量
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不算梯度、以 bf16 計算
            for i in range(0, len(rows), 1024):  # 每批 1024 局
                r = rows[i:i + 1024]  # 這一批的列
                res[i:i + 1024] = model(sup_in(r, tb(G["GL"][r])), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()  # 前向計算後存回 CPU
        return res  # 每局一個單位向量

    # 評估集：折 v（沒看過的玩家）與訓練折（看過的玩家）的留出自有棋局；原型取自非留出棋局、第 0 折以外
    ev_ok = ~is_held & allowed & ~is_ext[owner]  # 留出紀錄以競賽玩家為圖庫
    e_idx = np.flatnonzero(ev_ok)  # 可當原型的棋局
    e_o = owner[e_idx]  # 其玩家
    e_order = np.argsort(e_o, kind="stable")  # 依玩家穩定排序
    e_idx, e_o = e_idx[e_order], e_o[e_order]  # 同玩家排在一起
    proto_rows = e_idx[(np.arange(len(e_o)) - np.searchsorted(e_o, e_o, side="left")) < PROTO_GAMES]  # 每位玩家最多取前 PROTO_GAMES 局當原型
    hk = np.flatnonzero(is_held & allowed & (np.bincount(owner[ev_ok], minlength=C)[owner] > 0))  # 有原型的玩家的留出棋局
    unseen = fold[hk] == v  # 屬於留出折 v 的是沒看過的玩家

    def evaluate(model, step, label):  # 留出棋局的辨識準確率（只當紀錄）
        E = embed(model, proto_rows)  # 原型棋局的向量
        Pm = np.zeros((C, M.D_OUT), np.float32)  # 每位玩家的原型
        np.add.at(Pm, owner[proto_rows], E)  # 累加同玩家的向量
        has = np.linalg.norm(Pm, axis=1) > 0  # 有原型的玩家
        Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 正規化成單位向量
        S = embed(model, hk) @ Pm.T  # 留出棋局對每位玩家的餘弦分數
        S[:, ~has] = -9.0  # 沒有原型的玩家不可能被選中
        r = (S > S[np.arange(len(hk)), owner[hk]][:, None]).sum(1)  # 真正玩家的名次（0 = 第一名）
        res = {"step": step, "state": label}  # 這次評估的結果
        for nm, s in (("seen", ~unseen), ("unseen", unseen)):  # 分看過與沒看過的玩家
            if s.any():  # 有這類棋局
                res[nm] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}  # 局數與前 1／5／20 名命中率
        log(f"  step {step} [{label}]: held-out own games among {int(has.sum())} players: " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if isinstance(x, dict)))  # 記錄評估結果
        return res  # 加入訓練歷程

    sym_tab = None  # 棋盤對稱的落點對照表
    if a.sym_aug:  # [8, 363]：每種棋盤對稱下各點的去處（361 = 虛手與 362 = 補齊不變）
        yy, xx = np.divmod(np.arange(361), 19)  # 每個點的列與行
        maps = [(xx, yy), (18 - xx, yy), (xx, 18 - yy), (18 - xx, 18 - yy), (yy, xx), (18 - yy, xx), (yy, 18 - xx), (18 - yy, 18 - xx)]  # 八種對稱（旋轉與鏡射）後的座標
        sym_tab = np.stack([np.concatenate([b_ * 19 + a_, [361, 362]]) for a_, b_ in maps]).astype(np.int64)  # 換回點編號，補上 361、362
    cpu = lambda m: {k: x.detach().cpu().clone() for k, x in m.state_dict().items()}  # noqa: E731  網路狀態複製到 CPU
    meta = {"step": steps, "horizon": horizon, "drop": a.drop, "mask": a.mask, "ema": a.ema, "seed": a.seed, "net": v, "players_trained": int(len(tp)), "games": int(use.sum()), "all_folds": bool(a.all_folds), "ext_players": int(len(tp_ext)), "ext_share": a.ext_share, "ext_share_end": a.ext_share_end, "sym_aug": bool(a.sym_aug), "sym_game": bool(a.sym_game), "drop_open": a.drop_open, "drop_open_max": a.drop_open_max, "p_ep": int(P), "rank_ep": a.rank_ep, "rank_ep_share": a.rank_ep_share, "bank": bool(a.bank), "bank_start": a.bank_start, "bank_topk": a.bank_topk, "bank_ema": bool(a.bank_ema), "bank_age": a.bank_age, "bank_ramp": a.bank_ramp, "win": a.win, "win_len": list(a.win_len), "margin": a.margin, "cls_loss": a.cls_loss, "cls_margin": a.cls_margin, "cls_scale": a.cls_scale, "cls_weight": a.cls_weight, "episode_loss": not a.no_episode_loss}  # 存進檢查點與訓練紀錄的設定

    def write_ckpt(tag, at):  # 寫出檢查點
        for sfx, model, label in (("", net, "raw"), ("e", ema, "ema")):  # 原始權重與 EMA 權重各一個檔
            if a.ema_only and sfx == "":  # 只要 EMA 版
                continue  # 跳過原始版
            sd_ = cpu(model)  # 一份複本，存在載入端認得的兩個名稱下（以前兩份獨立複本讓每個檔變兩倍大）
            torch.save({"best_state": sd_, "net": sd_, "best": {**meta, "step": at, "state": label}, "step": at, "arch": arch, "win": win_cfg,  # 權重、設定、架構與視窗設定
                        "sup": ({"dims": list(a.sup_dims), "only": bool(a.sup_only), "name": a.sup_name} if a.sup_dims else None)}, out / f"ckpt_{v}_{tag}{sfx}{'_smoke' if a.smoke else ''}.pt")  # 有監督特徵的設定；寫到 ckpt_<網路>_<標記>[e].pt

    net.train()  # 訓練模式
    from o26_parallel_forward import make_forward  # 多張 GPU 只平行化嵌入計算的包裝
    training_forward = make_forward(net)  # 兩張以上 GPU 時用 DataParallel，否則就是網路本身
    log(f"training forward across {torch.cuda.device_count()} GPU(s); full episode loss on GPU 0")  # 記錄 GPU 數；完整的情節損失在 GPU 0
    ar, tl, ta, tc, t_last, every = np.arange(P), 0.0, 0.0, 0.0, time.time(), (50 if a.smoke else 500)  # 情節玩家編號、損失與準確率累計、計時、紀錄間隔
    for step in range(start, steps):  # 訓練迴圈
        if a.ext_share is not None and a.ext_share_end is not None:  # 外部比例隨訓練線性變化
            n_ext = min(int(round(P * (a.ext_share + (a.ext_share_end - a.ext_share) * step / max(steps - 1, 1)))), len(tp_ext))  # 這一步的外部帳號數
        if strata is not None and rng.random() < a.rank_ep_share:  # 這一步用同等級的情節
            pl = rank_draw("own", P) if a.ext_share is None or n_ext == 0 else np.concatenate([rank_draw("ext", n_ext), rank_draw("own", P - n_ext)])  # 外部與競賽玩家各自依等級抽
        elif nn_proto is not None and step >= a.nn_start and rng.random() < a.nn_ep and int(nn_ok[tp_own].sum()) >= 4 * (P - n_ext):  # 這一步用相像群集（已有原型的玩家夠多時）
            n_own = P - n_ext  # 競賽玩家數
            okm = nn_ok[tp_own]  # 已有原型的競賽玩家
            valid = tp_own[okm]  # 可當群集成員的玩家
            seeds = rng.choice(valid, n_own // a.nn_k, replace=False, p=None if p_own is None else p_own[okm] / p_own[okm].sum())  # 抽群集的種子玩家（沿用依局數的機率）
            sim = (nn_proto[tb(seeds)] @ nn_proto[tb(valid)].T).cpu().numpy()  # 種子對各玩家的相似度
            used = np.zeros(C, bool)  # 已被群集取用的玩家
            used[seeds] = True  # 種子本身已用
            chosen = []  # 這一步的競賽玩家
            for i_, sd_ in enumerate(seeds):  # 種子與其最近、尚未被先前群集取走的玩家
                near = valid[np.argsort(-sim[i_])[:4 * a.nn_k + len(seeds)]]  # 最相似的候選
                near = near[~used[near]][:a.nn_k - 1]  # 取未被用過的前 k - 1 位
                used[near] = True  # 標記已用
                chosen += [sd_, *near.tolist()]  # 加入群集
            if len(chosen) < n_own:  # 人數不足
                free = tp_own[~used[tp_own]]  # 尚未取用的競賽玩家
                chosen += rng.choice(free, n_own - len(chosen), replace=False).tolist()  # 隨機補滿
            pl = np.concatenate([rng.choice(tp_ext, n_ext, replace=False, p=p_ext), np.array(chosen, dtype=tp_own.dtype)])  # 外部帳號仍隨機抽
            nn_steps += 1  # 累計相像情節的步數（未再使用）
        else:  # 一般情節
            pl = rng.choice(tp, P, replace=False) if a.ext_share is None else np.concatenate([rng.choice(tp_ext, n_ext, replace=False, p=p_ext), rng.choice(tp_own, P - n_ext, replace=False, p=p_own)])  # 隨機抽玩家（有外部比例時分開抽）
        u = rng.random((P, W))  # 每位玩家各局的隨機鍵
        u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0  # 超出該玩家局數的欄位排到最後
        rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M.M_EP]].reshape(-1)  # 每位玩家不重複隨機取 M_EP 局
        gl = sup_in(rows, tb(G["GL"][rows]))  # 讀入主幹向量（需要時放入有監督特徵）
        if a.jitter > 0:  # 同一局不會兩次看起來完全一樣
            gl = gl + (a.jitter * net.sd.to(gl.dtype))[None, None] * torch.randn_like(gl)  # 加上以標準差為單位的高斯雜訊
        hide = None  # 要隱藏落點的手（None = 都不隱藏）
        if a.mask > 0:  # 遮手
            hit = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask  # 每局抽一個比例，遮掉該比例的手
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 主幹向量換成平均（標準化後為零）
            if a.mask_both:  # 也隱藏落點
                hide = hit  # 整手拿掉
        if a.chan_drop > 0:  # 逐維度丟棄
            cd = torch.rand(len(rows), 1, gl.shape[-1], device=dev) < a.chan_drop  # 每局抽要丟的維度
            gl = torch.where(cd, net.mu.to(gl.dtype)[None, None], gl)  # 換成平均
        if a.mask_pos > 0:  # 只隱藏落點
            hp = torch.rand(len(rows), T, device=dev) < torch.rand(len(rows), 1, device=dev) * a.mask_pos  # 每局抽一個比例的手
            hide = hp if hide is None else (hide | hp)  # 併入隱藏遮罩
        if a.drop_tail > 0:  # 隱藏後段
            cut_t = torch.where(torch.rand(len(rows), device=dev) < a.drop_tail, torch.randint(a.drop_tail_min, 221, (len(rows),), device=dev), torch.full((len(rows),), T + 1, dtype=torch.long, device=dev))  # 抽中的棋局從第 K 手起隱藏，其餘不截
            tail = torch.arange(T, device=dev)[None] >= cut_t[:, None]  # 從第 K 手（含）起的位置
            gl = torch.where(tail[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 主幹向量換成平均
            hide = tail if hide is None else (hide | tail)  # 落點也隱藏
        if a.drop_open > 0:  # 隱藏開局
            cut = torch.where(torch.rand(len(rows), device=dev) < a.drop_open, torch.randint(20, a.drop_open_max + 1, (len(rows),), device=dev), torch.zeros(len(rows), dtype=torch.long, device=dev))  # 抽中的棋局隱藏前 K 手（K 在 20 到 --drop-open-max）
            opening = torch.arange(T, device=dev)[None] < cut[:, None]  # 前 K 手的位置
            gl = torch.where(opening[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 主幹向量換成平均
            hide = opening if hide is None else (hide | opening)  # 落點也隱藏
        if a.win > 0:  # 只露出一段手數視窗
            len_w = tb(G["LEN"][rows].astype(np.int64))  # 每局手數
            w_len = torch.as_tensor(a.win_len, device=dev)[torch.randint(len(a.win_len), (len(rows),), device=dev)]  # 每局抽一個視窗長度
            w_start = (torch.rand(len(rows), device=dev) * (len_w - w_len).clamp(min=0).float()).long()  # 隨機的起點
            t_idx = torch.arange(T, device=dev)[None]  # 手數位置
            outside = ((t_idx < w_start[:, None]) | (t_idx >= (w_start + w_len)[:, None])) & (torch.rand(len(rows), device=dev) < a.win)[:, None]  # 被抽中的棋局在視窗外的手
            gl = torch.where(outside[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 主幹向量換成平均
            hide = outside if hide is None else (hide | outside)  # 落點也隱藏（真正的手數位置保留）
        with torch.autocast("cuda", dtype=torch.bfloat16):  # 以 bf16 前向計算
            pos_np = G["POS"][rows].astype(np.int64)  # 落點
            if sym_tab is not None:  # 棋盤對稱擴增
                pos_np = sym_tab[rng.integers(8, size=len(rows))[:, None], pos_np] if a.sym_game else sym_tab[rng.integers(8)][pos_np]  # 每局各自或整個情節共用一種對稱
            pos_t = tb(pos_np)  # 轉成 GPU 張量
            if hide is not None:  # 有要隱藏的落點
                pos_t = torch.where(hide, torch.full_like(pos_t, 362), pos_t)  # 362 = 補齊點：沒有落子、沒有位移
            len_np = G["LEN"][rows].astype(np.int64)  # 每局手數
            if a.ext_max_plies:  # 外部棋局只顯示到某一手
                len_np = np.where(is_ext[owner[rows]], np.minimum(len_np, a.ext_max_plies), len_np)  # 截短外部棋局
            mov_t, len_t, subj_t = tb(G["MOV"][rows].astype(np.int64)), tb(len_np), tb(G["SUBJ"][rows].astype(np.int64))  # 執色、手數與主體顏色
            z = training_forward(gl, pos_t, mov_t, len_t, subj_t)  # 每局主體的單位向量
        z = z.float().reshape(P, M.M_EP, M.D_OUT)  # 整理成 玩家 × 局 × 維度
        tot, sc = z.sum(1), net.log_scale.exp()  # 每位玩家的向量和；可學習的分數尺度
        logits = sc * z.reshape(-1, M.D_OUT) @ F.normalize(tot, dim=-1).T  # 每局對每位玩家原型的分數
        own_ = sc * (z * F.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)  # 一局對自己玩家原型（不含這一局）的分數
        tgt = tb(np.repeat(ar, M.M_EP))  # 每局的正確玩家
        logits = logits.scatter(1, tgt[:, None], (own_ - sc * a.margin)[:, None])  # 正確玩家的分數換成不含自己的版本，並扣掉邊界
        loss = F.cross_entropy(logits, tgt, label_smoothing=a.smooth)  # 情節的交叉熵損失
        if bank is not None and step >= a.bank_start:  # 加入跨批次記憶的負例
            bslots = bcol[tb(pl)]  # 本情節玩家的記憶槽
            off = ~bank_ok if not a.bank_age else ~(bank_ok & (step - bank_step <= a.bank_age))  # 無效或過舊的條目
            off[bslots] = True  # 沒有可用條目的玩家，以及本情節自己的玩家（其新原型已在 logits 中）
            if not bool(off.all()):  # 至少有一個可用條目
                lb = (sc * (z.reshape(-1, M.D_OUT) @ bank.T)).masked_fill(off[None], -1e4)  # 每局對記憶原型的分數（無效者壓到極低）
                if a.bank_topk:  # 只取最高的 k 個
                    lb = lb.topk(min(a.bank_topk, lb.shape[1]), dim=1).values  # 保留最難的負例
                lam = min(1.0, (step - a.bank_start + 1) / a.bank_ramp) if a.bank_ramp else 1.0  # 記憶損失的比重（可漸增）
                loss = (1 - lam) * loss + lam * F.cross_entropy(torch.cat([logits, lb], 1), tgt)  # 混合加入記憶負例後的損失
        if centres is not None:  # 分類損失
            ct = col[tb(np.repeat(pl, M.M_EP))]  # 每局玩家的中心編號
            cos = z.reshape(-1, M.D_OUT) @ F.normalize(centres, dim=-1).T  # 對所有中心的餘弦
            l_cls = F.cross_entropy(a.cls_scale * (cos - a.cls_margin * F.one_hot(ct, len(tp)).to(cos.dtype)), ct)  # CosFace：正確類別扣掉邊界後再放大
            loss = a.cls_weight * l_cls if a.no_episode_loss else loss + a.cls_weight * l_cls  # 只用或加上分類損失
            tc += float((cos.argmax(1) == ct).float().mean())  # 累計全玩家的前 1 名準確率
        opt.zero_grad(set_to_none=True)  # 清除梯度
        loss.backward()  # 反向傳播
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪
        opt.step()  # 更新權重
        sched.step()  # 更新學習率
        with torch.no_grad():  # 以下不需梯度
            d = min(a.ema, (1 + step) / (10 + step))  # EMA 衰減率：初期較小，讓 EMA 快速跟上
            for pe, pn in zip(ema.parameters(), net.parameters()):  # 逐參數更新
                pe.lerp_(pn, 1 - d)  # EMA 往目前的權重移動
            if bank is not None:  # 更新跨批次記憶
                bslots = bcol[tb(pl)]  # 本情節玩家的記憶槽
                if a.bank_ema:  # 記憶存 EMA 網路的原型
                    with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 前向計算
                        zb = ema(gl, pos_t, mov_t, len_t, subj_t)  # EMA 網路的向量
                    zb = zb.float().reshape(P, M.M_EP, M.D_OUT)  # 整理成 玩家 × 局 × 維度
                else:  # 用線上網路的輸出
                    zb = z.detach()  # 不追蹤梯度
                bank[bslots] = F.normalize(zb.sum(1), dim=-1)  # 寫入新原型
                bank_ok[bslots] = True  # 標記可用
                bank_step[bslots] = step  # 記下寫入步數
            if nn_proto is not None:  # 更新相像群集用的原型
                own_m = ~is_ext[pl]  # 本情節的競賽玩家
                nn_proto[tb(pl[own_m])] = F.normalize(z.detach().sum(1)[tb(np.flatnonzero(own_m))], dim=-1)  # 存入這些玩家的最新原型
                nn_ok[pl[own_m]] = True  # 標記已有原型
        tl += float(loss.detach())  # 累計損失
        ta += float((logits.argmax(1) == tgt).float().mean())  # 累計情節內的前 1 名準確率
        if (step + 1) % every == 0:  # 定期記錄
            log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, episode top-1 {ta / every:.3f} of {P}, scale {float(sc.detach()):.1f}, {(time.time() - t_last) / every:.2f} s/step" + (f", all-player top-1 {tc / every:.3f}" if centres is not None else ""))  # 記錄平均損失、準確率、尺度與速度
            tl, ta, tc, t_last = 0.0, 0.0, 0.0, time.time()  # 重設累計
        if step + 1 in a.snap and step + 1 < steps:  # 到了快照步數
            write_ckpt(f"{a.tag}k{(step + 1) // 1000}", step + 1)  # 寫出快照檢查點
            log(f"  snapshot at step {step + 1}: ckpt_{v}_{a.tag}k{(step + 1) // 1000}[e].pt")  # 記錄快照檔名
        if (step + 1) % eval_every == 0 or step + 1 == steps:  # 定期與最後一步評估
            hist += [evaluate(net, step + 1, "raw"), evaluate(ema, step + 1, "ema")]  # 原始與 EMA 網路各評估一次
            net.train()  # 評估後回到訓練模式
            if not a.no_state:  # 需要續跑檔
                torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "hist": hist}, state_file)  # 寫出續跑點
            t_last = time.time()  # 評估與存檔的時間不計入速度
    write_ckpt(a.tag, steps)  # 寫出最後的檢查點
    (out / f"train_{name}.json").write_text(json.dumps({**meta, "history": hist}, indent=1) + "\n")  # 寫出設定與評估歷程
    state_file.unlink(missing_ok=True)  # 完成後刪除續跑檔
    log(f"net {name}: wrote ckpt_{v}_{a.tag}.pt and ckpt_{v}_{a.tag}e.pt in {out}")  # 記錄輸出的檔案
    return 0  # 成功


def main_fork(argv):  # 一個工作內多個網路共用一份棋局
    """Several networks of one job on ONE copy of the games (METHOD_NOTES 2.57).
        pid_seq_train.py --fork <spec file> [--smoke] [--all-folds] [--ext-ply ...] [--ext-select ...]
    The games are read once into anonymous shared memory, then one trainer is forked per line of the spec file:
        <GPU id> <log file> <arguments of a normal run: --net v --tag t ...>
    A trainer sets its GPU before CUDA starts and writes to its log. Why: on the shared nodes every private copy (95-180 GB)
    is slow to allocate when the page cache is full, and eight copies run into the node's commit limit; /dev/shm is not
    reliable (it was full of another user's files on 25a-hgpn069)."""
    global SHARED_ANON, PRELOADED  # 要設定模組層級的旗標與預載資料
    import shlex  # 依 shell 規則切分 spec 檔的每一行
    ap = argparse.ArgumentParser()  # fork 模式自己的參數
    ap.add_argument("--fork", required=True)  # spec 檔
    ap.add_argument("--smoke", action="store_true")  # 冒煙測試
    ap.add_argument("--all-folds", action="store_true")  # 也讀入第 0 折
    ap.add_argument("--ext-ply", nargs="*", default=None)  # 外部帳號的逐手分段檔資料夾
    ap.add_argument("--ext-select", nargs="*", default=None)  # 外部帳號的挑選檔
    ap.add_argument("--ext-second", nargs="*", default=None)  # 第二主體的清單
    ap.add_argument("--sup-ply", nargs="*", default=None)  # 有監督逐手特徵的資料夾
    ap.add_argument("--sup-stats", default=None)  # 有監督特徵的尺度檔
    ap.add_argument("--ext-deep", nargs="*", default=None)  # 較深挑選的清單
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    log = lambda s_: print(f"{s_} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  印出訊息並附上經過秒數
    specs = [shlex.split(line) for line in Path(a.fork).read_text().splitlines() if line.strip() and not line.startswith("#")]  # 讀 spec 檔，略過空行與註解行
    SHARED_ANON = True  # 大陣列改放匿名共享記憶體
    G = load_games(a.smoke, log, a.all_folds, a.ext_ply, a.ext_select, None, a.ext_second, a.sup_ply, a.sup_stats, a.ext_deep)  # 只讀一次棋局
    for k in SHARED:  # 每個共享大陣列
        G[k].flags.writeable = False  # 訓練行程只讀；誤寫會被所有行程看到
    data = (["--smoke"] if a.smoke else []) + (["--all-folds"] if a.all_folds else []) + (["--ext-ply", *a.ext_ply] if a.ext_ply else []) + (["--ext-select", *a.ext_select] if a.ext_select else []) + (["--ext-second", *a.ext_second] if a.ext_second else []) + (["--sup-ply", *a.sup_ply] if a.sup_ply else []) + (["--sup-stats", a.sup_stats] if a.sup_stats else []) + (["--ext-deep", *a.ext_deep] if a.ext_deep else [])  # 資料相關參數，附加給每個訓練行程
    pids = {}  # 子行程 -> 紀錄檔
    for gpu, logfile, *rest in specs:  # spec 的每一行一個訓練行程
        sys.stdout.flush()  # fork 前清空緩衝，避免重複輸出
        pid = os.fork()  # 子行程與父行程共用已讀入的記憶體頁
        if pid == 0:  # 子行程
            try:  # 任何例外都轉成結束碼
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # 在 CUDA 啟動前指定 GPU
                fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)  # 開啟自己的紀錄檔
                os.dup2(fd, 1); os.dup2(fd, 2)  # 標準輸出與標準錯誤導向紀錄檔
                sys.stdout = os.fdopen(1, "w", buffering=1); sys.stderr = os.fdopen(2, "w", buffering=1)  # 改為逐行緩衝
                PRELOADED = G  # 讓 main 沿用已讀好的棋局
                rc = main(rest + data)  # 執行一般的訓練
            except BaseException:  # noqa: BLE001  連 SystemExit 也攔下
                import traceback  # 印出錯誤堆疊
                traceback.print_exc()  # 寫進紀錄檔
                rc = 1  # 以失敗結束
            sys.stdout.flush(); sys.stderr.flush()  # 結束前清空緩衝
            os._exit(int(rc or 0))  # 直接結束子行程，不執行繼承來的清理
        pids[pid] = logfile  # 記下子行程
        log(f"trainer {pid} on GPU {gpu}: {' '.join(rest)} -> {logfile}")  # 記錄啟動的訓練行程
    failed = 0  # 是否有失敗的訓練行程
    while pids:  # 等待全部子行程
        pid, status = os.wait()  # 任一子行程結束
        ok = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0  # 正常結束且結束碼為 0
        log(f"trainer {pid} ({pids.pop(pid)}) {'done' if ok else 'FAILED'}")  # 記錄結果
        failed |= not ok  # 累計失敗
    return int(failed)  # 有任何失敗就回傳 1


if __name__ == "__main__":  # 直接執行時
    sys.exit(main_fork(sys.argv[1:]) if "--fork" in sys.argv[1:] else main())  # 有 --fork 走共用資料模式，否則訓練單一網路
