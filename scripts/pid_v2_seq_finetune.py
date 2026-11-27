#!/usr/bin/env python
"""Pool adaptation of the sequence identity model (METHOD_NOTES 2.31): a short fine-tune of the rank session's network
(AI_CUP_2026-1 scripts/m_series/m88_seqid.py) on ONE pool's own candidates (600 players x 10 games, ids given by the organisers), with
the model's own objective: every game is a query against the prototypes of the players in the episode, its own game left out.
No question game enters the fine-tune; a pool is adapted on its own candidates only (Public and Private never mixed).
Why: the network identifies players it was trained on far better than unseen ones (held-out games: 0.21-0.23 against 0.15-0.17
top-1 in the rank session's evaluation); official players are unseen.
  --mode sim --fold v --seed s        the simulated pool of pid_v2_sim2.py (fold v, net v) -> <out>/sim_f<v>_s<s>_t<step>.npz
  --mode official --pool pub2|priv2 --net v   an added official pool under net v -> <out>/official_<pool>_net<v>_t<step>.npz
  --mode official ... --holdout h [--holdout-seed s]   the labelled pseudo-pool of METHOD_NOTES 2.41: h games of every candidate
                                      (aicup_pid.v2.candidate_holdout) are kept out of the fine-tune, like question games
                                      -> <out>/offsim_<pool>_h<h>s<s>_net<v>_t<step>.npz
Each output holds the vectors of every game of the pool (candidates and questions) after <step> fine-tune steps (0 = as is).
Run from the AI_CUP_2026-1 checkout."""
import argparse  # 解析命令列參數
import os  # os.replace 原子換檔
import glob  # 列出逐手特徵的分段檔
import math  # 餘弦排程
import multiprocessing as mp  # 平行讀取訓練棋局的分段檔
import sys  # 設定 sys.path 與結束碼
import time  # 計時
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取官方候選檔與分折表
import torch  # 微調網路

AICUP1 = Path.cwd()  # 須從 AI_CUP_2026-1 的根目錄執行
R2 = Path(__file__).resolve().parents[1]  # 本儲存庫（AI_CUP_2026-2）的根目錄
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "scripts/m_series"), str(R2 / "src"), str(R2 / "scripts")]  # 兩個儲存庫的套件與腳本放到搜尋路徑最前面
import m88_seqid as m88  # noqa: E402  序列網路的定義（棋力 session 的 M88）


def _exact(net, A):  # 精確視窗網路需要的顏色打包
    """A for a network with the exact window (pid_winx.py, METHOD_NOTES 2.180): its colours carry the removal plies
    (computed once per set of arrays). Any other network: A as it is."""
    if A is None or not getattr(net, "win_exact", False):  # 沒有資料，或不是精確視窗網路
        return A  # 原樣回傳
    if "MOVX" not in A:  # 這組陣列還沒打包過
        import pid_winx  # 精確視窗的模組
        A["MOVX"] = pid_winx.pack_mov(A["POS"], A["MOV"], A["LEN"])  # 把每手那顆子被提走的手數一併編進顏色值
    return dict(A, MOV=A["MOVX"])  # 以打包後的顏色取代 MOV


DP = False  # --dp：微調的訓練前向傳遞分散到所有可見的 GPU（損失仍在 GPU 0）


def _training_forward(net):  # 訓練時用的前向傳遞物件
    """The network itself, or with --dp and two or more visible GPUs a DataParallel copy of it for the training forward
    (the same construction as the trainer's parallel helper: autocast is set again inside every replica thread). For the
    512x10 networks with the second window, whose 2,048-game episode does not fit one 140 GB card (METHOD_NOTES 2.204)."""
    if not DP or torch.cuda.device_count() < 2:  # 沒開 --dp 或只有一張卡
        return net  # 直接用網路本身

    class BF16Replica(torch.nn.Module):  # 包一層，讓每個副本都在 bf16 autocast 下執行
        def __init__(self, base):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.base = base  # 被包的原網路

        def forward(self, *args, **kwargs):  # 前向傳遞
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.is_autocast_enabled("cuda")):  # 在副本執行緒內重設 bf16 autocast，啟用與否沿用目前狀態
                return self.base(*args, **kwargs)  # 呼叫原網路

    return torch.nn.DataParallel(BF16Replica(net), device_ids=list(range(torch.cuda.device_count())))  # 分散到所有可見的 GPU


def embed(net, A, idx, batch=1024):  # 算出一組棋局的向量
    A = _exact(net, A)  # 精確視窗網路先打包顏色
    net.eval()  # 推論模式（關閉 dropout）
    out = np.zeros((len(idx), m88.D_OUT), np.float32)  # 每局一個 128 維向量
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不算梯度、用 bf16
        for i in range(0, len(idx), batch):  # 分批計算
            r = idx[i:i + batch]  # 這一批的棋局索引
            out[i:i + batch] = net(torch.as_tensor(A["GL"][r]).cuda(), torch.as_tensor(A["POS"][r].astype(np.int64)).cuda(), torch.as_tensor(A["MOV"][r].astype(np.int64)).cuda(),  # 逐手向量、落點與顏色
                                   torch.as_tensor(A["LEN"][r].astype(np.int64)).cuda(), torch.as_tensor(A["SUBJ"][r].astype(np.int64)).cuda()).float().cpu().numpy()  # 長度與主體方顏色；結果取回 CPU
    net.train()  # 回到訓練模式
    return out  # 每局的向量


def sym_table():  # 棋盤 8 種對稱的落點對照表
    """[8, 363]: where a point goes under each board symmetry (361 = pass and 362 = padding stay); as experiments/pid_seq_train.py --sym-aug."""
    yy, xx = np.divmod(np.arange(361), 19)  # 每個點的列與行
    maps = [(xx, yy), (18 - xx, yy), (xx, 18 - yy), (18 - xx, 18 - yy), (yy, xx), (18 - yy, xx), (yy, 18 - xx), (18 - yy, 18 - xx)]  # 8 種對稱後的 (x, y)
    return np.stack([np.concatenate([b_ * 19 + a_, [361, 362]]) for a_, b_ in maps]).astype(np.int16)  # 換回點編號，pass 與補齊保持原值


def finetune(net, A, owner, steps_out, save, lr, p_ep, m_ep, seed, log, horizon=None, mix=None, warm=50, views=1, sym_aug=False):  # 題池微調的主迴圈
    """A: arrays of every pool game; owner [n] candidate index or -1 (question game). Calls save(step, vectors) at steps_out."""
    A, mix = _exact(net, A), _exact(net, mix)  # 精確視窗網路先打包顏色
    rng = np.random.default_rng(seed)  # 抽情節用的亂數
    torch.manual_seed(seed)  # 固定網路端的亂數
    cand = np.flatnonzero(owner >= 0)  # 候選棋局的索引
    C = int(owner.max()) + 1  # 候選人數
    by = [cand[owner[cand] == c] for c in range(C)]  # 每位候選人的棋局索引
    m_ep = min(m_ep, min(len(b) for b in by))  # 每人取的局數不超過局數最少的候選人
    p_ep = min(p_ep, C)  # 每個情節的人數不超過候選人數
    everything = np.arange(len(owner))  # 題池全部棋局（輸出向量用）
    save0 = save  # 保留原本的存檔函式
    SYM = sym_table() if views > 1 or sym_aug else None  # 需要對稱時才建表
    rng_sym = np.random.default_rng(seed + 7919)  # 對稱用獨立的亂數流：有無增強抽到的情節都相同

    def save(step, vec):  # 有 --views k 時，另存題池在棋盤對稱 1 .. k-1 下的向量（後綴 v<i>）
        save0(step, vec, "")  # 原方向的向量
        for i in range(1, views):  # 其他對稱視角
            B = dict(A)  # 淺複製，只換 POS
            B["POS"] = SYM[i][A["POS"].astype(np.int64)]  # 套用第 i 種對稱
            save0(step, embed(net, B, everything), f"v{i}")  # 存這個視角的向量
    mix_by = None  # 預設不混入其他玩家
    if mix is not None:  # 每個情節混入另一批玩家（佔一半；--mix-fold 為訓練玩家，--mix-original 為同一邊原題池的候選人）：學題池的同時保住一般的辨識力
        m_own = mix["owner"]  # 混入棋局的玩家索引
        ids = [c for c in np.unique(m_own) if (m_own == c).sum() >= m_ep]  # 局數足夠的玩家
        mix_by = [np.flatnonzero(m_own == c) for c in ids]  # 每位混入玩家的棋局索引
        log(f"  mixing in {len(mix_by)} training players ({len(m_own)} games)")  # 回報混入的規模
    if 0 in steps_out:  # 要存未微調的向量
        save(0, embed(net, A, everything))  # 第 0 步 = 原網路
    last = max(steps_out)  # 微調的總步數
    if last == 0:  # 只要第 0 步
        return  # 不必訓練
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)  # 微調網路的全部參數（AdamW，權重衰減 0.05）
    hz = horizon or last  # 餘弦排程跑 `horizon` 步，即使微調較早停止（11-12：第一次的 t = 100 來自 1,000 步的排程）
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, hz) / hz)))  # 線性暖身乘上餘弦衰減
    net.train()  # 訓練模式
    fwd = _training_forward(net)  # 訓練用的前向傳遞（可能是 DataParallel）
    if fwd is not net:  # 用了多張 GPU
        log(f"  training forward across {torch.cuda.device_count()} GPUs (--dp)")  # 回報
    ar, tl, ta, t0 = np.arange(p_ep), 0.0, 0.0, time.time()  # 情節內的玩家編號；損失與正確率的累計；計時
    for step in range(last):  # 每一步一個情節
        n_pool = p_ep if mix_by is None else p_ep // 2  # 情節中題池候選人的人數
        pl = rng.choice(C, n_pool, replace=False)  # 抽候選人
        rows = np.concatenate([rng.choice(by[c], m_ep, replace=False) for c in pl])  # 每人抽 m_ep 局
        batch = {k: A[k][rows] for k in ("GL", "POS", "MOV", "LEN", "SUBJ")}  # 組成這個情節的輸入
        if mix_by is not None:  # 混入另一半的玩家
            mrows = np.concatenate([rng.choice(mix_by[c], m_ep, replace=False) for c in rng.choice(len(mix_by), p_ep - n_pool, replace=False)])  # 抽混入玩家與其棋局
            batch = {k: np.concatenate([batch[k], mix[k][mrows]]) for k in batch}  # 接在題池候選人之後
        if sym_aug:  # 每步一種隨機棋盤對稱，套在落點上（同 experiments/pid_seq_train.py --sym-aug）
            batch["POS"] = SYM[rng_sym.integers(8)][batch["POS"].astype(np.int64)]  # 套用對稱
        with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 前向傳遞
            z = fwd(torch.as_tensor(batch["GL"]).cuda(), torch.as_tensor(batch["POS"].astype(np.int64)).cuda(), torch.as_tensor(batch["MOV"].astype(np.int64)).cuda(),  # 逐手向量、落點與顏色
                    torch.as_tensor(batch["LEN"].astype(np.int64)).cuda(), torch.as_tensor(batch["SUBJ"].astype(np.int64)).cuda())  # 長度與主體方顏色
        z = z.float().reshape(p_ep, m_ep, m88.D_OUT)  # [玩家, 局, 128]
        tot = z.sum(1)  # 每人各局向量的和（原型）
        sc = net.log_scale.exp()  # 網路學到的 logit 縮放倍數（相當於溫度的倒數）
        logits = sc * z.reshape(-1, m88.D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T  # 每局對每位玩家原型的相似度
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1)  # 自己那一局不算進自己玩家的原型
        tgt = torch.as_tensor(np.repeat(ar, m_ep)).cuda()  # 每局的正解玩家
        logits = logits.scatter(1, tgt[:, None], own[:, None])  # 正解那一欄換成排除自己後的相似度
        loss = torch.nn.functional.cross_entropy(logits, tgt)  # 情節內的交叉熵
        opt.zero_grad(set_to_none=True)  # 清掉上一步的梯度
        loss.backward()  # 反向傳播
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪
        opt.step()  # 更新權重
        sched.step()  # 更新學習率
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())  # 累計損失與情節內的 top-1
        if (step + 1) % 50 == 0:  # 每 50 步回報一次
            log(f"  step {step + 1}/{last}: loss {tl / 50:.4f}, episode top-1 {ta / 50:.3f} of {p_ep}, {(time.time() - t0) / (step + 1):.2f} s/step")  # 印出平均損失、正確率與速度
            tl = ta = 0.0  # 累計歸零
        if step + 1 in steps_out:  # 到了要存的步數
            save(step + 1, embed(net, A, everything))  # 存題池全部棋局的向量


ARCH0 = (m88.D, m88.LAYERS, m88.HEADS, m88.FFN)  # m88 預設的網路大小，載入其他大小後要還原
# 依檢查點記錄的名稱（pid_seq_train.py --sup-name）選有監督的特徵集：官方題池的分段檔、競賽
# 訓練棋局的分段檔（sim 模式）、該特徵集的統計檔
SUP_SETS = {None: (["/work/u2507115/pid_v2/sup_pools/ply"], ["/work/u2507115/pid_v2/probeb/ply", "/work/u2507115/pid_v2/probec/ply"], "/work/u2507115/pid_v2/probeb/stats.npz"),  # 檢查點沒記錄名稱時用的特徵集（sup_pools／probeb／probec）
            "js": (["/work/u2507115/pid_v2/supj/pools"], ["/work/u2507115/pid_v2/probep/ply_js", "/work/u2507115/pid_v2/supj/comp"], "/work/u2507115/pid_v2/probep/stats_js.npz")}  # 檢查點記錄名稱為 js 時用的特徵集（supj／probep）


def load_net(ckpt, state="best"):  # 載入一個基礎檢查點
    st = torch.load(ckpt, map_location="cpu", weights_only=False)  # 讀檢查點（含設定）
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = st.get("arch", ARCH0)  # 以 pid_seq_train.py --arch 訓練的網路依其自己的大小重建
    net = m88.build_model()  # 依目前的大小建網路
    m88.D, m88.LAYERS, m88.HEADS, m88.FFN = ARCH0  # 還原預設大小
    if st.get("win"):  # 帶視窗輸入的網路（pid_seq_train_w.py --shape-win，METHOD_NOTES 2.170）
        import pid_win  # 視窗編碼器與分派
        pid_win.wrap(net, st["win"])  # 依存下的設定加上視窗編碼器
    net.load_state_dict(st["best_state" if state == "best" else "net"])  # 載入最佳或最後的權重
    net.sup_cfg = st.get("sup")  # pid_seq_train.py --sup-dims：帶有監督特徵的輸入維度（METHOD_NOTES 2.98）
    return net.cuda()  # 搬到 GPU


def main(argv=None):  # 命令列進入點
    global DP  # --dp 要改全域旗標
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--mode", required=True, choices=("sim", "official", "oldsim", "orig"))  # 四種題池
    ap.add_argument("--side", choices=("pub", "priv"), default="pub", help="oldsim: which ORIGINAL pool")  # 原題池的哪一邊（oldsim／orig）
    ap.add_argument("--ply-original", default="/work/u2507115/pid_v2/ply_original")  # 原題池候選人的逐手特徵
    ap.add_argument("--ply-origq", default="/work/u2507115/pid_v2/ply_origq", help="orig mode: per-ply parts of the ORIGINAL pools' question games (pid_v2_ply_extract.py --original-questions)")  # 原題池題目棋局的逐手特徵
    ap.add_argument("--fold", type=int, default=1)  # sim 模式的折
    ap.add_argument("--seed", type=int, default=0)  # sim 模式的題池種子，也決定微調的亂數
    ap.add_argument("--pool", choices=("pub2", "priv2"), default="pub2")  # official 模式的新增池
    ap.add_argument("--net", type=int, default=-1)  # 網路編號 v（讀 ckpt_<v><後綴>.pt）：1–4 為訓練時留出該折的網路，-1 為沒有留出任何一折的網路
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 100, 300, 600, 1000])  # 要輸出向量的步數
    ap.add_argument("--lr", type=float, default=1e-4)  # 學習率
    ap.add_argument("--horizon", type=int, default=None, help="length of the cosine schedule (default: the last step asked for)")  # 餘弦排程的長度
    ap.add_argument("--tag", default="", help="suffix of the output names (a recipe label)")  # 輸出檔名的後綴
    ap.add_argument("--mix-fold", type=int, default=0, help="mix training players of this fold into every episode (0 = none; the network must not be scored on that fold)")  # 混入某一折的訓練玩家
    ap.add_argument("--p-ep", type=int, default=256)  # 每個情節的玩家數
    ap.add_argument("--m-ep", type=int, default=8)  # 每位玩家的局數
    ap.add_argument("--ckpt-dir", default=str(AICUP1 / "results/m88"))  # 基礎檢查點的資料夾
    ap.add_argument("--ckpt-suffix", default="", help="checkpoint file ckpt_<v><suffix>.pt")  # 檢查點檔名的後綴
    ap.add_argument("--ply-dir", default="/work/u2507115/pid_v2/ply_official")  # 新增池的逐手特徵
    ap.add_argument("--tests-dir", default=str(R2 / "data" / "tests_1111"))  # 官方 v2 檔所在資料夾
    ap.add_argument("--out", default="/work/u2507115/pid_v2/seq_ft")  # 向量的輸出資料夾
    ap.add_argument("--state", choices=("best", "last"), default="best", help="the checkpoint's state with the best held-out top-1 on seen players, or the one after its final step")  # 用檢查點的哪一份權重
    ap.add_argument("--holdout", type=int, default=0, help="official mode: games per candidate kept out of the fine-tune (pseudo-pool; 0 = the real pool)")  # 擬題池：每人留出的局數
    ap.add_argument("--holdout-seed", type=int, default=0)  # 留局的抽樣種子（oldsim 模式也用它切模擬題池）
    ap.add_argument("--views", type=int, default=1, help="1-8: also write the vectors of the whole pool under that many board symmetries (test-time views of a network "  # 另存幾種對稱視角的向量
                                                         "trained with --sym-aug; the same symmetry for every game of the pool; METHOD_NOTES 2.59)")  # 說明文字的後半
    ap.add_argument("--sym-aug", action="store_true", help="one random board symmetry per fine-tune step (for networks trained with --sym-aug; METHOD_NOTES 2.69)")  # 微調時加對稱增強
    ap.add_argument("--dp", action="store_true", help="split the fine-tune's training forward over the visible GPUs (two or more); for networks whose episode does not fit one card")  # 訓練前向傳遞分散到多張 GPU
    ap.add_argument("--warm", type=int, default=50, help="steps of the linear learning-rate warm-up (50: a 50-step fine-tune never leaves the warm-up; METHOD_NOTES 2.55)")  # 學習率線性暖身的步數
    ap.add_argument("--mix-original", default=None, help="official mode: directory of scripts/pid_v2_ply_extract.py --original; the candidates of the ORIGINAL pool of the same side "  # 混入同一邊原題池的候選人
                                                           "(pub for pub2, priv for priv2; never the other side) fill half of every episode (METHOD_NOTES 2.54)")  # 說明文字的後半
    a = ap.parse_args(argv)  # 解析參數
    DP = a.dp  # 設定全域旗標
    t0 = time.time()  # 計時起點
    log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 不存在就建立
    if a.mode == "sim":  # 訓練玩家的模擬題池
        import aicup_pid  # 訓練資料的位置
        from aicup_pid import v2  # 每局列與訓練棋局清單
        from aicup_pid.paths import FOLDS_CSV  # 玩家分折表
        from pid_v2_sim2 import build  # 與 pid_v2_sim2.py 相同的題池建法
        games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")  # 訓練棋局標上所屬的折
        rows = v2.load_side_rows(AICUP1)  # 每局列（決定哪些局可用）
        z = np.load(R2 / "data" / "seq" / f"fold{a.fold}.npz")  # 該折已有序列向量的棋局
        g = games[(games.fold == a.fold) & games.game_key.isin(rows.keys()) & games.game_key.isin(set(z["key"].astype(str)))]  # 該折可用的棋局
        cands, questions, _owner, _block, _sgfs, _info = build(g, a.seed)  # 建模擬題池
        players = sorted(cands)  # 候選人排序
        own = {k: i for i, p in enumerate(players) for k, _ in cands[p]}  # 候選棋局 → 候選人索引
        need = set(own) | {k for q in questions for k, _ in q}  # 題池需要的全部棋局
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{a.fold}").glob("games_s*_p*.csv"))]  # 該折的逐手特徵分段檔
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        with mp.get_context("fork").Pool(12) as pool:  # 平行讀取分段檔
            for i in range(0, len(files), 48):  # 每次 48 個檔
                for key, _pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):  # 讀成補齊後的陣列
                    m = np.array([k in need for k in key])  # 只留題池用到的局
                    if m.any():  # 這個分段檔有需要的局
                        for name, arr in (("key", key), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):  # 各欄
                            parts[name].append(arr[m])  # 收集
        A = {k: np.concatenate(x) for k, x in parts.items()}  # 串接成題池的陣列
        if set(A["key"].tolist()) != need:  # 有局沒有逐手向量
            raise ValueError(f"{len(need) - len(set(A['key'].tolist()))} pool games without per-ply vectors")  # 停下
        tag, ckpt = f"sim_f{a.fold}_s{a.seed}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.fold}{a.ckpt_suffix}.pt"  # 輸出名稱；用沒看過該折玩家的網路
    elif a.mode == "oldsim":  # aicup_pid.v2.original_sim 的模擬題池：只在其 10 局候選集上微調 -> oldsim_<side>_s<seed>_net<v><tag>_t<step>.npz
        from aicup_pid import v2  # 只在原題池模擬時匯入 v2 模組（original_sim）
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        for f in sorted(glob.glob(f"{a.ply_original}/ply_*.npz")):  # 原題池候選人的逐手分段檔
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀一個分段檔
            key = z["key"].astype(str)  # 棋局鍵
            m = np.array([k.split(":")[2].startswith(a.side + "_g_") for k in key])  # 只留這一邊的候選棋局
            if m.any():  # 有這一邊的局
                for name in parts:  # 各欄
                    parts[name].append(key[m] if name == "key" else z[name][m])  # 收集
        A = {k: np.concatenate(x) for k, x in parts.items()}  # 串接
        cands, _q, _t, _s = v2.original_sim(Path(a.tests_dir) / f"player_identification_candidates_{'public' if a.side == 'pub' else 'private'}.csv", set(A["key"].tolist()), a.holdout_seed)  # 切出模擬題池
        players = sorted(cands)  # 候選人排序
        own = {k: i for i, p in enumerate(players) for k, _ in cands[p]}  # 只有 10 局候選集算候選棋局
        tag, ckpt = f"oldsim_{a.side}_s{a.holdout_seed}_net{a.net}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.net}{a.ckpt_suffix}.pt"  # 輸出名稱與檢查點
    elif a.mode == "orig":  # METHOD_NOTES 2.240：某一邊「真正的」原題池：在其候選人已存的棋局（ply_original）上微調，輸出這些棋局與該題池題目棋局的向量 -> orig_<side>_net<v><tag>_t<step>.npz
        tagname = "public" if a.side == "pub" else "private"  # 檔名用的邊
        c = pd.read_csv(Path(a.tests_dir) / f"player_identification_candidates_{tagname}.csv", dtype=str, keep_default_na=False)  # 原題池候選檔
        players = sorted(c.player_id.unique())  # 候選人排序
        pidx = {p: i for i, p in enumerate(players)}  # 候選人 → 索引
        own_all = {f"c:{i}:{r.game_id}": pidx[r.player_id] for i, r in enumerate(c.itertuples())}  # 每局候選棋局的擁有者
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        for f in sorted(glob.glob(f"{a.ply_original}/ply_*.npz")) + sorted(glob.glob(f"{a.ply_origq}/ply_*.npz")):  # 候選人與題目棋局的分段檔
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀一個分段檔
            key = z["key"].astype(str)  # 棋局鍵
            m = np.array([(k in own_all and k.split(":")[2].startswith(a.side + "_g_")) or k.startswith(f"q:{a.side}_q_") for k in key])  # 這一邊的候選棋局與題目棋局
            if m.any():  # 有需要的局
                for name in parts:  # 各欄
                    parts[name].append(key[m] if name == "key" else z[name][m])  # 收集
        A = {k: np.concatenate(x) for k, x in parts.items()}  # 串接
        own = {k: own_all[k] for k in A["key"].tolist() if k in own_all}  # 有逐手特徵的候選棋局
        assert len(set(own.values())) == len(players), "a candidate of the original pool has no stored game"  # 每位候選人至少要有一局
        assert int(sum(k.startswith("q:") for k in A["key"].tolist())) > 0, "no question game of the original pool: run pid_v2_ply_extract.py --original-questions"  # 必須有題目棋局
        tag, ckpt = f"orig_{a.side}_net{a.net}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.net}{a.ckpt_suffix}.pt"  # 輸出名稱與檢查點
    else:  # official：官方新增池
        tagname = "public" if a.pool == "pub2" else "private"  # 檔名用的邊
        c = pd.read_csv(Path(a.tests_dir) / f"player_identification_candidates2_{tagname}.csv", dtype=str, keep_default_na=False)  # 新增池候選檔
        players = sorted(c.player_id.unique())  # 候選人排序
        pidx = {p: i for i, p in enumerate(players)}  # 候選人 → 索引
        own = {f"c:{i}:{r.game_id}": pidx[r.player_id] for i, r in enumerate(c.itertuples())}  # 候選棋局 → 擁有者
        parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        for f in sorted(glob.glob(f"{a.ply_dir}/ply_*.npz")):  # 新增池的逐手分段檔
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀一個分段檔
            key = z["key"].astype(str)  # 棋局鍵
            m = np.array([(k in own) or k.startswith(f"q:{a.pool}_") for k in key])  # 這個題池的候選棋局與題目棋局
            if m.any():  # 有需要的局
                for name in parts:  # 各欄
                    parts[name].append(key[m] if name == "key" else z[name][m])  # 收集
        A = {k: np.concatenate(x) for k, x in parts.items()}  # 串接
        tag, ckpt = f"official_{a.pool}_net{a.net}{a.tag}", Path(a.ckpt_dir) / f"ckpt_{a.net}{a.ckpt_suffix}.pt"  # 輸出名稱與檢查點
        if a.holdout:  # 擬題池模式
            from aicup_pid import v2  # candidate_holdout 與 official_pool
            cands, _q, _ids, _sgfs = v2.official_pool(Path(a.tests_dir) / f"player_identification_candidates2_{tagname}.csv", Path(a.tests_dir) / f"player_identification_test2_{tagname}.csv")  # 與擬題池評估相同的候選清單
            assert {k for g in cands.values() for k, _ in g} == set(own), "candidate keys differ from aicup_pid.v2.official_pool"  # 兩邊的候選鍵必須一致
            held = {k for g in v2.candidate_holdout(cands, a.holdout, a.holdout_seed).values() for k, _ in g}  # 留出的棋局（與擬題池評估相同的抽法）
            own = {k: i for k, i in own.items() if k not in held}  # 留出的局不進微調，當成題目
            tag = f"offsim_{a.pool}_h{a.holdout}s{a.holdout_seed}_net{a.net}{a.tag}"  # 擬題池的輸出名稱
    owner = np.array([own.get(k, -1) for k in A["key"]])  # 每局的擁有者索引；題目棋局與留出的局為 -1，不進微調
    log(f"{tag}: {len(owner)} games ({int((owner >= 0).sum())} candidate games of {len(players)} players, {int((owner < 0).sum())} question games); network {ckpt.name}")  # 回報題池規模與檢查點
    net = load_net(ckpt, a.state)  # 載入基礎檢查點
    if net.sup_cfg:  # 題池棋局要有與網路訓練時相同的有監督特徵
        assert not a.mix_fold and not a.mix_original, "the mixing options have no supervised features"  # 混入選項沒有有監督特徵
        sys.path.insert(0, str(R2 / "scripts"))  # 確保找得到 pid_sup
        import pid_sup  # 有監督逐手特徵的工具
        pools_, train_, stats_ = SUP_SETS[net.sup_cfg.get("name")]  # 依檢查點記錄的名稱選特徵集
        stores = train_ if a.mode == "sim" else pools_  # sim 用訓練棋局的特徵，其餘用官方題池的特徵
        idx_s, D_s = pid_sup.dense(pid_sup.load_sup(stores, stats_), A["key"].tolist(), A["GL"].shape[1])  # 對齊題池棋局順序的特徵
        if (idx_s < 0).any():  # 有局沒有有監督特徵
            raise KeyError(f"{int((idx_s < 0).sum())} pool games without supervised per-ply features, e.g. {A['key'][idx_s < 0][:3].tolist()}")  # 停下
        z_s = D_s[idx_s].astype(np.float32)  # 依題池順序取出
        A["GL"] = pid_sup.put(A["GL"].astype(np.float32), z_s, net.mu.float().cpu().numpy(), net.sd.float().cpu().numpy(), tuple(net.sup_cfg["dims"]), bool(net.sup_cfg["only"]))  # 寫進網路指定的輸入維度
        log(f"supervised features in input dimensions {net.sup_cfg['dims']}{' (alone)' if net.sup_cfg['only'] else ''} for all {len(idx_s)} games")  # 回報

    def save(step, vec, view=""):  # 寫出某一步的題池向量
        dest = out / f"{tag}{view}_t{step}.npz"  # 檔案只會完整出現或不存在：其他工作會列出這個資料夾（平行的工作鏈）
        tmp = dest.with_name(dest.name + ".tmp.npz")  # 先寫暫存檔
        np.savez(tmp, key=A["key"], emb=vec.astype(np.float16), step=np.array(step))  # 鍵、fp16 向量與步數
        os.replace(tmp, dest)  # 原子換名
        log(f"  vectors after {step} steps written{' (symmetry view ' + view[1:] + ')' if view else ''}")  # 回報

    mix = None  # 預設不混入
    if a.mix_fold:  # 混入某一折的訓練玩家
        files = [f for d in m88.DIRS for f in sorted((d / f"fold{a.mix_fold}").glob("games_s*_p*.csv"))]  # 該折的逐手特徵分段檔
        parts = {x: [] for x in ("pid", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        with mp.get_context("fork").Pool(12) as pool:  # 平行讀取
            for i in range(0, len(files), 48):  # 每次 48 個檔
                for _key, pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):  # 讀成補齊後的陣列
                    for name, arr in (("pid", pid), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):  # 各欄
                        parts[name].append(arr)  # 收集
        mix = {k: np.concatenate(x) for k, x in parts.items()}  # 串接
        mix["owner"] = np.unique(mix.pop("pid"), return_inverse=True)[1]  # 玩家編號換成索引
    if a.mix_original:  # 混入同一邊原題池的候選人
        assert a.mode == "official" and mix is None  # 只用於新增池，且不與 --mix-fold 併用
        side = a.pool[:-1]  # 題池名稱去掉結尾的 2：pub2 -> pub、priv2 -> priv
        c0 = pd.read_csv(Path(a.tests_dir) / f"player_identification_candidates_{'public' if side == 'pub' else 'private'}.csv", dtype=str, usecols=["game_id", "player_id"])  # 原題池候選檔（只讀兩欄）
        own0 = {f"c:{i}:{g}": p for i, (g, p) in enumerate(zip(c0.game_id, c0.player_id))}  # 鍵 → 候選人編號
        parts = {x: [] for x in ("pid", "GL", "POS", "MOV", "LEN", "SUBJ")}  # 收集各欄
        for f in sorted(glob.glob(f"{a.mix_original}/ply_*.npz")):  # 原題池候選人的逐手分段檔
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀一個分段檔
            key = z["key"].astype(str)  # 棋局鍵
            m = np.array([k.split(":")[2].startswith(side + "_g_") for k in key])  # 只留同一邊（Public／Private 不混）
            if m.any():  # 有需要的局
                parts["pid"].append(np.array([own0[k] for k in key[m]]))  # 擁有者
                for name in ("GL", "POS", "MOV", "LEN", "SUBJ"):  # 各欄
                    parts[name].append(z[name][m])  # 收集
        mix = {k: np.concatenate(x) for k, x in parts.items()}  # 串接
        assert all(p.startswith(side + "_p_") for p in np.unique(mix["pid"]))  # 確認沒有混到另一邊
        mix["owner"] = np.unique(mix.pop("pid"), return_inverse=True)[1]  # 玩家編號換成索引
        log(f"original pool '{side}': {len(mix['owner'])} candidate games of {int(mix['owner'].max()) + 1} candidates in every episode's second half")  # 回報
    finetune(net, A, owner, sorted(set(a.steps)), save, a.lr, a.p_ep, a.m_ep, 1000 + a.seed, log, a.horizon, mix, a.warm, a.views, a.sym_aug)  # 執行微調並存向量
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
