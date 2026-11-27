#!/usr/bin/env python
"""Per-ply trunk vectors of the v2 release's added pools, in the layout of the rank session's sequence identity model
(AI_CUP_2026-1 scripts/m_series/m88_seqid.py read_part): per game the first 256 plies of BOTH players, GL [256, 128] fp16 (board-pooled
64 + played-point 64), POS (tensor position, 361 pass, 362 padding), MOV (0 Black / 1 White), LEN, SUBJ (the player's colour).
Same replay, network and projections as scripts/pid_trunk_extract.py / AI_CUP_2026-1 scripts/trunk_extract.py. Sharded by
shards.sh (--shard i --nshards N); one .npz per part, resumable. Keys as pid_trunk_extract.py --official2."""
import argparse  # 解析命令列參數
import multiprocessing as mp  # 用多個行程平行重播棋局
import os  # os.replace 原子換檔
import sys  # 設定 sys.path 與結束碼
import time  # 計時
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列與 .npz 存檔
import torch  # 在 GPU 上執行主幹

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
for d in (ROOT / "src", ROOT / "scripts"):  # 本儲存庫的套件與腳本
    if str(d) not in sys.path:  # 尚未加入時
        sys.path.insert(0, str(d))  # 加到搜尋路徑最前面
import aicup_pid  # noqa: E402,F401  匯入時把 AI_CUP_2026-1 的 src 與 scripts 放上 sys.path
import trunk_extract as te  # noqa: E402  AI_CUP_2026-1 的重播、投影與向量計算
from aicup.humansl import load_humansl  # noqa: E402  載入人類棋力網路（Human SL）
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402  中性詮釋資料列與主幹輸出
from aicup.submission import check_game  # noqa: E402  判斷棋局能否使用（19 路、能合法重播、該方至少一手）
from pid_trunk_extract import official2_tasks  # noqa: E402  新增池的任務清單與鍵

T_MAX = 256  # 每局最多取的手數（與 m88_seqid 相同）


def original_tasks(tests_dir, per_candidate):  # --original：原題池候選人的棋局
    """{key: (sgf, 'B')} of the ORIGINAL pools' candidate games (pub_ / priv_; Black only), the first `per_candidate` usable games
    of every candidate in file order; keys c:<row in its file>:<game_id> as pid_trunk_extract.py --official. No question game."""
    import pandas as pd  # 只有這個模式需要
    from collections import Counter  # 計算每位候選人已收的局數
    out = {}  # 鍵 → (棋譜, 'B')
    for tag in ("public", "private"):  # 兩個原題池
        c = pd.read_csv(Path(tests_dir) / f"player_identification_candidates_{tag}.csv", dtype=str, keep_default_na=False)  # 原題池候選檔
        seen = Counter()  # 每位候選人已收的局數
        for i, r in enumerate(c.itertuples()):  # 依檔案順序
            if seen[r.player_id] < per_candidate and check_game(r.sgf_content, "B")[0]:  # 這位候選人還沒收滿，且棋局可用
                out[f"c:{i}:{r.game_id}"] = (r.sgf_content, "B")  # 收進任務
                seen[r.player_id] += 1  # 計數加一
    return out  # 回傳任務


def original_question_tasks(tests_dir):  # --original-questions：原題池的題目棋局
    """{key: (sgf, colour)} of the ORIGINAL pools' QUESTION games (pub_q_ / priv_q_; METHOD_NOTES 2.240): keys q:<question_id>:<i>
    as the added pools' question games (pid_trunk_extract.official2_tasks)."""
    import pid_trunk_extract as pte  # 借用它已匯入的 pools 模組
    out = {}  # 鍵 → (棋譜, 顏色)
    for tag in ("public", "private"):  # 兩個原題池
        for q in pte.pools.read_questions(Path(tests_dir) / f"player_identification_test_{tag}.csv"):  # 逐題讀取
            for i, (sgf, colour) in enumerate(q["games"]):  # 題目中的每一局
                if check_game(sgf, colour or "B")[0]:  # 只收可用的棋局；未標顏色時以黑方檢查
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")  # 題目棋局的鍵；未標顏色時視為黑方
    return out  # 回傳任務


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--tests-dir", required=True)  # 官方 CSV 的資料夾
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--workers", type=int, default=10)  # 重播棋局的行程數
    ap.add_argument("--shard", type=int, default=0)  # 分片編號（shards.sh 給）
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--part-size", type=int, default=200)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=512)  # 主幹推論的批次大小
    ap.add_argument("--original", type=int, default=0, help="instead of the added pools: this many candidate games per candidate of the ORIGINAL pools (METHOD_NOTES 2.54)")  # 改抽原題池候選人的棋局：每人依檔案順序取這麼多局可用的局（0 = 不用）
    ap.add_argument("--original-questions", action="store_true", help="instead of the added pools: the QUESTION games of the ORIGINAL pools (METHOD_NOTES 2.240)")  # 改抽原題池的題目棋局
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 不存在就建立
    assert not (a.original and a.original_questions)  # 兩種原題池模式不能同時用
    tasks = original_question_tasks(a.tests_dir) if a.original_questions else original_tasks(a.tests_dir, a.original) if a.original else official2_tasks(a.tests_dir)  # 依模式取得要抽的棋局
    mine = sorted(tasks)[a.shard::a.nshards]  # 本分片負責的局
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]  # 切成固定局數的分段
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"ply_s{a.shard:03d}_p{k:04d}.npz").exists()]  # 已寫出的分段檔略過（續跑）
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(tasks)} games, {len(todo)} parts to do", flush=True)  # 回報工作量
    if not todo:  # 全部做完了
        return 0  # 直接結束
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]  # 讀網路設定給特徵抽取用
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重播用的行程池，各自建立 Features
    try:  # 確保最後結束行程池
        model, _ = load_humansl(str(te.CKPT), "cuda")  # 載入人類棋力網路（Human SL）
        model.eval()  # 推論模式
        rows = neutral_rows()  # 中性詮釋資料列（黑、白各一列）
        proj_g = proj_l = None  # 隨機投影矩陣，處理第一個分段時建立
        for k, part in todo:  # 逐個分段處理
            done = list(pool.imap(te.replay, [(q, *tasks[q]) for q in part], chunksize=1))  # 平行重播雙方的每一手，保持原順序
            n = len(done)  # 這個分段的局數
            GL = np.zeros((n, T_MAX, 128), np.float16)  # 每手的主幹向量，補齊處為零
            POS = np.full((n, T_MAX), 362, np.int16)  # 落點，預設 362 = 補齊
            MOV = np.zeros((n, T_MAX), np.int8)  # 每手的行棋方：0 黑、1 白
            LEN = np.zeros(n, np.int16)  # 每局實際取用的手數
            SUBJ = np.array([0 if tasks[d[0]][1] == "B" else 1 for d in done], np.int8)  # 要辨識的玩家所執的顏色
            lens = [min(len(d[3]), T_MAX) for d in done]  # 每局最多取 256 手
            bins = np.concatenate([d[1][:m] for d, m in zip(done, lens)])  # 串接各局前 m 手的盤面特徵
            globs = np.concatenate([d[2][:m] for d, m in zip(done, lens)])  # 全域特徵
            pos = np.concatenate([d[3][:m] for d, m in zip(done, lens)])  # 落點
            mover = np.concatenate([d[4][:m] for d, m in zip(done, lens)])  # 行棋方
            if proj_g is None:  # 第一個分段：建立投影
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 只為取得主幹的通道數
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(bins[:1]).cuda(), torch.as_tensor(globs[:1]).cuda(), torch.as_tensor(mover[:1]).cuda().long(), rows)  # 用一個局面跑主幹
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)  # 固定種子的隨機投影（與 pid_trunk_extract.py 相同）
            g, l = te.vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, a.batch)  # 每手投影後的全盤與落點向量
            start = 0  # 這局在逐手陣列中的起點
            for i, m in enumerate(lens):  # 逐局填入補齊後的陣列
                GL[i, :m, :64], GL[i, :m, 64:] = g[start:start + m], l[start:start + m]  # 前 64 維全盤、後 64 維落點
                POS[i, :m], MOV[i, :m], LEN[i] = pos[start:start + m], mover[start:start + m], m  # 落點、顏色與長度
                start += m  # 下一局的起點
            path = out / f"ply_s{a.shard:03d}_p{k:04d}.npz"  # 這個分段的輸出檔
            tmp = path.with_name(path.name + ".tmp.npz")  # 先寫暫存檔
            np.savez(tmp, key=np.array([d[0] for d in done]), GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ)  # 寫出這個分段檔
            os.replace(tmp, path)  # 原子換名：分段檔只會完整出現，續跑不會讀到寫一半的檔
            print(f"part {k + 1}/{len(parts)}: {n} games, {int(sum(lens))} plies ({time.time() - t0:.0f} s)", flush=True)  # 回報進度
    finally:  # 不論成功或失敗
        pool.terminate()  # 結束重播行程
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
