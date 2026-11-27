#!/usr/bin/env python
"""Per-ply supervised features of training games for probe B (METHOD_NOTES 2.88): the humanv0 trunk's board-pooled and
played-point vectors at the SUBJECT's plies (replay of scripts/pid_trunk_extract.py --fast), pushed through the projection of
scripts/pid_sup_fit.py instead of the random one. Parts <out>/sup_s<shard>_p<k>.npz: keys, n (plies per game), t (ply numbers),
z ([plies, K] float16, unstandardised). Resumable per part; shards as pid_trunk_extract.py.
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --proj proj.npz --out DIR --workers 10
--select: the external games of the selections of scripts/ext_fox_select.py instead (METHOD_NOTES 2.101; the subject is the
selection's colour; a record the replay cannot follow is stored without plies; a shard is the games whose key hashes to it, so
a process only holds its own share of the records).
"""
import argparse  # 解析命令列參數
import multiprocessing as mp  # 以 fork 的工作行程平行重播棋譜
import os  # 以原子改名寫出檔案
import sys  # 調整模組搜尋路徑與結束碼
import time  # 計時與速度回報
import zlib  # 以 crc32 雜湊棋局鍵來分配分片
from pathlib import Path  # 路徑操作

import numpy as np  # 陣列運算
import torch  # 讀取 KataGo 檢查點內的設定

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
for d in (ROOT / "src", ROOT / "scripts"):  # 本儲存庫的套件與腳本資料夾
    if str(d) not in sys.path:  # 尚未在搜尋路徑中
        sys.path.insert(0, str(d))  # 加到最前面
import aicup_pid  # noqa: E402  本儲存庫套件，提供訓練集路徑
import pid_trunk_extract as PT  # noqa: E402  （replay_target、full_vectors、read_keys；它匯入的模組會把 AI_CUP_2026-1 放上搜尋路徑）
from aicup.humansl import load_humansl  # noqa: E402  載入 Human SL 網路
from aicup.l1b import neutral_rows  # noqa: E402  黑白各一列的中性 metadata
from aicup.scoring import load_games  # noqa: E402  依棋局鍵讀取訓練棋譜

te = PT.te  # AI_CUP_2026-1 的 trunk_extract（檢查點路徑、工作行程初始化）
MAX_T = 0  # --max-plies（在 fork 工作行程池之前設定）


def replay_external(task):  # 外部棋譜的重播，失敗時當成沒有手
    """PT.replay_target for an external record: one the replay cannot follow counts as a game without plies; only the plies
    with a number below MAX_T are kept (the sequence networks read the first 256 plies)."""
    try:  # 外部棋譜可能有重播不了的內容
        d = PT.replay_target(task)  # 只計算主體那一方各手的輸入特徵
    except Exception:  # noqa: BLE001  任何重播錯誤
        return task[0], None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)  # 當成沒有手的棋局
    if MAX_T and len(d[3]):  # 有手數上限且這局有手
        ok = d[5] < MAX_T  # 手數編號在上限之內
        if not ok.any():  # 一手都不剩
            return d[0], None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)  # 當成沒有手的棋局
        d = (d[0], d[1][ok], d[2][ok], d[3][ok], d[4][ok], d[5][ok])  # 只留上限內的手
    return d  # 回傳 (key, bins, globs, pos, mover, t)


def main(argv=None):  # 抽取一個分片的逐手有監督特徵
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--keys", default=None, help="training game keys (text file)")  # 訓練棋局鍵的清單
    ap.add_argument("--pools", default=None, help="instead of --keys: the official pools of this tests folder - every game of the added pools, and the games of the original pools that have per-ply vectors in --original-ply")  # 改抽官方題池的棋局
    ap.add_argument("--original-ply", default="/work/u2507115/pid_v2/ply_original")  # 原題池逐手向量的資料夾
    ap.add_argument("--proj", required=True)  # pid_sup_fit.py 的投影檔
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--workers", type=int, default=10)  # 重播棋譜的工作行程數
    ap.add_argument("--shard", type=int, default=0)  # 本行程負責的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--part-size", type=int, default=200)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=512)  # GPU 前向計算的批次大小
    ap.add_argument("--max-parts", type=int, default=0)  # 本次處理滿這麼多段的局數就停（0 = 不限）
    ap.add_argument("--stats", default=None, help="npz (mean, sd) of pid_sup.write_stats: the parts are written as int8 `zq` = standardised value x 20 (half the size)")  # 量化成 int8 時用的尺度
    ap.add_argument("--skip-keys", default=None, help="a key list whose games are left out (already extracted elsewhere)")  # 要略過的棋局
    ap.add_argument("--select", nargs="+", default=None, help="instead of --keys: external games, the select.csv files of scripts/ext_fox_select.py (game_key, sgf_content, color)")  # 改抽外部棋局
    ap.add_argument("--max-plies", type=int, default=0, help="with --select: keep only the subject's plies whose ply number is below this (0: all)")  # 外部棋局只留前幾手
    ap.add_argument("--scalars", action="store_true", help="append four policy scalars to every ply's features: surprisal, log(1 + rank), entropy, distance to the policy's top point (METHOD_NOTES 2.102)")  # 每手再加 4 個 policy 純量
    ap.add_argument("--ckpt", default=None, help="another KataGo checkpoint than humanv0 (probe K, METHOD_NOTES 2.119)")  # 改用其他 KataGo 檢查點
    ap.add_argument("--compress", action="store_true", help="write the parts deflate-compressed (np.load reads them the same; METHOD_NOTES 2.164)")  # 以 deflate 壓縮寫出
    ap.add_argument("--out2", default=None, help="with --scalars and float parts: also write the same parts WITHOUT the scalars here (two stores from one pass)")  # 同時寫一份不含純量的儲存
    a = ap.parse_args(argv)  # 解析參數
    global MAX_T  # 工作行程讀的是模組層級的 MAX_T
    MAX_T = a.max_plies if a.select else 0  # 只對外部棋局截手數
    t0 = time.time()  # 計時起點
    P = np.load(a.proj)  # 讀入投影檔
    mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)  # 標準化參數與投影矩陣
    blocks = [str(b) for b in P["blocks"]] if "blocks" in P.files else ["g", "x"]  # 投影檔記錄的區塊（沒記錄時為 g、x）；"e"：policy 預期的落點向量
    use_pol = "e" in blocks or a.scalars  # 需要 policy 時改走 policy_vectors
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 確保存在
    tasks_all = None  # 棋局鍵 -> (棋譜, 顏色)；訓練棋局則為 None（之後依鍵讀取）
    if a.pools:  # 官方題池
        import glob  # 列出原題池的逐手向量檔
        tasks_all = PT.official2_tasks(a.pools)  # 新增池的全部可用棋局
        want = set()  # 原題池中已有逐手向量的棋局
        for f in glob.glob(f"{a.original_ply}/ply_*.npz"):  # 原題池的逐手向量檔
            if not f.endswith(".tmp.npz"):  # 略過寫到一半的暫存檔
                want.update(np.load(f)["key"].astype(str).tolist())  # 收集其棋局鍵
        old = PT.official_tasks(a.pools)  # 原題池的全部可用棋局
        tasks_all.update({k: v for k, v in old.items() if k in want})  # 只加入有逐手向量的那些
        keys = sorted(tasks_all)  # 固定順序
    elif a.select:  # 外部棋局
        import csv  # 讀取挑選檔
        csv.field_size_limit(10 ** 9)  # 欄位含整份棋譜，放寬長度上限
        tasks_all, n_rows = {}, 0  # 本分片的棋局；全部挑選檔的紀錄總數
        for f in a.select:  # 每個挑選檔
            with open(f, newline="") as fh:  # 開啟 CSV
                for r in csv.DictReader(fh):  # 每筆紀錄
                    n_rows += 1  # 計數
                    if zlib.crc32(r["game_key"].encode()) % a.nshards == a.shard:  # 棋局鍵雜湊到本分片
                        tasks_all[r["game_key"]] = (r["sgf_content"], r["color"])  # 只保留本分片的棋譜
        keys = sorted(tasks_all)  # 固定順序
        print(f"{n_rows} records in {len(a.select)} selections, {len(keys)} in this shard ({time.time() - t0:.0f} s)", flush=True)  # 回報紀錄數與本分片局數
    else:  # 訓練棋局
        keys = sorted(set(PT.read_keys(a.keys)) - (set(PT.read_keys(a.skip_keys)) if a.skip_keys else set()))  # 去掉要略過的棋局
    st = np.load(a.stats) if a.stats else None  # 量化用的尺度（沒有則寫 float16）
    mine = keys if a.select else keys[a.shard::a.nshards]  # 本分片的棋局（外部棋局已依雜湊分好）
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]  # 每 --part-size 局一段
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"sup_s{a.shard:03d}_p{k:04d}.npz").exists()]  # 尚未寫出的段（可續跑）
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do; projection of {'+'.join(blocks)} to {W.shape[1]} dims" + (" + 4 policy scalars" if a.scalars else ""), flush=True)  # 回報局數、段數與投影設定
    if not todo:  # 全部已完成
        return 0  # 不必載入網路
    config = torch.load(str(a.ckpt or te.CKPT), map_location="cpu", weights_only=a.ckpt is None)["config"]  # 檢查點內的模型設定，供工作行程建立特徵器
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重播棋譜的工作行程池
    try:  # 確保最後關閉工作池
        model, _ = load_humansl(str(a.ckpt or te.CKPT), "cuda")  # 把 Human SL 網路（或 --ckpt 指定的檢查點）載入 GPU
        model.eval()  # 推論模式
        rows = neutral_rows()  # 黑白各一列的中性 metadata
        done_games = empty = 0  # 已處理局數與沒有手的局數
        for k, part in todo:  # 每個未完成的段
            sgf = {q: tasks_all[q] for q in part} if tasks_all is not None else load_games(part, aicup_pid.TRAIN_DIR)  # 這一段的棋譜與顏色
            done = list(pool.imap(replay_external if a.select else PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))  # 平行重播，取主體各手的輸入特徵
            some = [d for d in done if len(d[3])]  # 有手的棋局
            empty += len(done) - len(some)  # 累計沒有手的局數
            if some:  # 至少一局有手
                bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])  # 合併各手的盤面與全域特徵
                pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])  # 合併落點與執色
                if use_pol:  # 需要 policy 的區塊或純量
                    fg, fl, fe, sc_, _bad = PT.policy_vectors(model, bins, globs, pos, mover, rows, a.batch)  # 全盤、落點、預期向量與 policy 純量
                    src = {"g": fg, "x": fl, "e": fe}  # 區塊名稱對應
                    z = ((np.hstack([src[b] for b in blocks]) - mu) / sd) @ W  # 依投影檔的區塊順序標準化後投影
                    if a.scalars:  # 加上 4 個 policy 純量
                        z = np.hstack([z, np.stack([sc_[:, 0], np.log1p(sc_[:, 1]), sc_[:, 2], sc_[:, 4]], axis=1)])  # 意外程度、log(1 + 名次)、熵、與 policy 首選點的距離（不取首選機率那一欄）
                else:  # 只用全盤與落點向量
                    fg, fl = PT.full_vectors(model, bins, globs, pos, mover, rows, a.batch)  # 未投影的兩個向量
                    z = ((np.hstack([fg, fl]) - mu) / sd) @ W  # 標準化後投影
            else:  # 整段都沒有手
                z = np.zeros((0, W.shape[1] + (4 if a.scalars else 0)), np.float32)  # 空的特徵陣列
            if not np.isfinite(z).all():  # 檢查數值
                raise ValueError(f"part {k}: non-finite features")  # 有非有限值就中止
            path = out / f"sup_s{a.shard:03d}_p{k:04d}.npz"  # 這一段的輸出檔
            tmp = path.with_name(path.name + ".tmp.npz")  # 先寫暫存檔
            feat = {"z": z.astype(np.float16)} if st is None else {"zq": np.clip(np.rint((z - st["mean"]) / st["sd"] * 20.0), -127, 127).astype(np.int8)}  # float16，或量化成 int8（標準化值 × 20）
            head = dict(keys=np.array([d[0] for d in done]), n=np.array([len(d[3]) for d in done], np.int32), t=np.concatenate([d[5] for d in done]).astype(np.int16))  # 棋局鍵、每局主體的手數與各手的手數編號
            if a.out2:  # 先寫第二份，讓一段只在兩份儲存都有時才算完成
                assert a.scalars and st is None, "--out2 goes with --scalars and float parts"  # 檢查選項組合
                p2 = Path(a.out2) / path.name  # 第二份的輸出檔
                p2.parent.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
                t2 = p2.with_name(p2.name + ".tmp.npz")  # 第二份的暫存檔
                np.savez(t2, z=z[:, :W.shape[1]].astype(np.float16), **head)  # 不含純量的特徵
                os.replace(t2, p2)  # 原子改名
            (np.savez_compressed if a.compress else np.savez)(tmp, **head, **feat)  # 寫出暫存檔（可選擇壓縮）
            os.replace(tmp, path)  # 原子改名：檔案出現即代表這一段完成
            done_games += len(part)  # 累計局數
            if (k % 10) == 0:  # 段編號是 10 的倍數時回報
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s" + (f", {empty} without plies" if empty else ""), flush=True)  # 進度與速度
            if a.max_parts and done_games >= a.max_parts * a.part_size:  # 本次已處理滿 --max-parts 段的局數
                break  # 提前結束
    finally:  # 無論成功與否
        pool.close()  # 不再送出工作
        pool.join()  # 等工作行程結束
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)  # 回報耗時
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值結束
