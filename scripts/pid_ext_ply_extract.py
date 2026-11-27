#!/usr/bin/env python
"""Per-ply trunk vectors of the external games chosen by scripts/ext_fox_select.py (METHOD_NOTES 2.46), in exactly the form
of the rank session's M90 parts (AI_CUP_2026-1 scripts/m_series/m90_ply_extract.py: same network, replay, projection, the same
quantisation statistics results/m90/glstats.npz), so that the sequence identity networks train on them next to the
competition's games. That session's modules are imported read-only; nothing of theirs is written.
Parts <out>/ply_x_s<shard>of<n>_p<k>.npz: key, LEN, SUBJ, POS, MOV, GLQ (the first 256 plies). Resumable per part.
Run from the AI_CUP_2026-1 checkout:
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I <this> --select <select.csv> --out <dir> --workers 10
"""
import argparse  # 命令列參數
import csv  # 讀取 select.csv
import multiprocessing as mp  # 平行重播棋譜
import os  # 寫完後原子性換名
import sys  # 匯入路徑
import time  # 計時
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 陣列運算與寫檔
import torch  # 在 GPU 上跑主幹網路

ROOT = Path.cwd()  # 須在 AI_CUP_2026-1 的根目錄執行
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/m_series")]  # 借用棋力 session 的模組（唯讀）
import trunk_extract as te  # noqa: E402  棋力 session 的重播、投影與向量計算
from aicup.humansl import load_humansl  # noqa: E402  載入人類棋力網路（Human SL）
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402  中性的中繼資料列與主幹特徵

STATS, T_MAX, Q = ROOT / "results/m90/glstats.npz", 256, 20.0  # 與 M90 相同的量化統計、手數上限與量化倍率


def safe_replay(task):  # 重播一局，失敗時不中斷整批
    try:  # 正常重播
        return te.replay(task)  # 回傳每手的輸入特徵、落點、執色與主體標記
    except Exception as e:  # 重播無法照著走下去的棋譜：略過不收
        return (task[0], None, repr(e)[:80])  # 標記為失敗並留下原因


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--select", required=True)  # ext_fox_select.py 產生的 select.csv
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--shard", type=int, default=0)  # 這個行程的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--workers", type=int, default=10)  # 重播的平行行程數
    ap.add_argument("--part", type=int, default=200)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=256)  # 主幹網路的批次大小
    ap.add_argument("--limit", type=int, default=0, help="smoke run: only this many games")  # 煙霧測試的局數上限
    ap.add_argument("--start", type=int, default=0, help="rows [start, stop) of the selection (applied before the shards), for a selection extracted in stages")  # 分階段抽取時的起始列
    ap.add_argument("--stop", type=int, default=0)  # 分階段抽取時的結束列
    ap.add_argument("--max-plies", type=int, default=T_MAX, help="store (and run the trunk on) only the first this many plies of a game (METHOD_NOTES 2.70; default 256)")  # 每局只存前幾手
    ap.add_argument("--compress", action="store_true", help="write the parts deflate-compressed (np.load reads them the same; METHOD_NOTES 2.164)")  # 以壓縮格式寫分段檔
    ap.add_argument("--light", action="store_true", help="no trunk: the parts hold the plies' points and movers only, GLQ is zero (the mean after de-quantisation). For games that only "  # 不跑主幹的輕量模式
                                                         "networks reading the supervised features alone (--sup-only) will train on (METHOD_NOTES 2.126); no GPU is used")  # 給只讀有監督特徵的網路用
    ap.add_argument("--account-games", type=int, nargs=2, default=None, metavar=("FROM", "TO"),  # 只取每個帳號的第 FROM 到 TO-1 局
                    help="only the games number FROM .. TO-1 of every account, in the selection's order (0 4: the first four of each; a later run with 4 8 adds the rest)")  # 例如 0 4 取每個帳號前四局，之後再以 4 8 補上其餘
    a = ap.parse_args()  # 解析參數
    t0 = time.time()  # 計時起點
    csv.field_size_limit(10 ** 9)  # 棋譜欄位很長，放寬 csv 的欄位上限
    assert 1 <= a.max_plies <= T_MAX  # 手數上限必須在 1 到 256 之間
    with open(a.select, newline="") as fh:  # 讀取選出的外部棋局
        if a.account_games:  # 只取每個帳號的一段棋局
            seen, tasks = {}, []  # 各帳號已讀到的局數與工作清單
            for r in csv.DictReader(fh):  # 依 select.csv 的順序
                i = seen.get(r["account"], 0)  # 這一局是該帳號的第幾局
                seen[r["account"]] = i + 1  # 累計
                if a.account_games[0] <= i < a.account_games[1]:  # 在指定範圍內
                    tasks.append((r["game_key"], r["sgf_content"], r["color"]))  # 加入工作：棋局鍵、棋譜、主體執色
        else:  # 取全部棋局
            tasks = [(r["game_key"], r["sgf_content"], r["color"]) for r in csv.DictReader(fh)]  # 每局一個工作
    tasks = tasks[a.start:(a.stop or len(tasks))][a.shard::a.nshards]  # 先取 [start, stop) 再切出本分片
    tasks = tasks[:a.limit] if a.limit else tasks  # 煙霧測試只取前幾局
    st = np.load(STATS)  # M90 的量化統計
    mu, sd = st["mu"], st["sd"]  # 每一維的平均與標準差
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    print(f"{len(tasks)} games in this shard ({time.time() - t0:.0f} s)", flush=True)  # 回報本分片的局數
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]  # 網路設定（工作行程建立輸入特徵用）
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重播用的工作行程池
    try:  # 結束時一定關閉行程池
        model = None  # 輕量模式不載入網路
        if not a.light:  # 需要主幹向量時
            model, _ = load_humansl(str(te.CKPT), "cuda")  # 載入人類棋力網路到 GPU
            model.eval()  # 推論模式
        nrows, proj, plies, skipped, t1 = neutral_rows(), None, 0, 0, time.time()  # 中性中繼資料列、投影矩陣、統計與計時
        for k, start in enumerate(range(0, len(tasks), a.part)):  # 逐分段處理
            path = out / f"ply_x_s{a.shard}of{a.nshards}_p{k:05d}.npz"  # 這個分段檔的路徑
            if path.exists():  # 已完成的分段
                continue  # 續跑時跳過
            res = list(pool.map(safe_replay, tasks[start:start + a.part], chunksize=2))  # 平行重播這一段的棋局
            done = [d for d in res if d[1] is not None and len(d[3]) > 0]  # 重播成功且至少有一手的棋局
            skipped += len(res) - len(done)  # 累計略過的局數
            if not done:  # 這一段全部失敗
                continue  # 不寫檔
            if proj is None and not a.light:  # 第一次需要投影時
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 只為了得知主幹的通道數與裝置
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)  # 對一個局面跑主幹
                proj = te.projections(trunk.shape[1], trunk.device)  # 與棋力 session 相同種子的隨機投影
            lens = np.array([len(d[3]) for d in done])  # 每局的手數
            pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])  # 所有手的落點與執色
            keep = np.concatenate([np.arange(n) for n in lens]) < a.max_plies  # 只保留每局前 max_plies 手
            if a.light:  # 只存落點與執色；去量化後 0 就是每一維的平均
                q = np.zeros((int(keep.sum()), len(mu)), np.int8)  # 主幹向量全部填 0
            else:  # 一般模式
                bins, globs = np.concatenate([d[1] for d in done]), np.concatenate([d[2] for d in done])  # 所有手的盤面與全域輸入特徵
                if a.max_plies < T_MAX:  # 主幹只跑會存下的手（每個局面是各自獨立計算的）
                    bins, globs, pos, mover = bins[keep], globs[keep], pos[keep], mover[keep]  # 先截掉不存的手，省下計算
                    keep = np.ones(len(pos), bool)  # 剩下的手全部保留
                g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch)  # 全盤池化與落點的 64 維向量
                q = np.clip(np.rint((np.hstack([g, l])[keep].astype(np.float32) - mu) / sd * Q), -127, 127).astype(np.int8)  # 以 M90 的統計量化成 int8
            subj = np.array([int(d[4][d[5]][0]) if d[5].any() else 0 for d in done], np.int8)  # 每局主體的執色（0 黑／1 白）
            tmp = path.with_name(path.name + ".tmp.npz")  # 先寫暫存檔
            (np.savez_compressed if (a.light or a.compress) else np.savez)(tmp, key=np.array([d[0] for d in done]), LEN=np.minimum(lens, a.max_plies).astype(np.int16), SUBJ=subj, POS=pos[keep].astype(np.int16),  # 寫出與 M90 欄位相同的分段檔
                                                           MOV=mover[keep].astype(np.int8), GLQ=q)  # 執色與量化後的主幹向量
            os.replace(tmp, path)  # 完整寫完才換成正式檔名
            plies += int(lens.sum())  # 累計手數
            if k % 20 == 0:  # 每 20 個分段回報一次
                print(f"part {k + 1}/{(len(tasks) + a.part - 1) // a.part}: {plies} plies, {plies / (time.time() - t1):.0f} plies/s, skipped {skipped}", flush=True)  # 進度、速度與略過數
        print(f"done: {len(tasks)} games, {plies} plies, skipped {skipped}, {(time.time() - t0) / 60:.1f} min", flush=True)  # 總結
    finally:  # 不論成功或失敗
        pool.close()  # 關閉行程池
        pool.join()  # 等待工作行程結束


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
