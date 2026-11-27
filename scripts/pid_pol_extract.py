#!/usr/bin/env python
"""Probe P (METHOD_NOTES 2.102): features of a move RELATIVE TO WHAT THE NETWORK EXPECTED at that position.

Every feature so far describes the position (board-pooled trunk vector) and the point that was played (trunk vector at that
point). Most of the variation of the played-point vector comes from the position, not from the player: in a fight everybody
plays contact moves. The same forward pass also gives the network's policy, so the played-point vector that an average player
would produce there is known: e = sum over points of policy(point) x trunk(point). The difference x - e is the player's choice
with the position's contribution removed (a background model, as in authorship attribution).

One pass over the games (scripts/pid_trunk_extract.py policy_vectors; the subject's plies only) writes
  <out>/rows/pol_s<shard>_p<k>.npz   per game: keys, n_target, full [4C] (mean / std of pooled and played-point vectors, as
                                     pid_trunk_extract.py --full), pole [2C] (mean / std of e over the non-pass plies),
                                     pold [C] (std of x - e), pols [36] (surprisal / rank / entropy statistics, whole game
                                     and three phases)
  <out>/ply_r/sup_s<shard>_p<k>.npz  per ply, the layout of scripts/pid_sup_ply_extract.py (keys, n, t, z float16):
                                     z = the EXISTING supervised projection (--proj: mu, sd, W over [pooled, played]) applied
                                     to [pooled, played - expected], 64 dims
  <out>/ply_fc/...                   z = [the existing supervised features (64), the expected vector through the played-point
                                     half of the same projection (64)]: a network can form the difference itself
  <out>/ply_s/...                    z = [surprisal, log(1 + rank), entropy, distance to the policy's top point]
Resumable per part (a part counts as done when its rows file exists; the per-ply files are written first).
  bash scripts/nano4/shards.sh 8 python -I <this> --keys keys.txt --proj proj.npz --out DIR --workers 10
"""
import argparse  # 命令列參數
import multiprocessing as mp  # 平行重播棋譜
import os  # 寫完後原子性換名
import sys  # 匯入路徑與結束碼
import time  # 計時
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 陣列運算與寫檔
import torch  # 在 GPU 上跑網路

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
for d in (ROOT / "src", ROOT / "scripts"):  # 套件與同目錄的腳本
    if str(d) not in sys.path:  # 尚未在匯入路徑上
        sys.path.insert(0, str(d))  # 加到最前面
import aicup_pid  # noqa: E402  匯入前須先設定 sys.path
import pid_trunk_extract as PT  # noqa: E402  只重播主體的手、策略向量與每局彙總
from aicup.humansl import load_humansl  # noqa: E402  載入人類棋力網路（Human SL）
from aicup.l1b import neutral_rows  # noqa: E402  中性的中繼資料列
from aicup.scoring import load_games  # noqa: E402  依棋局鍵讀取棋譜

te = PT.te  # 棋力 session 的 trunk_extract 模組
PHASES = 4  # 全局加上三個階段
N_STAT = 9  # 每個階段的統計量個數


def game_rows(X, E, S, pos, ts, lens):  # 由逐手的量算出每局的統計
    """pole [G, 2C], pold [G, C], pols [G, PHASES x N_STAT] (see the module docstring)."""
    C = X.shape[1]  # 主幹的通道數
    pole, pold, pols = np.zeros((len(lens), 2 * C), np.float32), np.zeros((len(lens), C), np.float32), np.zeros((len(lens), PHASES * N_STAT), np.float32)  # 每局的三組輸出
    start = 0  # 目前這局在逐手陣列中的起點
    for i, (n, t) in enumerate(zip(lens, ts)):  # 逐局處理
        if n:  # 這局有主體的手
            sl = slice(start, start + n)  # 這局的手
            on = pos[sl] < te.PASS_POS  # 落在盤上（非虛手）的手
            if on.any():  # 至少一手落在盤上
                e = E[sl][on]  # 網路預期的落點向量
                d = X[sl][on] - e  # 實際落點向量減去預期
                pole[i, :C], pole[i, C:], pold[i] = e.mean(axis=0), e.std(axis=0), d.std(axis=0)  # 預期向量的平均與標準差、差值的標準差
            sc = S[sl]  # 這局每手的策略統計
            for j, pm in enumerate((np.ones(n, bool), t < 60, (t >= 60) & (t < 150), t >= 150)):  # 全局與三個階段（依手的序號）
                if pm.any():  # 該階段有手
                    q = sc[pm]  # 該階段的手
                    r = q[:, 1]  # 實際落點在策略中的名次
                    pols[i, N_STAT * j:N_STAT * (j + 1)] = [q[:, 0].mean(), q[:, 0].std(), (r == 0).mean(), (r < 3).mean(), (r < 10).mean(), np.log1p(r).mean(),  # 意外度的平均與標準差、名次在前 1／3／10 的比例、log(1 + 名次) 的平均
                                                            q[:, 2].mean(), q[:, 3].mean(), q[:, 4].mean()]  # 熵、最高機率與到最高點距離的平均
        start += n  # 移到下一局
    return pole, pold, pols  # 回傳三組每局統計


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--keys", required=True, help="training game keys (text file)")  # 要處理的訓練棋局鍵
    ap.add_argument("--proj", default=None, help="the existing supervised projection (scripts/pid_sup_fit.py); not needed with --rows-only")  # 既有的有監督投影
    ap.add_argument("--ckpt", default=None, help="another KataGo checkpoint than humanv0 (probe K, METHOD_NOTES 2.119: a strong network as the feature extractor)")  # 換用其他 KataGo 檢查點（探針 K）
    ap.add_argument("--rows-only", action="store_true", help="write only the per-game rows (for fitting a projection), no per-ply stores")  # 只寫每局列
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--workers", type=int, default=10)  # 重播的平行行程數
    ap.add_argument("--shard", type=int, default=0)  # 這個行程的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--part-size", type=int, default=200)  # 每個分段的局數
    ap.add_argument("--batch", type=int, default=512)  # 網路的批次大小
    ap.add_argument("--max-parts", type=int, default=0)  # 這次執行最多處理的分段數（0＝不限）
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    if not a.rows_only:  # 要寫逐手特徵時才需要投影
        P = np.load(a.proj)  # 讀取有監督投影
        mu, sd, W = P["mu"].astype(np.float32), P["sd"].astype(np.float32), P["W"].astype(np.float32)  # 標準化參數與投影矩陣
        C = len(mu) // 2  # 每一半（池化、落點）的維數
    ckpt = a.ckpt or te.CKPT  # 預設用 humanv0
    out = Path(a.out)  # 輸出資料夾
    for d in (("rows",) if a.rows_only else ("rows", "ply_r", "ply_fc", "ply_s")):  # 需要的子資料夾
        (out / d).mkdir(parents=True, exist_ok=True)  # 建立子資料夾
    keys = sorted(set(PT.read_keys(a.keys)))  # 去重並排序的棋局鍵
    mine = keys[a.shard::a.nshards]  # 本分片負責的棋局
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]  # 切成分段
    todo = [(k, p) for k, p in enumerate(parts) if not (out / "rows" / f"pol_s{a.shard:03d}_p{k:04d}.npz").exists()]  # rows 檔還不存在的分段
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do", flush=True)  # 回報工作量
    if not todo:  # 全部做完了
        return 0  # 直接結束
    config = torch.load(str(ckpt), map_location="cpu", weights_only=a.ckpt is None)["config"]  # 外來的檢查點不只含張量（同棋力 session 的 experiments/ai_state_extract.py 的載入方式）
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重播用的工作行程池

    def save(path, **arrays):  # 以暫存檔加換名的方式寫 .npz
        tmp = path.with_name(path.name + ".tmp.npz")  # 暫存檔名
        np.savez(tmp, **arrays)  # 先寫暫存檔
        os.replace(tmp, path)  # 寫完才換成正式檔名

    try:  # 結束時一定關閉行程池
        model, _ = load_humansl(str(ckpt), "cuda")  # 載入網路到 GPU
        model.eval()  # 推論模式
        rows = neutral_rows()  # 中性的中繼資料列
        done_games = bad_all = 0  # 已完成局數與不合法落點的計數
        for k, part in todo:  # 逐分段處理
            sgf = load_games(part, aicup_pid.TRAIN_DIR)  # 讀取這一段的棋譜
            done = list(pool.imap(PT.replay_target, [(q, *sgf[q]) for q in part], chunksize=1))  # 重播棋局，只取主體那一方各手的輸入
            some = [d for d in done if len(d[3])]  # 主體至少下過一手的棋局（其他局沒有輸入特徵）
            bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])  # 所有手的盤面與全域輸入特徵
            pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])  # 所有手的落點與執色
            lens, ts = [len(d[3]) for d in done], [d[5] for d in done]  # 每局的手數與各手的序號
            G, X, E, S, bad = PT.policy_vectors(model, bins, globs, pos, mover, rows, a.batch)  # 池化、落點、預期向量與策略統計
            bad_all += bad  # 累計被判為不合法的實際落點
            if not a.rows_only:  # 要寫逐手特徵時
                z_old = ((np.hstack([G, X]) - mu) / sd) @ W  # 既有的有監督特徵
                c = (E / sd[C:]) @ W[C:]  # 預期向量只經過投影中落點的那一半（不減平均：z_old - c 就是 [pooled, played - expected] 的投影）
            for arr in ((S,) if a.rows_only else (z_old, c, S)):  # 檢查要寫出的特徵
                if not np.isfinite(arr).all():  # 有非有限值
                    raise ValueError(f"part {k}: non-finite features")  # 中止，避免寫出壞資料
            full = PT.aggregate_full(G, X, pos, lens)  # 每局 [4C] 的平均與標準差
            pole, pold, pols = game_rows(X, E, S, pos, ts, lens)  # 每局相對於預期的統計
            head = dict(keys=np.array([d[0] for d in done]), n=np.array(lens, np.int32), t=np.concatenate(ts).astype(np.int16))  # 逐手檔共用的欄位
            name = f"sup_s{a.shard:03d}_p{k:04d}.npz"  # 逐手檔名（同 pid_sup_ply_extract.py）
            if not a.rows_only:  # 先寫逐手檔
                save(out / "ply_r" / name, z=(z_old - c).astype(np.float16), **head)  # 扣掉預期後的有監督特徵
                save(out / "ply_fc" / name, z=np.hstack([z_old, c]).astype(np.float16), **head)  # 既有特徵與預期向量並列
                save(out / "ply_s" / name, z=np.stack([S[:, 0], np.log1p(S[:, 1]), S[:, 2], S[:, 4]], axis=1).astype(np.float16), **head)  # 四個策略純量：意外度、log(1 + 名次)、熵、到最高點的距離
            save(out / "rows" / f"pol_s{a.shard:03d}_p{k:04d}.npz", keys=head["keys"], n_target=np.array(lens, np.int16), full=full.astype(np.float16),  # 最後寫每局列：它存在就代表這一段完成
                 pole=pole.astype(np.float16), pold=pold.astype(np.float16), pols=pols)  # 相對於預期的每局統計
            done_games += len(part)  # 累計局數
            if (k % 10) == 0:  # 每 10 個分段回報一次
                print(f"part {k + 1}/{len(parts)}: {done_games} games, {done_games / (time.time() - t0):.1f} games/s, played moves outside my legality mask {bad_all}", flush=True)  # 進度、速度與不合法落點數
            if a.max_parts and done_games >= a.max_parts * a.part_size:  # 達到 --max-parts 的分段上限
                break  # 提前結束
    finally:  # 不論成功或失敗
        pool.close()  # 關閉行程池
        pool.join()  # 等待工作行程結束
    print(f"shard {a.shard} done in {(time.time() - t0) / 60:.1f} min; played moves outside my legality mask {bad_all}", flush=True)  # 總結耗時與不合法落點數
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
