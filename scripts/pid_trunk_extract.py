#!/usr/bin/env python
"""Line B4 (trunk variant): per-game trunk-style features of the pristine humanv0 trunk (METHOD_NOTES 2.21).

Every game is replayed exactly as AI_CUP_2026-1's scripts/trunk_extract.py replays it (same Features, rules, handicap fix,
random projections to 64 + 64 dims with the same seed); the trunk is run on the TARGET player's positions only (Black in
this task), and each game is stored as one 256-dimensional row: mean and standard deviation of the board-pooled vector
and of the played-point vector over the target's plies (passes have no played point). Rows are written in parts
(<out>/<tag>_s<global shard>_p<k>.npz: keys, feats float32 [G, 256], n_target) and the run resumes per part.

Inputs: --official (both official pools: candidate games keyed c:<row>:<game_id> as predict_pid.py keys them, question
games q:<question_id>:<i>) or --keys FILE (training game keys, one per line or a CSV with a game_key column). Sharding:
shards.sh appends --shard i --nshards N per GPU; --node k --nodes M spreads one key list over M such jobs.

  sbatch -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J trunkx scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 \\
      python -I ../BK_AICUP_2026-2/scripts/pid_trunk_extract.py --official --out ../BK_AICUP_2026-2/data/trunk/official --node 0 --nodes 2 --workers 10
"""
import argparse  # 解析命令列參數
import multiprocessing as mp  # 用多個行程平行重播棋局
import os  # os.replace 原子換檔
import sys  # 設定 sys.path 與結束碼
import time  # 計算耗時與速度
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列運算與 .npz 存檔
import pandas as pd  # 讀取 CSV
import torch  # 在 GPU 上執行主幹

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
if str(ROOT / "src") not in sys.path:  # 尚未加入本儲存庫的 src 時
    sys.path.insert(0, str(ROOT / "src"))  # 讓 aicup_pid 可以匯入
import aicup_pid  # noqa: E402  匯入時會把 AI_CUP_2026-1/src 放上 sys.path
from aicup_pid import pools  # noqa: E402  讀取官方候選檔與題目檔
from aicup_pid.symmetry import game_sym, transform_sgf  # noqa: E402  --sym-seed 的棋盤旋轉／鏡像
from aicup_pid.paths import AICUP1  # noqa: E402,F401  匯入後 AI_CUP_2026-1 的 scripts 資料夾已在 sys.path 上
import trunk_extract as te  # noqa: E402  AI_CUP_2026-1 的重播、投影與向量計算
from aicup.humansl import load_humansl  # noqa: E402  載入人類棋力網路（Human SL）
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402  中性詮釋資料列與主幹輸出
from aicup.scoring import load_games  # noqa: E402  依鍵讀取訓練棋局
from aicup.submission import check_game  # noqa: E402  判斷棋局能否使用（19 路、能合法重播、目標方至少一手）

DIM = 4 * te.PROJ_DIM  # 每局一列的維度：4 × 64 = 256
T = ROOT / "data" / "tests"  # 官方 CSV 的預設資料夾


def official_tasks(tests_dir=T):  # 原題池兩池的抽取任務（--official）
    """{key: (sgf, colour)} of every usable game of both official pools in tests_dir (default data/tests)."""
    out = {}  # 鍵 → (棋譜, 目標方顏色)
    for c_name, q_name in (("player_identification_candidates_public.csv", "player_identification_test_public.csv"),  # 先 Public 的候選檔與題目檔
                           ("player_identification_candidates_private.csv", "player_identification_test_private.csv")):  # 再 Private
        cands = pools.read_candidates(Path(tests_dir) / c_name)  # 讀候選人棋局（原題池的候選棋局一律執黑，否則報錯）
        for i, r in enumerate(cands.itertuples()):  # 逐列處理候選棋局
            out[f"c:{i}:{r.game_id}"] = (r.sgf, "B")  # 鍵的寫法同 predict_pid.py：以列號定位（game_id 不必唯一）；目標方為黑
        for q in pools.read_questions(Path(tests_dir) / q_name):  # 逐題讀取題目
            for i, (sgf, colour) in enumerate(q["games"]):  # 題目中的每一局
                if check_game(sgf, colour or "B")[0]:  # 只收可用的棋局；未標顏色時以黑方檢查
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")  # 題目棋局的鍵；未標顏色時視為黑方
    return out  # 回傳全部任務


def official2_tasks(tests_dir):  # v2 新增池的抽取任務（--official2）
    """{key: (sgf, colour)} of the 11-08 v2 release's ADDED pools (candidates2 / test2, public and private): both colours, the
    colour column is the player's side. Keys as official_tasks: c:<row in its file>:<game_id>, q:<question_id>:<i>."""
    out = {}  # 鍵 → (棋譜, 玩家顏色)
    for tag in ("public", "private"):  # 兩個新增池分開讀
        cands = pd.read_csv(Path(tests_dir) / f"player_identification_candidates2_{tag}.csv", dtype=str, keep_default_na=False)  # 新增池候選檔（含 color 欄）
        for i, r in enumerate(cands.itertuples()):  # 逐列處理候選棋局
            if check_game(r.sgf_content, r.color)[0]:  # 只收可用、且該玩家顏色至少有一手的棋局
                out[f"c:{i}:{r.game_id}"] = (r.sgf_content, r.color)  # 以檔內列號為鍵
        for q in pools.read_questions(Path(tests_dir) / f"player_identification_test2_{tag}.csv"):  # 新增池的題目檔
            for i, (sgf, colour) in enumerate(q["games"]):  # 題目中的每一局
                if check_game(sgf, colour or "B")[0]:  # 只收可用的棋局；未標顏色時以黑方檢查
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")  # 題目棋局的鍵與玩家顏色
    return out  # 回傳全部任務


def read_keys(path):  # 讀取 --keys 指定的訓練棋局鍵
    p = Path(path)  # 轉成路徑物件
    if p.suffix == ".csv":  # CSV 格式
        return pd.read_csv(p, dtype=str)["game_key"].tolist()  # 取 game_key 欄
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]  # 文字檔：每行一個鍵，略過空行


def aggregate(g, l, pos, lens):  # 逐手向量彙整成每局一列
    """[G, 256]: per game mean / std of the pooled and of the played-point vectors over its target plies."""
    out = np.zeros((len(lens), DIM), np.float32)  # 沒有目標手的局維持全零
    start = 0  # 這局在逐手陣列中的起點
    for i, n in enumerate(lens):  # 逐局處理
        if n:  # 這局有目標方的手
            gi = g[start:start + n].astype(np.float32)  # 這局各手的全盤池化向量
            li = l[start:start + n][pos[start:start + n] < te.PASS_POS].astype(np.float32)  # 落點向量，排除 pass
            out[i, 0:64], out[i, 64:128] = gi.mean(axis=0), gi.std(axis=0)  # 前 128 維：全盤向量的平均與標準差
            if len(li):  # 至少有一手不是 pass
                out[i, 128:192], out[i, 192:256] = li.mean(axis=0), li.std(axis=0)  # 後 128 維：落點向量的平均與標準差
        start += n  # 移到下一局的起點
    return out  # 每局一列的矩陣


def replay_target(task):  # --fast 用的重播：只算目標方的輸入特徵
    """trunk_extract.replay for the TARGET side's plies only: the game is played through as there, but the input features of the
    other side's positions are never computed (they are most of the replay's cost; the rank session's replay_side, AI_CUP_2026-1
    experiments/nano4/m118_fullrows_extract.py, checked against the full replay). Returns (key, bins, globs, pos, mover, t) for the
    target's plies, t their ply numbers; bins and globs are None when the target has no ply."""
    idx, sgf, color = task  # 鍵、棋譜、目標方顏色
    stones, moves = te.setup_and_moves(sgf)  # 解析讓子石與著手序列
    gs = te.start_state(stones, te.RULES)  # 以相同規則與讓子修正建立起始局面
    bins, globs, pos, mover, ts = [], [], [], [], []  # 收集目標方各手的資料
    for t, (c, x, y) in enumerate(moves):  # 依序重播每一手
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE  # 這手的行棋方
        loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)  # 落點位置（pass 另外處理）
        if c == color:  # 只有目標方的手才計算特徵
            b, g = gs.get_input_features(te._FEATURES)  # 落子前局面的盤面與全域特徵
            bins.append(b[0].astype(np.uint8))  # 盤面特徵只有 0／1，以 uint8 省記憶體
            globs.append(g[0].astype(np.float32))  # 全域特徵
            pos.append(te.PASS_POS if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board))  # 落點的張量位置；pass 為 361
            mover.append(0 if c == "B" else 1)  # 行棋方：0 黑、1 白
            ts.append(t)  # 記下這一手的手數
        gs.play(pla, loc)  # 不論哪一方都要落子，局面才會正確
    if not pos:  # 目標方一手都沒有
        return idx, None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)  # 回傳空陣列
    return idx, np.stack(bins), np.stack(globs), np.asarray(pos, np.int16), np.asarray(mover, np.int8), np.asarray(ts, np.int64)  # 疊成陣列回傳


def full_vectors(model, bins, globs, pos, mover, rows, batch):  # --full：未投影的主幹向量
    """[n, C] float32 board-pooled and played-point trunk vectors WITHOUT the random projection (C = 384; a pass has a zero
    played-point vector), computed as trunk_extract.vectors computes them before it projects (METHOD_NOTES 2.69, probe)."""
    n = len(pos)  # 局面數
    out_g = out_l = None  # 通道數要等第一批算完才知道
    with torch.no_grad():  # 只做推論
        for i in range(0, n, batch):  # 分批送進 GPU
            j = min(n, i + batch)  # 這一批的結尾
            b, g = torch.as_tensor(bins[i:j]).cuda(), torch.as_tensor(globs[i:j]).cuda()  # 盤面與全域特徵搬到 GPU
            c = torch.as_tensor(mover[i:j]).cuda().long()  # 行棋方，用來選詮釋資料列
            with torch.autocast("cuda", dtype=torch.bfloat16):  # 以 bf16 加速主幹
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)  # 主幹最終層 [B, C, 19, 19]
            trunk = trunk.float()  # 轉回 float32 再取向量
            p = torch.as_tensor(pos[i:j]).cuda().long()  # 落點的張量位置
            on = p < te.PASS_POS  # 這一手有落點（不是 pass）
            local = torch.zeros(j - i, trunk.shape[1], device=trunk.device)  # pass 的落點向量維持為零
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]  # 取落點那一格的主幹向量
            if out_g is None:  # 第一批：依通道數配置輸出
                out_g, out_l = np.empty((n, trunk.shape[1]), np.float32), np.empty((n, trunk.shape[1]), np.float32)  # 全盤與落點兩個輸出陣列
            out_g[i:j], out_l[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy()  # 全盤平均池化與落點向量
    return out_g, out_l  # 兩個 [n, C] 陣列


def policy_vectors(model, bins, globs, pos, mover, rows, batch):  # 探針 P：主幹向量加上策略的預期
    """full_vectors plus what the network itself EXPECTED at every position (METHOD_NOTES 2.102, probe P). One forward pass
    gives the trunk and the policy (channel 0, the side to move, the neutral metadata row); with the policy over the legal
    board points (on board, empty, not the ko point; pass left out and the rest renormalised):
      g [n, C]  board-pooled trunk vector               x [n, C]  trunk vector at the played point (zero for a pass)
      e [n, C]  policy-weighted mean of the trunk vectors of the board points = the played-point vector the network expects
                (zero for a pass, so that x - e is zero there)
      s [n, 5]  the played move's surprisal (-log p, pass included in that distribution), its rank in the policy (0 = the top
                move), the policy's entropy, its top probability, the Chebyshev distance between the played point and the
                policy's top board point (0 for a pass)
    Also returns the number of played moves my legality mask called illegal (their surprisal is clipped; should be ~0)."""
    from aicup.l1b import _GraphExtraOutputs  # 取得主幹中間輸出的工具
    n = len(pos)  # 局面數
    C = None  # 主幹通道數，第一批時決定
    S = np.zeros((n, 5), np.float32)  # 每手 5 個策略統計量
    bad = 0  # 被遮罩判為不合法的實際著手數
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device="cuda")  # 中性詮釋資料列放到 GPU
    meta = model.get_has_metadata_encoder()  # 網路是否有詮釋資料編碼器
    with torch.no_grad():  # 只做推論
        for i in range(0, n, batch):  # 分批計算
            j = min(n, i + batch)  # 這一批的結尾
            b, g = torch.as_tensor(bins[i:j]).cuda().float(), torch.as_tensor(globs[i:j]).cuda().float()  # 輸入特徵搬到 GPU
            c = torch.as_tensor(mover[i:j]).cuda().long()  # 行棋方
            extra = _GraphExtraOutputs(["trunkfinal"])  # 要求回傳主幹最終層
            with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 前向傳遞
                outputs = model(b, g, input_meta=rows_t[c] if meta else None, extra_outputs=extra)  # 一次同時得到主幹與策略
                logits = model.postprocess_output(outputs)[0][0][:, 0, :]  # 策略通道 0 的 logits [B, 362]
            trunk, logits = extra.returned["trunkfinal"].float(), logits.float()  # 轉回 float32
            if C is None:  # 第一批：依通道數配置輸出
                C = trunk.shape[1]  # 主幹通道數
                G, X, E = np.empty((n, C), np.float32), np.empty((n, C), np.float32), np.empty((n, C), np.float32)  # g、x、e 三個輸出
            legal_b = ((b[:, 0] - b[:, 1] - b[:, 2] - b[:, 6]) > 0.5).flatten(1)  # [B, 361] 在盤上、空點且不是劫點
            legal = torch.cat([legal_b, torch.ones(j - i, 1, dtype=torch.bool, device=b.device)], dim=1)  # 再加上一律合法的 pass
            lp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)  # 合法著手上的對數機率
            p = torch.as_tensor(pos[i:j].astype(np.int64)).cuda()  # 實際落點
            on = p < te.PASS_POS  # 不是 pass 的手
            lpp = lp.gather(1, p[:, None])[:, 0]  # 實際著手的對數機率
            miss = ~torch.isfinite(lpp)  # 被遮罩判為不合法的實際著手
            bad += int(miss.sum())  # 累計不合法的數量
            lpp = torch.where(miss, torch.full_like(lpp, -20.0), lpp).clamp(min=-20.0)  # 截斷在 -20，避免無限大
            rank = (lp > lpp[:, None]).sum(1).float()  # 實際著手在策略中的名次
            pr = lp.exp()  # 機率分布
            ent = -(pr * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum(1)  # 策略的熵（不合法點不計）
            pb = torch.softmax(logits[:, :361].masked_fill(~legal_b, float("-inf")), dim=1)  # 只在合法盤上點重新正規化
            pb = torch.where(torch.isfinite(pb), pb, torch.zeros_like(pb))  # 滿盤：沒有合法點時改為 0
            e = torch.einsum("bp,bcp->bc", pb, trunk.flatten(2))  # 以策略加權的主幹向量 = 預期的落點向量
            local = torch.zeros(j - i, C, device=trunk.device)  # pass 的落點向量為零
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]  # 實際落點的主幹向量
            e = e * on[:, None].float()  # pass 的預期向量也設為零
            top = pb.argmax(1)  # 策略第一個盤上選點
            dist = torch.maximum((p // 19 - top // 19).abs(), (p % 19 - top % 19).abs()).float() * on.float()  # 與第一選點的 Chebyshev 距離
            G[i:j], X[i:j], E[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy(), e.cpu().numpy()  # 存 g、x、e
            S[i:j] = torch.stack([-lpp, rank, ent, pr.max(1).values, dist], dim=1).cpu().numpy()  # 存 5 個統計量
    return G, X, E, S, bad  # 四個陣列與不合法數


def aggregate_full(g, l, pos, lens):  # --full 的每局彙整
    """[G, 4C]: per game mean / std of the unprojected pooled vectors and of the unprojected played-point vectors (as aggregate)."""
    C = g.shape[1]  # 主幹通道數（384）
    out = np.zeros((len(lens), 4 * C), np.float32)  # 每局 4C 維
    start = 0  # 這局在逐手陣列中的起點
    for i, n in enumerate(lens):  # 逐局處理
        if n:  # 這局有目標方的手
            gi = g[start:start + n]  # 全盤池化向量
            li = l[start:start + n][pos[start:start + n] < te.PASS_POS]  # 落點向量，排除 pass
            out[i, 0:C], out[i, C:2 * C] = gi.mean(axis=0), gi.std(axis=0)  # 全盤向量的平均與標準差
            if len(li):  # 至少有一手不是 pass
                out[i, 2 * C:3 * C], out[i, 3 * C:4 * C] = li.mean(axis=0), li.std(axis=0)  # 落點向量的平均與標準差
        start += n  # 下一局的起點
    return out  # 每局一列的矩陣


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    src = ap.add_mutually_exclusive_group(required=True)  # 三種輸入來源擇一
    src.add_argument("--official", action="store_true")  # 原題池兩池
    src.add_argument("--official2", action="store_true", help="the v2 release's added pools (candidates2 / test2), both colours")  # v2 新增池
    ap.add_argument("--tests-dir", default=str(T), help="folder of the official CSVs for --official (a re-released set goes in its own folder)")  # 官方 CSV 的資料夾
    src.add_argument("--keys", help="training game keys (text file or CSV with game_key)")  # 訓練棋局的鍵清單
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--tag", default="part")  # 輸出檔名前綴
    ap.add_argument("--workers", type=int, default=10)  # 重播棋局的行程數
    ap.add_argument("--shard", type=int, default=0)  # 本節點內的分片編號（shards.sh 給）
    ap.add_argument("--nshards", type=int, default=1)  # 每個節點的分片數
    ap.add_argument("--node", type=int, default=0)  # 本工作是第幾個節點
    ap.add_argument("--nodes", type=int, default=1)  # 分攤同一份鍵清單的工作數
    ap.add_argument("--part-size", type=int, default=200)  # 每個分段檔的局數
    ap.add_argument("--batch", type=int, default=512)  # 主幹推論的批次大小
    ap.add_argument("--max-games", type=int, default=0, help="smoke tests only")  # 只用於冒煙測試
    ap.add_argument("--fast", action="store_true", help="replay with the input features of the target side only (replay_target): the same rows, about twice as fast")  # 只算目標方特徵的快速重播
    ap.add_argument("--max-parts", type=int, default=0, help="stop after this many parts (a check run)")  # 檢查用：寫完幾個分段檔就停
    ap.add_argument("--full", action="store_true", help="also store the per-game mean / std of the UNPROJECTED trunk vectors (array `full`, [G, 4 x 384]); the 256-dim rows are then computed from the same vectors")  # 另存未投影向量的統計
    ap.add_argument("--sym-seed", type=int, default=None, help="rotate / mirror every game by aicup_pid.symmetry.game_sym(key, sgf, seed) before the replay (simulated re-release; HD >= 2 games stay as they are)")  # 模擬重發：每局依種子與鍵選一種旋轉／鏡像（HD ≥ 2 的局不轉）
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    shard, nshards = a.node * a.nshards + a.shard, a.nodes * a.nshards  # 換算成全域分片編號與總數
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 不存在就建立
    if a.official or a.official2:  # 官方棋局：棋譜直接從 CSV 取
        tasks_all = official2_tasks(a.tests_dir) if a.official2 else official_tasks(a.tests_dir)  # 依選項取新增池或原題池
        keys = sorted(tasks_all)  # 排序，讓各分片的切法一致
    else:  # 訓練棋局：只有鍵，棋譜之後再讀
        tasks_all, keys = None, sorted(set(read_keys(a.keys)))  # 去重並排序
    if a.max_games:  # 冒煙測試
        keys = keys[: a.max_games]  # 只取前幾局
    mine = keys[shard::nshards]  # 本分片負責的局
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]  # 切成固定局數的分段
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz").exists()]  # 已寫出的分段檔略過（續跑）
    print(f"shard {shard}/{nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do", flush=True)  # 回報工作量
    if not todo:  # 全部做完了
        return 0  # 直接結束
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]  # 讀網路設定給特徵抽取用
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))  # 重播用的行程池，各自建立 Features
    try:  # 確保最後關閉行程池
        model, _ = load_humansl(str(te.CKPT), "cuda")  # 載入人類棋力網路（Human SL）
        model.eval()  # 推論模式
        rows = neutral_rows()  # 中性詮釋資料列（黑、白各一列）
        proj_g = proj_l = None  # 隨機投影矩陣，處理第一個分段時建立
        games_done = plies_done = 0  # 進度統計
        for k, part in todo:  # 逐個分段處理
            sgf = {q: tasks_all[q] for q in part} if tasks_all is not None else load_games(part, aicup_pid.TRAIN_DIR)  # 取得這批棋譜與顏色
            if a.sym_seed is not None:  # 模擬重發的旋轉／鏡像
                sgf = {q: (transform_sgf(v[0], game_sym(q, v[0], a.sym_seed)), *v[1:]) for q, v in sgf.items()}  # 每局依（種子, 鍵）的雜湊選 8 種對稱之一；HD ≥ 2 的局不轉
            if a.fast:  # 只回傳目標方的手，並附上手數
                done = list(pool.imap(replay_target, [(q, *sgf[q]) for q in part], chunksize=1))  # 平行重播，保持原順序
                tsel = [d[5] for d in done]  # 各局目標方各手的手數
                some = [d for d in done if len(d[3])]  # 有目標手的局（其餘的 bins 為 None）
                bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])  # 串接所有目標手的輸入特徵
                pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])  # 串接落點與行棋方
                lens = [len(d[3]) for d in done]  # 每局目標方的手數
            else:  # 完整重播：雙方的手都算，再挑出目標方
                done = list(pool.imap(te.replay, [(q, *sgf[q]) for q in part], chunksize=1))  # 與 trunk_extract 相同的重播
                sel = [d[5] for d in done]  # 每局目標方的手（遮罩）
                tsel = [np.flatnonzero(m) for m in sel]  # 目標手的手數
                bins = np.concatenate([d[1][m] for d, m in zip(done, sel)])  # 只留目標手的盤面特徵
                globs = np.concatenate([d[2][m] for d, m in zip(done, sel)])  # 只留目標手的全域特徵
                pos = np.concatenate([d[3][m] for d, m in zip(done, sel)])  # 目標手的落點
                mover = np.concatenate([d[4][m] for d, m in zip(done, sel)])  # 目標手的行棋方
                lens = [int(m.sum()) for m in sel]  # 每局目標方的手數
            if proj_g is None:  # 第一個分段：建立投影
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 只為取得主幹的通道數
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(bins[:1]).cuda(), torch.as_tensor(globs[:1]).cuda(),  # 用一個局面跑主幹
                                                    torch.as_tensor(mover[:1]).cuda().long(), rows)  # 行棋方與詮釋資料列
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)  # 固定種子的隨機投影，與 AI_CUP_2026-1 的 trunk_extract.py 相同
            full = None  # 預設不存未投影的統計
            if a.full:  # 主幹只跑一次：取得未投影向量，再由它算出一般的投影向量
                fg, fl = full_vectors(model, bins, globs, pos, mover, rows, a.batch)  # 未投影的全盤與落點向量
                full = aggregate_full(fg, fl, pos, lens)  # 每局 4 × 384 維的統計
                g = (fg @ proj_g.float().cpu().numpy()).astype(np.float16)  # 投影成 64 維全盤向量
                l = (fl @ proj_l.float().cpu().numpy()).astype(np.float16)  # 投影成 64 維落點向量
            else:  # 一般情況
                g, l = te.vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, a.batch)  # 直接算投影後的向量
            feats = aggregate(g, l, pos, lens)  # 每局 256 維的列
            if not np.isfinite(feats).all():  # 出現 NaN 或無限大
                raise ValueError(f"part {k}: non-finite features")  # 停下，不寫壞資料
            # 目標方各手的分階段平均（t < 60、60 <= t < 150、t >= 150；各 128 維，該階段沒有手時為零）：定義與
            # AI_CUP_2026-1 scripts/m_series/m84_rows_extract.py 的 side_rows 相同（pass 的落點向量維持為零）
            phase = np.zeros((len(lens), 384), np.float16)  # 每局三個階段 × 128 維
            X = np.hstack([g.astype(np.float32), l.astype(np.float32)])  # 每手 128 維（全盤 64＋落點 64）
            start = 0  # 這局在逐手陣列中的起點
            for i, (t_, n_) in enumerate(zip(tsel, lens)):  # 逐局處理
                for j, pm in enumerate((t_ < 60, (t_ >= 60) & (t_ < 150), t_ >= 150)):  # 三個階段的遮罩
                    if pm.any():  # 這個階段有目標手
                        phase[i, 128 * j:128 * (j + 1)] = X[start:start + n_][pm].mean(axis=0)  # 該階段的平均
                start += n_  # 下一局的起點
            path = out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz"  # 這個分段的輸出檔
            tmp = path.with_name(path.name + ".tmp.npz")  # 先寫暫存檔
            extra = {}  # 額外存的陣列
            if full is not None:  # 未投影：整局平均／標準差 [4 x 384] 與三個階段的平均 [3 x 768]；每局 7.7 KB
                XF = np.hstack([fg, fl])  # 每手 768 維的未投影向量
                fullp = np.zeros((len(lens), 3 * XF.shape[1]), np.float32)  # 三個階段的平均
                start = 0  # 這局的起點
                for i, (t_, n_) in enumerate(zip(tsel, lens)):  # 逐局處理
                    for j, pm in enumerate((t_ < 60, (t_ >= 60) & (t_ < 150), t_ >= 150)):  # 三個階段
                        if pm.any():  # 這個階段有目標手
                            fullp[i, XF.shape[1] * j:XF.shape[1] * (j + 1)] = XF[start:start + n_][pm].mean(axis=0)  # 該階段的平均
                    start += n_  # 下一局的起點
                extra = {"full": full.astype(np.float16), "fullp": fullp.astype(np.float16)}  # 以 fp16 存，省空間
            np.savez(tmp, keys=np.array([d[0] for d in done]), feats=feats, n_target=np.array(lens, np.int16), phase=phase, **extra)  # 寫出這個分段檔
            os.replace(tmp, path)  # 原子換名：分段檔只會完整出現，續跑不會讀到寫一半的檔
            games_done += len(part)  # 累計局數
            plies_done += sum(len(d[3]) for d in done)  # 累計重播的手數
            if a.max_parts and games_done >= a.max_parts * a.part_size:  # 檢查用的上限
                break  # 提早結束
            el = time.time() - t0  # 已用時間
            print(f"part {k + 1}/{len(parts)}: {games_done} games, {games_done / el:.1f} games/s, {plies_done / el:.0f} plies/s replayed", flush=True)  # 回報速度
    finally:  # 不論成功或失敗
        pool.close()  # 不再接受新任務
        pool.join()  # 等行程結束
    print(f"shard {shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)  # 回報總耗時
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
