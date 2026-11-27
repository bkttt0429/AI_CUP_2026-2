#!/usr/bin/env python
"""Simulated-pool CV of the player-identification ladder's CPU components (docs/records/ARCHITECTURE.md sections 2-3).

Each --pools entry names the folds of one simulated pool (aicup_pid.pools.make_pool: up to 400 players with >= 105 Black
games, 100 candidate games each, the rest cut into 5/10/15/20-game questions). Per pool the opening model and the style
model are fitted on the candidates' games and score every question (one log score per candidate); each component alone
and their fusion give Top-5 lists. The fusion weights for a pool are fitted on the other pools' questions (no pool sees its
own); with a single pool the fusion uses equal weights. Pre-specified: fusion vs opening alone (player bootstrap of the
official per-question score). Writes --out (JSON) and, with --fusion-out, the fusion weights fitted on all pools for
predict_pid.py. Fold 0 is refused.

  python -I scripts/pid_cv.py --pools 1,2 3,4 --out results/pid_cv.json --fusion-out results/pid_fusion.json --workers 12
"""
import argparse  # 命令列參數
import sys  # 匯入路徑、錯誤輸出與結束碼
import time  # 計時
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 數值運算
import pandas as pd  # 讀取玩家分折表

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
if str(ROOT / "src") not in sys.path:  # 匯入路徑上還沒有 src 時
    sys.path.insert(0, str(ROOT / "src"))  # 讓 aicup_pid 套件可以匯入
import aicup_pid  # noqa: E402  （匯入時會把 AI_CUP_2026-1 放上 sys.path）
from aicup.ladder import to_json  # noqa: E402  輸出 JSON 的共用函式
from aicup.scoring import load_games  # noqa: E402  依棋局鍵讀取棋譜
from aicup_pid import pools  # noqa: E402  模擬題池與官方計分
from aicup_pid.components import COMPONENTS, fit_models, score_questions  # noqa: E402  開局與風格兩個成分
from aicup_pid.fusion import Fusion, topk  # noqa: E402  融合權重與取前 k 名
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置
from aicup_pid.provenance import pid_environment  # noqa: E402  記錄執行環境
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402  以玩家為單位的成對 bootstrap

PRE = [("fusion", "opening")]  # 事先指定的比較：融合對只用開局
REPORT = [("style", "opening"), ("fusion", "style")]  # 只回報的比較（不作為採用依據）


def parse_pools(specs):  # 把 --pools 的字串轉成各題池的折清單
    out = []  # 每個題池的折
    for s in specs:  # 逐一處理每個項目
        folds = sorted({int(x) for x in s.split(",") if x.strip()})  # 以逗號分開的折編號，去重後排序
        if not folds:  # 空的項目
            raise ValueError(f"empty pool spec {s!r}")  # 視為錯誤
        out.append(folds)  # 加入這個題池
    return out  # 回傳全部題池的折


def run_pool(games, folds_df, folds, a, cfg, log=print):  # 建一個模擬題池並算出各成分的分數矩陣
    t0 = time.time()  # 計時起點
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed,  # 抽出候選人與題目
                                      pool_id="p" + "".join(map(str, folds)), order=a.order)  # 題池代號由折編號組成
    if a.max_questions:  # 煙霧測試
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)  # 只留前 N 題
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks})  # 候選人與題目用到的全部棋局
    log(f"pool {folds}: {info['players']} players ({info['eligible']} eligible), {len(cands)} candidate games, {len(qs)} questions "  # 回報題池規模
        f"{info['questions_by_n']}; loading {len(keys)} games", flush=True)  # 與要載入的局數
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}  # 讀取棋譜文字
    missing = [k for k in keys if k not in sgfs]  # 訓練檔中找不到的棋局
    if missing:  # 有缺漏
        raise ValueError(f"{len(missing)} games not found in the training files, e.g. {missing[:3]}")  # 中止並列出例子
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()  # 候選人 → 其棋局鍵
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=a.workers)  # 在候選人的棋局上擬合開局與風格模型
    S = score_questions(opening, style, [list(k) for k in qs["game_keys"]], moves, feats)  # 開局與風格各自的分數矩陣（題目 × 候選人）
    truth = np.array([opening.index[p] for p in qs["player_id"]])  # 每題正解候選人的索引
    log(f"pool {folds}: components in {time.time() - t0:.0f} s", flush=True)  # 回報耗時
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "seconds": time.time() - t0}  # 題池資訊、候選人順序、題目、分數與正解


def evaluate(S, players, qs, truth):  # 由分數矩陣算出官方分數與逐題得分
    idx = topk(S, pools.TOP_K)  # 每題分數最高的 5 位候選人索引
    tops = [[players[j] for j in row] for row in idx]  # 換成玩家編號
    per_q = np.array([pools.question_score(t, players[y]) for t, y in zip(tops, truth)])  # 逐題的官方分數 e^-(r-1)
    return pools.summarise(tops, [players[y] for y in truth], qs["n_games"].to_numpy()), per_q  # 彙總指標與逐題分數


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="folds of each simulated pool, e.g. 1,2 3,4")  # 每個模擬題池的折
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)  # 每個題池的候選人數
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)  # 每位候選人的棋局數
    ap.add_argument("--seed", type=int, default=0)  # 抽題池的種子
    ap.add_argument("--depth", type=int, default=12)  # 開局模型看的手數
    ap.add_argument("--alpha", type=float, default=2.0)  # 開局模型向題池背景收縮的強度
    ap.add_argument("--beta", type=float, default=5.0)  # 開局模型背景分布的平滑參數
    ap.add_argument("--shrink", type=float, default=0.1)  # 風格模型共變異數的收縮係數
    ap.add_argument("--workers", type=int, default=1)  # 解析棋譜的平行行程數
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle", help="how a player's games are split: shuffle (default) or file = earliest games as the candidate's (drift stress test)")  # 玩家棋局的切分順序
    ap.add_argument("--n-boot", type=int, default=2000)  # bootstrap 次數
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests: only the first N questions of each pool")  # 煙霧測試用的題數上限
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 玩家分折表
    ap.add_argument("--out", required=True)  # 結果 JSON
    ap.add_argument("--fusion-out", default=None, help="write the fusion weights fitted on all pools (for predict_pid.py)")  # 融合權重的輸出檔
    a = ap.parse_args(argv)  # 解析參數
    try:  # 讀資料與建題池時的預期錯誤以結束碼 2 回報
        specs = parse_pools(a.pools)  # 解析題池設定
        if any(0 in f for f in specs):  # 第 0 折不可當交叉驗證的題池
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")  # 第 0 折保留給最後的檢查
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})  # 玩家分折表
        games = pools.black_games(aicup_pid.TRAIN_DIR)  # 訓練集所有的執黑棋局
        cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink}  # 成分模型的設定
        runs = [run_pool(games, folds_df, f, a, cfg) for f in specs]  # 逐題池建立並打分
    except (FileNotFoundError, ValueError, KeyError) as e:  # 預期內的錯誤
        print(e, file=sys.stderr)  # 印出原因
        return 2  # 以結束碼 2 結束
    results, per_q_all = [], {c: [] for c in (*COMPONENTS, "fusion")}  # 各題池的結果與各臂的逐題分數
    pids_all = []  # 各題池題目的正解玩家
    for i, r in enumerate(runs):  # 逐題池評估
        others = [o for j, o in enumerate(runs) if j != i]  # 其他題池
        if others:  # 有其他題池時交叉擬合
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in COMPONENTS], np.concatenate([o["truth"] for o in others]))  # 在其他題池的題目上擬合融合權重
            weights = fus.w.tolist()  # 記錄權重
        else:  # 只有一個題池
            fus = Fusion()  # 不擬合
            fus.w = np.ones(len(COMPONENTS))  # 改用相等權重
            weights = None  # 記為未擬合
        S_f = fus.scores([r["S"][c] for c in COMPONENTS])  # 這個題池的融合分數
        scores, per_q = {}, {}  # 各臂的彙總與逐題分數
        for c in COMPONENTS:  # 每個成分單獨評估
            scores[c], per_q[c] = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])  # 單一成分的分數
        scores["fusion"], per_q["fusion"] = evaluate(S_f, r["players"], r["questions"], r["truth"])  # 融合的分數
        for c in per_q:  # 收集逐題分數
            per_q_all[c].append(per_q[c])  # 供跨題池的成對比較
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))  # 正解玩家（bootstrap 的抽樣單位）
        results.append({"pool": r["info"], "scores": scores, "fusion_weights": weights, "seconds": r["seconds"]})  # 記錄這個題池的結果
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f} (top1 {scores[c]['top1']:.3f})" for c in scores)  # 印出各臂的分數
              + (f"  weights {np.round(weights, 3).tolist()}" if weights else "  (equal weights)"), flush=True)  # 與融合權重
    pid = np.concatenate(pids_all)  # 全部題目的正解玩家
    comps = {}  # 比較結果
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):  # 事先指定與只回報兩組
        comps[group] = {}  # 這一組的結果
        for x, z in pairs:  # 每一對比較
            d = paired_player_bootstrap(np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z]), pid, n=a.n_boot)  # 以玩家為單位的成對 bootstrap
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]  # 各題池的平均差
            d.update({"per_pool_diff": per_pool, "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool))})  # 通過＝區間下限 > 0 且每個題池都為正
            comps[group][f"{x} vs {z}"] = d  # 存入結果
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)  # 印出差值、區間與各題池的差
                  + f"  -> {'PASS' if d['pass'] else 'no'}", flush=True)  # 與判定
    overall = {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}  # 各臂在全部題目上的平均分數
    out = {"pools": [p for p in results], "overall_score": overall, "comparisons": comps, "cfg": cfg, "n_players": a.n_players,  # 結果文件
           "cand_games": a.cand_games, "seed": a.seed, "order": a.order, "questions": int(len(pid)), **pid_environment("cpu")}  # 含設定、題數與執行環境
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫出結果
    print(f"wrote {a.out}")  # 回報輸出位置
    if a.fusion_out:  # 需要融合權重檔時
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in COMPONENTS], np.concatenate([r["truth"] for r in runs]))  # 在全部題池上擬合融合權重
        doc = {"components": list(COMPONENTS), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs],  # 融合文件：成分、權重、設定與所用題池
               "questions": int(len(pid)), **pid_environment("cpu")}  # 題數與執行環境
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")  # 寫出融合文件
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")  # 回報權重
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
