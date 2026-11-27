#!/usr/bin/env python
"""Line B7: reference-player affinity (aicup_pid.ref) as an extra fusion component, pre-specified vs fusion3o.

Pools from one fold pair (make_pool, seed 0); reference players from the OTHER fold pair (>= --ref-min-black Black games,
up to --ref-cap games each), so no reference is a pool player. Per pool: opening, style, local, orient (N=1) as in
fusion3o, plus the ref component (cosine of mean z-scored log-likelihood profiles over the references). Fusion weights
cross-fitted on the other pool. Pre-specified: fusion3o_ref vs fusion3o; report only: ref vs style, ref vs opening,
fusion3_ref vs fusion3. Fold 0 is refused.

  python -I scripts/pid_ref_cv.py --pools 1,2 3,4 --workers 12 --out results/ref/ref_cv.json --fusion-out results/pid_fusion3o_ref.json
"""
import argparse  # 命令列參數
import sys  # 匯入路徑、錯誤輸出與結束碼
import time  # 計時
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 數值運算
import pandas as pd  # 讀取玩家分折表

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
for d in (ROOT / "src", ROOT / "scripts"):  # 套件與同目錄的腳本
    if str(d) not in sys.path:  # 尚未在匯入路徑上
        sys.path.insert(0, str(d))  # 加到最前面
import aicup_pid  # noqa: E402  （匯入時會把 AI_CUP_2026-1 放上 sys.path）
from aicup.ladder import to_json  # noqa: E402  輸出 JSON 的共用函式
from aicup.scoring import load_games  # noqa: E402  依棋局鍵讀取棋譜
from aicup_pid import pools  # noqa: E402  模擬題池與官方計分
from aicup_pid.components import COMPONENTS, features_all, fit_from_parsed, fit_local, fit_orient, fit_ref, parse_all, score_components, score_ref  # noqa: E402  各成分的擬合與打分
from aicup_pid.fusion import Fusion  # noqa: E402  融合權重
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置
from aicup_pid.provenance import pid_environment  # noqa: E402  記錄執行環境
from aicup_pid.ref import reference_players  # noqa: E402  選出參考玩家
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402  以玩家為單位的成對 bootstrap
from pid_cv import evaluate, parse_pools  # noqa: E402  沿用 pid_cv.py 的評估與題池解析

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005  # 通過門檻：整體差值，或 5 局題的差值
THREE = (*COMPONENTS, "local")  # fusion3 的成分：開局、風格、局部
FOUR = (*THREE, "orient")  # fusion3o 的成分：再加方向
FIVE = (*FOUR, "ref")  # fusion3o_ref 的成分：再加參考玩家
FUSIONS = (("fusion", COMPONENTS), ("fusion3", THREE), ("fusion3o", FOUR), ("fusion3_ref", (*THREE, "ref")), ("fusion3o_ref", FIVE))  # 要評估的各種融合與其成分
PRE = [("fusion3o_ref", "fusion3o")]  # 事先指定的比較
REPORT = [("ref", "style"), ("ref", "opening"), ("fusion3_ref", "fusion3")]  # 只回報的比較（不作為採用依據）


def run_pool(games, folds_df, folds, ref_folds, a, cfg, log=print):  # 建一個題池並算出五個成分的分數矩陣
    t0 = time.time()  # 計時起點
    cands, qs, info = pools.make_pool(games, folds_df, folds, n_players=a.n_players, cand_games=a.cand_games, seed=a.seed, pool_id="p" + "".join(map(str, folds)), order=a.order)  # 抽出候選人與題目
    if a.max_questions:  # 煙霧測試
        qs = qs.iloc[:a.max_questions].reset_index(drop=True)  # 只留前 N 題
    refs = reference_players(games, folds_df, ref_folds, min_black=a.ref_min_black, cap=a.ref_cap, seed=a.seed, exclude=set(cands["player_id"]))  # 從另一組折對選參考玩家，排除題池玩家
    keys = sorted(set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks} | {k for ks in refs.values() for k in ks})  # 候選人、題目與參考玩家的全部棋局
    log(f"pool {folds}: {info['players']} players, {len(cands)} candidate games, {len(qs)} questions; references from folds {ref_folds}: {len(refs)} players, "  # 回報題池與參考玩家的規模
        f"{sum(len(v) for v in refs.values())} games; loading {len(keys)} games", flush=True)  # 與要載入的局數
    sgfs = {k: v[0] for k, v in load_games(keys, aicup_pid.TRAIN_DIR).items()}  # 讀取棋譜文字
    moves, feats = parse_all(sgfs, a.workers), features_all({k: sgfs[k] for k in set(cands["game_key"]) | {k for ks in qs["game_keys"] for k in ks}}, a.workers)  # 解析全部棋局；風格特徵只算候選人與題目的棋局
    by_player = cands.groupby("player_id")["game_key"].apply(list).to_dict()  # 候選人 → 其棋局鍵
    opening, style = fit_from_parsed(by_player, moves, feats, cfg)  # 擬合開局與風格模型
    local, orient = fit_local(by_player, moves, cfg), fit_orient(by_player, moves, cfg)  # 擬合局部棋形與方向模型
    q_keys = [list(k) for k in qs["game_keys"]]  # 每題的棋局鍵
    t1 = time.time()  # 參考玩家成分的計時起點
    ref_model = fit_ref(refs, moves, cfg)  # 在參考玩家的棋局上擬合
    S_ref = score_ref(ref_model, by_player, q_keys, moves)  # 題目與候選人剖面之間的餘弦分數
    log(f"pool {folds}: reference model {len(ref_model.players)} players x {ref_model.logp.shape[1]} tokens, profiles and cosines in {time.time() - t1:.0f} s", flush=True)  # 回報參考模型的大小與耗時
    S = score_components(FIVE, opening, style, q_keys, moves, feats, local=local, orient=orient, ref=S_ref)  # 五個成分的分數矩陣
    truth = np.array([opening.index[p] for p in qs["player_id"]])  # 每題正解候選人的索引
    log(f"pool {folds}: components in {time.time() - t0:.0f} s", flush=True)  # 回報耗時
    return {"info": info, "players": opening.players, "questions": qs, "S": S, "truth": truth, "n_refs": len(ref_model.players), "ref_tokens": int(ref_model.logp.shape[1]),  # 題池資訊、分數、正解與參考模型規模
            "by_player": by_player, "q_keys": q_keys, "seconds": time.time() - t0}  # 以及後續成分（pid_embed_cv.py）需要的棋局清單


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"], help="exactly two fold pairs: each pool's references come from the other")  # 兩組折對
    ap.add_argument("--n-players", type=int, default=pools.N_CANDIDATES)  # 每個題池的候選人數
    ap.add_argument("--cand-games", type=int, default=pools.CAND_GAMES)  # 每位候選人的棋局數
    ap.add_argument("--seed", type=int, default=0)  # 抽題池與參考玩家的種子
    ap.add_argument("--depth", type=int, default=12)  # 開局模型看的手數
    ap.add_argument("--alpha", type=float, default=2.0)  # 開局與參考玩家模型向背景收縮的強度
    ap.add_argument("--beta", type=float, default=5.0)  # 開局與參考玩家模型背景分布的平滑參數
    ap.add_argument("--shrink", type=float, default=0.1)  # 風格模型共變異數的收縮係數
    ap.add_argument("--local-alpha", type=float, default=2.0)  # 局部棋形模型的收縮強度
    ap.add_argument("--local-beta", type=float, default=5.0)  # 局部棋形模型背景的平滑參數
    ap.add_argument("--start", type=int, default=12)  # 局部棋形從第幾手開始計
    ap.add_argument("--orient-alpha", type=float, default=2.0)  # 方向模型的收縮強度
    ap.add_argument("--orient-beta", type=float, default=5.0)  # 方向模型背景的平滑參數
    ap.add_argument("--orient-depth", type=int, default=12)  # 方向模型判斷對稱所看的手數
    ap.add_argument("--orient-raw", type=int, default=1)  # 方向模型記錄原始座標的手數
    ap.add_argument("--ref-min-black", type=int, default=30, help="a reference player needs this many Black games")  # 參考玩家至少要有的執黑局數
    ap.add_argument("--ref-cap", type=int, default=40, help="games per reference player")  # 每位參考玩家使用的局數上限
    ap.add_argument("--ref-orient-raw", type=int, default=2, help="raw moves in the reference orientation tokens")  # 參考玩家方向記號的原始座標手數
    ap.add_argument("--workers", type=int, default=1)  # 解析棋譜的平行行程數
    ap.add_argument("--order", choices=list(pools.ORDERS), default="shuffle")  # 玩家棋局的切分順序
    ap.add_argument("--n-boot", type=int, default=2000)  # bootstrap 次數
    ap.add_argument("--max-questions", type=int, default=0, help="smoke tests only")  # 煙霧測試用的題數上限
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 玩家分折表
    ap.add_argument("--out", required=True)  # 結果 JSON
    ap.add_argument("--fusion-out", default=None, help="write the fusion3o_ref weights fitted on all pools (for predict_pid.py)")  # fusion3o_ref 權重的輸出檔
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    cfg = {"depth": a.depth, "alpha": a.alpha, "beta": a.beta, "shrink": a.shrink, "local_alpha": a.local_alpha, "local_beta": a.local_beta, "local_start": a.start,  # 成分模型的設定：開局、風格、局部棋形
           "orient_alpha": a.orient_alpha, "orient_beta": a.orient_beta, "orient_depth": a.orient_depth, "orient_raw": a.orient_raw,  # 方向
           "ref_alpha": a.alpha, "ref_beta": a.beta, "ref_min_black": a.ref_min_black, "ref_cap": a.ref_cap, "ref_orient_raw": a.ref_orient_raw, "ref_seed": a.seed}  # 參考玩家（沿用開局的 alpha、beta）
    try:  # 讀資料與建題池時的預期錯誤以結束碼 2 回報
        specs = parse_pools(a.pools)  # 解析題池設定
        if len(specs) != 2:  # 參考玩家要取自另一組，所以必須正好兩組
            raise ValueError("give exactly two fold pairs")  # 否則中止
        if any(0 in f for f in specs):  # 第 0 折不可當交叉驗證的題池
            raise ValueError("fold 0 is kept for the final check; it is not a CV pool")  # 第 0 折保留給最後的檢查
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})  # 玩家分折表
        games = pools.black_games(aicup_pid.TRAIN_DIR)  # 訓練集所有的執黑棋局
        runs = [run_pool(games, folds_df, f, specs[1 - i], a, cfg) for i, f in enumerate(specs)]  # 每個題池的參考玩家取自另一組折對
    except (FileNotFoundError, ValueError, KeyError) as e:  # 預期內的錯誤
        print(e, file=sys.stderr)  # 印出原因
        return 2  # 以結束碼 2 結束
    arms = (*FIVE, *[name for name, _ in FUSIONS])  # 所有要評估的臂：五個成分與各融合
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []  # 各題池結果、逐題分數、正解玩家、題目局數
    for i, r in enumerate(runs):  # 逐題池評估
        others = [o for j, o in enumerate(runs) if j != i]  # 另一個題池
        weights, fused = {}, {}  # 各融合的權重與分數
        for name, comps in FUSIONS:  # 每一種融合
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))  # 在另一個題池上擬合權重
            weights[name] = fus.w.tolist()  # 記錄權重
            fused[name] = fus.scores([r["S"][c] for c in comps])  # 套用到這個題池
        scores = {}  # 各臂的彙總分數
        for c in FIVE:  # 五個成分單獨評估
            scores[c], per_q = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])  # 單一成分的分數
            per_q_all[c].append(per_q)  # 收集逐題分數
        for name, _ in FUSIONS:  # 各種融合
            scores[name], per_q = evaluate(fused[name], r["players"], r["questions"], r["truth"])  # 融合的分數
            per_q_all[name].append(per_q)  # 收集逐題分數
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))  # 正解玩家（bootstrap 的抽樣單位）
        n_all.append(r["questions"]["n_games"].to_numpy())  # 每題的局數
        results.append({"pool": r["info"], "scores": scores, **{f"{name}_weights": weights[name] for name, _ in FUSIONS}, "n_refs": r["n_refs"], "ref_tokens": r["ref_tokens"], "seconds": r["seconds"]})  # 記錄這個題池的結果、各融合的權重與參考模型規模
        print(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores) + f"  weights3o_ref {np.round(weights['fusion3o_ref'], 3).tolist()}", flush=True)  # 印出各臂分數與 fusion3o_ref 的權重
    pid, n_games = np.concatenate(pids_all), np.concatenate(n_all)  # 合併兩個題池
    comps = {}  # 比較結果
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):  # 事先指定與只回報兩組
        comps[group] = {}  # 這一組的結果
        for x, z in pairs:  # 每一對比較
            qx, qz = np.concatenate(per_q_all[x]), np.concatenate(per_q_all[z])  # 兩臂的逐題分數
            d = paired_player_bootstrap(qx, qz, pid, n=a.n_boot)  # 以玩家為單位的成對 bootstrap
            per_pool = [float(np.mean(px - pz)) for px, pz in zip(per_q_all[x], per_q_all[z])]  # 各題池的平均差
            n5 = n_games == 5  # 5 局題
            d["diff_n5"] = float(np.mean((qx - qz)[n5])) if n5.any() else None  # 5 局題的平均差
            d.update({"per_pool_diff": per_pool, "by_n": {str(n): float(np.mean((qx - qz)[n_games == n])) for n in sorted(set(n_games.tolist()))},  # 各題池的差值與按題目局數分的差值
                      "pass": bool(d["ci95"][0] > 0 and all(v > 0 for v in per_pool) and (d["diff"] >= MIN_DIFF or ((d["diff_n5"] or 0) >= MIN_DIFF_N5)))})  # 通過：區間下限 > 0、兩題池皆正，且整體或 5 局題達門檻
            comps[group][f"{x} vs {z}"] = d  # 存入結果
            print(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)  # 印出差值、區間與各題池的差
                  + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)  # 與 5 局題的差值和判定
    out = {"pools": results, "overall_score": {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}, "comparisons": comps, "cfg": cfg,  # 結果文件：各題池、各臂平均分數、比較與設定
           "order": a.order, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cpu")}  # 切分順序、題數、耗時與執行環境
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫出結果
    print(f"wrote {a.out}")  # 回報輸出位置
    if a.fusion_out:  # 需要融合文件時
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in FIVE], np.concatenate([r["truth"] for r in runs]))  # 在兩個題池上擬合 fusion3o_ref 的權重
        doc = {"components": list(FIVE), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),  # 融合文件：成分、權重、設定、所用題池與題數
               "ref_folds_official": [1, 2, 3, 4], **pid_environment("cpu")}  # 官方檔的參考玩家取自第 1-4 折
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")  # 寫出融合文件
        print(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")  # 回報權重
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
