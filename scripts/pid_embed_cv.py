#!/usr/bin/env python
"""Line B8: the learnt player embedding (aicup_pid.embed) on top of fusion3o_ref_trunk, pre-specified (METHOD_NOTES 2.23).

Pools: pid_ref_cv.run_pool (make_pool seed 0, fold pairs 1,2 / 3,4, references from the other pair), trunk rows from
--store. For each pool the embedding is trained on the OTHER fold pair's players (every Black game with a row in --store,
>= min_games rows), so no pool player is ever seen; two variants:
  emb   training players only
  embx  training players + the 800 players of the void 10-30 official candidate pools (--old-official folder and its
        trunk rows --old-store; labels = the organisers' player ids; the probed Public answers are NOT used)
Rows under a random board symmetry (--aug-store, pid_trunk_extract --sym-seed) serve as augmentation where present.
Arms (weights cross-fitted on the other pool): f3oreft (current main), f3oreft_emb, f3oreft_embx, f3oref_embx (embx in
place of trunk). Pre-specified: f3oreft_embx vs f3oreft (player bootstrap, both pools the same sign, >= 0.003 overall or
>= 0.005 at n=5). Report only: f3oreft_emb vs f3oreft, embx vs trunk, emb vs trunk, f3oref_embx vs f3oreft.
--model-dir keeps each pool's embedding (emb[x]_p<folds>.pt) for pid_final_cv.py; --final-model-out trains embx on folds
1-4 + the old official pools for the new official files; --fusion-out writes f3oreft_embx's weights fitted on both pools.
Fold 0 is refused.

  python -I scripts/pid_embed_cv.py --store data/trunk/train data/trunk/train_rest --aug-store data/trunk/train_sym0 \\
      --old-official data/tests --old-store data/trunk/official --model-dir results/embed --out results/embed/embed_cv.json \\
      --fusion-out results/pid_fusion3o_ref_trunk_emb.json --final-model-out results/embed/embx_final.pt
"""
import argparse  # 命令列參數
import sys  # 匯入路徑、錯誤輸出與結束碼
import time  # 計時
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 數值運算
import pandas as pd  # 讀取玩家分折表
import torch  # 訓練與保存嵌入

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
for d in (ROOT / "src", ROOT / "scripts"):  # 套件與同目錄的腳本
    if str(d) not in sys.path:  # 尚未在匯入路徑上
        sys.path.insert(0, str(d))  # 加到最前面
import aicup_pid  # noqa: E402  匯入前須先設定 sys.path
from aicup.ladder import to_json  # noqa: E402  輸出 JSON 的共用函式
from aicup_pid import pools  # noqa: E402  模擬題池與官方候選人檔的讀取
from aicup_pid.embed import PlayerEmbedding, score_embed  # noqa: E402  學習式玩家嵌入
from aicup_pid.fusion import Fusion  # noqa: E402  融合權重
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置
from aicup_pid.provenance import pid_environment  # noqa: E402  記錄執行環境
from aicup_pid.stats import paired_player_bootstrap  # noqa: E402  以玩家為單位的成對 bootstrap
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402  主幹成分與每局主幹列的讀取
import pid_ref_cv  # noqa: E402  沿用其題池建立與五個成分
from pid_cv import evaluate, parse_pools  # noqa: E402  沿用 pid_cv.py 的評估與題池解析

MIN_DIFF, MIN_DIFF_N5 = 0.003, 0.005  # 通過門檻：整體差值，或 5 局題的差值
BASE = pid_ref_cv.FIVE  # 開局、風格、局部、方向、參考玩家
MAIN = (*BASE, "trunk")  # 現行主線：再加主幹成分
FUSIONS = (("f3oreft", MAIN), ("f3oreft_emb", (*MAIN, "emb")), ("f3oreft_embx", (*MAIN, "embx")), ("f3oref_embx", (*BASE, "embx")))  # 要評估的各種融合與其成分
PRE = [("f3oreft_embx", "f3oreft")]  # 事先指定的比較
REPORT = [("f3oreft_emb", "f3oreft"), ("embx", "trunk"), ("emb", "trunk"), ("f3oref_embx", "f3oreft")]  # 只回報的比較（不作為採用依據）
OFFICIAL = ("player_identification_candidates_public.csv", "player_identification_candidates_private.csv")  # 舊官方題池的兩個候選人檔


def old_official_rows(folder, store):  # 10-30 版舊官方候選人題池的每局主幹列
    """{'old:<player_id>': [G, 256]} of the old official candidate pools (keys c:<row>:<game_id> as pid_trunk_extract)."""
    out = {}  # 玩家 → 各局的列
    for name in OFFICIAL:  # Public 與 Private 兩個候選人檔
        cands = pools.read_candidates(Path(folder) / name)  # 讀取候選人檔
        for i, r in enumerate(cands.itertuples()):  # 逐列處理（列號是棋局鍵的一部分）
            out.setdefault(f"old:{r.player_id}", []).append(store[f"c:{i}:{r.game_id}"])  # 加上 old: 前綴，與訓練玩家區分
    return {p: np.stack(v) for p, v in out.items()}  # 每位玩家疊成 [G, 256]


def training_rows(games, folds_df, folds, store, aug):  # 指定各折訓練玩家的列與擴增列
    """({player: [G, 256]}, {player: [G, 256] augmented, plain where no augmented row}) of the given folds' Black games."""
    g = games.merge(folds_df, on="player_id")  # 每局附上擁有者的折
    g = g[g.fold.isin(list(folds)) & g.game_key.isin(store.keys())]  # 只留指定折且有主幹列的棋局
    rows, arows = {}, {}  # 原本的列與擴增列
    for p, keys in g.groupby("player_id")["game_key"]:  # 逐玩家處理
        keys = list(keys)  # 該玩家的棋局鍵
        rows[p] = np.stack([store[k] for k in keys])  # 原本的列
        arows[p] = np.stack([aug.get(k, store[k]) for k in keys])  # 擴增列（沒有時用原本的列）
    return rows, arows  # 回傳兩組列


def train(rows, arows, cfg, log):  # 訓練一個玩家嵌入
    t = time.time()  # 計時起點
    m = PlayerEmbedding(**cfg).fit(rows, arows, log=log)  # 以情節方式訓練
    log(f"  trained on {m.players_trained} players in {time.time() - t:.0f} s", flush=True)  # 回報訓練玩家數與耗時
    return m  # 回傳模型


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--store", nargs="+", required=True)  # 訓練棋局的每局主幹列資料夾
    ap.add_argument("--aug-store", nargs="*", default=[])  # 隨機棋盤對稱版本的列（資料擴增）
    ap.add_argument("--old-official", default=str(ROOT / "data" / "tests"))  # 10-30 版舊官方檔的資料夾
    ap.add_argument("--old-store", nargs="+", default=[str(ROOT / "data" / "trunk" / "official")])  # 舊官方棋局的主幹列
    ap.add_argument("--pools", nargs="+", default=["1,2", "3,4"])  # 兩組折對
    ap.add_argument("--steps", type=int, default=3000)  # 嵌入的訓練步數
    ap.add_argument("--min-games", type=int, default=8)  # 玩家進入訓練所需的最少列數
    ap.add_argument("--workers", type=int, default=1)  # 解析棋譜的平行行程數
    ap.add_argument("--n-boot", type=int, default=2000)  # bootstrap 次數
    ap.add_argument("--model-dir", default=None)  # 保存各題池嵌入的資料夾
    ap.add_argument("--out", required=True)  # 結果 JSON
    ap.add_argument("--fusion-out", default=None)  # f3oreft_embx 的融合文件
    ap.add_argument("--final-model-out", default=None, help="repo-relative path (recorded in the fusion document as emb_model_official)")  # 最終 embx 的權重檔
    ap.add_argument("--fusion-out-plain", default=None, help="fusion document of main + emb trained on training players only (no official game)")  # 主線＋只用訓練玩家訓練之 emb 的融合文件
    ap.add_argument("--final-plain-model-out", default=None, help="repo-relative path of the training-only embedding (folds 1-4), recorded in --fusion-out-plain")  # 只用第 1-4 折訓練的最終嵌入
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    log = print  # 紀錄直接印出
    ecfg = {"steps": a.steps, "min_games": a.min_games, "seed": 0}  # 嵌入的訓練設定
    defaults = argparse.Namespace(n_players=pools.N_CANDIDATES, cand_games=pools.CAND_GAMES, seed=0, order="shuffle", max_questions=0, workers=a.workers,  # 給 pid_ref_cv.run_pool 的題池參數（同其預設）
                                  ref_min_black=30, ref_cap=40)  # 參考玩家的執黑局數門檻與局數上限
    cfg = {"depth": 12, "alpha": 2.0, "beta": 5.0, "shrink": 0.1, "local_alpha": 2.0, "local_beta": 5.0, "local_start": 12, "orient_alpha": 2.0,  # 各成分的設定（同 pid_ref_cv.py 的預設）
           "orient_beta": 5.0, "orient_depth": 12, "orient_raw": 1, "ref_alpha": 2.0, "ref_beta": 5.0, "ref_min_black": 30, "ref_cap": 40, "ref_orient_raw": 2,  # 方向與參考玩家成分
           "ref_seed": 0, "trunk_feats": "all", "trunk_shrink": 0.1}  # 參考玩家的種子；主幹成分用全部 256 維、收縮 0.1
    specs = parse_pools(a.pools)  # 解析題池設定
    if len(specs) != 2 or any(0 in f for f in specs):  # 必須正好兩組折對且不含第 0 折
        print("give exactly two fold pairs without fold 0", file=sys.stderr)  # 說明原因
        return 2  # 以結束碼 2 結束
    store, aug = load_store(*a.store), (load_store(*a.aug_store) if a.aug_store else {})  # 讀入主幹列與擴增列
    old = old_official_rows(a.old_official, load_store(*a.old_store))  # 舊官方題池玩家的列
    log(f"stores: {len(store)} training rows, {len(aug)} augmentation rows, {len(old)} old official players ({sum(len(v) for v in old.values())} rows); "  # 回報資料量
        f"device {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)  # 與使用的裝置
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 玩家分折表
    games = pools.black_games(aicup_pid.TRAIN_DIR)  # 訓練集所有的執黑棋局
    if a.model_dir:  # 要保存各題池的嵌入時
        Path(a.model_dir).mkdir(parents=True, exist_ok=True)  # 建立資料夾
    runs = []  # 各題池的結果
    for i, f in enumerate(specs):  # 逐題池處理
        r = pid_ref_cv.run_pool(games, folds_df, f, specs[1 - i], defaults, cfg)  # 五個成分（參考玩家取自另一組折對）
        model = fit_trunk(r["by_player"], store, cfg)  # 在候選人的主幹列上擬合主幹成分
        if model.players != r["players"]:  # 候選人順序必須一致
            raise ValueError("candidate order differs")  # 否則分數矩陣的欄會錯位
        r["S"]["trunk"] = score_trunk(model, r["q_keys"], store)  # 主幹成分的分數矩陣
        rows, arows = training_rows(games, folds_df, specs[1 - i], store, aug)  # 另一組折對玩家的訓練列
        if set(rows) & set(r["by_player"]):  # 題池玩家不可出現在訓練集
            raise ValueError("a pool player is in the embedding's training set")  # 發現就中止
        tag = "p" + "".join(map(str, f))  # 題池代號，用於檔名
        for name, extra in (("emb", {}), ("embx", old)):  # 兩種變體：只用訓練玩家／再加舊官方題池玩家
            log(f"pool {f}: training {name} on folds {specs[1 - i]}" + (" + old official pools" if extra else ""), flush=True)  # 回報正在訓練的變體
            m = train({**rows, **extra}, {**arows, **extra}, ecfg, log)  # 訓練嵌入
            r["S"][name] = score_embed(m, r["by_player"], r["q_keys"], store)  # 嵌入成分的分數矩陣
            r.setdefault("emb_history", {})[name] = m.history  # 保存訓練歷程
            if a.model_dir:  # 要保存時
                torch.save(m.state_dict(), Path(a.model_dir) / f"{name}_{tag}.pt")  # 寫出這個題池的嵌入
        runs.append(r)  # 收集題池結果
    comps_all = (*MAIN, "emb", "embx")  # 所有單一成分
    arms = (*comps_all, *[n for n, _ in FUSIONS])  # 所有要評估的臂
    results, per_q_all, pids_all, n_all = [], {c: [] for c in arms}, [], []  # 各題池結果、逐題分數、正解玩家、題目局數
    for i, r in enumerate(runs):  # 逐題池評估
        others = [o for j, o in enumerate(runs) if j != i]  # 另一個題池
        weights, fused = {}, {}  # 各融合的權重與分數
        for name, comps in FUSIONS:  # 每一種融合
            fus = Fusion().fit([np.vstack([o["S"][c] for o in others]) for c in comps], np.concatenate([o["truth"] for o in others]))  # 在另一個題池上擬合權重
            weights[name], fused[name] = fus.w.tolist(), fus.scores([r["S"][c] for c in comps])  # 記錄權重並套用到這個題池
        scores = {}  # 各臂的彙總分數
        for c in comps_all:  # 單一成分
            scores[c], pq = evaluate(r["S"][c], r["players"], r["questions"], r["truth"])  # 單一成分的彙總與逐題分數
            per_q_all[c].append(pq)  # 收集逐題分數
        for name, _ in FUSIONS:  # 各種融合
            scores[name], pq = evaluate(fused[name], r["players"], r["questions"], r["truth"])  # 融合的彙總與逐題分數
            per_q_all[name].append(pq)  # 收集逐題分數
        pids_all.append(r["questions"]["player_id"].to_numpy(dtype=str))  # 正解玩家（bootstrap 的抽樣單位）
        n_all.append(r["questions"]["n_games"].to_numpy())  # 每題的局數
        results.append({"pool": r["info"], "scores": scores, "weights": weights, "emb_history": r["emb_history"]})  # 記錄這個題池的結果
        log(f"pool {r['info']['folds']}: " + "  ".join(f"{c} {scores[c]['score']:.4f}" for c in scores) + f"  embx weights {np.round(weights['f3oreft_embx'], 3).tolist()}", flush=True)  # 印出各臂分數與 f3oreft_embx 的權重
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
            log(f"[{group}] {x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  pools " + " ".join(f"{v:+.4f}" for v in per_pool)  # 印出差值、區間與各題池的差
                + f"  n=5 {d['diff_n5']:+.4f}  -> {'PASS' if d['pass'] else 'no'}", flush=True)  # 與 5 局題的差值和判定
    out = {"pools": results, "overall_score": {c: float(np.mean(np.concatenate(per_q_all[c]))) for c in per_q_all}, "comparisons": comps, "cfg": cfg,  # 結果文件：各題池、各臂平均分數、比較與設定
           "embed_cfg": ecfg, "questions": int(len(pid)), "seconds": time.time() - t0, **pid_environment("cuda" if torch.cuda.is_available() else "cpu")}  # 嵌入設定、題數、耗時與執行環境
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫出結果
    log(f"wrote {a.out}")  # 回報輸出位置
    if a.fusion_out:  # 需要 f3oreft_embx 的融合文件時
        comps_ = (*MAIN, "embx")  # 擬合用的成分
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps_], np.concatenate([r["truth"] for r in runs]))  # 在兩個題池上擬合
        doc = {"components": [*MAIN, "emb"], "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),  # 成分名稱寫成 emb，實際模型由 emb_model_official 指定（embx）
               "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", "emb_model_official": a.final_model_out,  # 官方檔使用的參考折、主幹列與嵌入模型
               **pid_environment("cpu")}  # 執行環境
        Path(a.fusion_out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        Path(a.fusion_out).write_text(to_json(doc, indent=1) + "\n")  # 寫出融合文件
        log(f"wrote {a.fusion_out}: weights {np.round(fus.w, 4).tolist()}")  # 回報權重
    if a.fusion_out_plain:  # 11-10：10-30 的題池在 11-11 的版本中仍然有效，所以備妥一個從未看過任何官方棋局的嵌入
        comps_ = (*MAIN, "emb")  # 擬合用的成分
        fus = Fusion().fit([np.vstack([r["S"][c] for r in runs]) for c in comps_], np.concatenate([r["truth"] for r in runs]))  # 在兩個題池上擬合
        doc = {"components": list(comps_), "weights": fus.w.tolist(), "cfg": cfg, "fitted_on_pools": [r["info"]["folds"] for r in runs], "questions": int(len(pid)),  # 融合文件：成分、權重、設定、所用題池與題數
               "ref_folds_official": [1, 2, 3, 4], "trunk_store_official": "data/trunk/official", "emb_model_official": a.final_plain_model_out,  # 官方檔使用的參考折、主幹列與只用訓練玩家的嵌入
               "note": "emb trained on training folds 1-4 only: no official candidate or question game enters any training", **pid_environment("cpu")}  # 註明沒有任何官方棋局進入訓練
        Path(a.fusion_out_plain).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
        Path(a.fusion_out_plain).write_text(to_json(doc, indent=1) + "\n")  # 寫出融合文件
        log(f"wrote {a.fusion_out_plain}: weights {np.round(fus.w, 4).tolist()}")  # 回報權重
    if a.final_plain_model_out:  # 需要只用訓練玩家的最終嵌入時
        rows, arows = training_rows(games, folds_df, (1, 2, 3, 4), store, aug)  # 第 1-4 折全部玩家的列
        log("final emb: folds 1-4 only", flush=True)  # 回報
        m = train(rows, arows, ecfg, log)  # 只用訓練玩家訓練最終 emb
        dest = Path(a.final_plain_model_out) if Path(a.final_plain_model_out).is_absolute() else ROOT / a.final_plain_model_out  # 相對路徑以儲存庫根目錄為準
        dest.parent.mkdir(parents=True, exist_ok=True)  # 建立資料夾
        torch.save(m.state_dict(), dest)  # 寫出嵌入權重
        log(f"wrote {dest}")  # 回報輸出位置
    if a.final_model_out:  # 需要最終 embx 嵌入時
        rows, arows = training_rows(games, folds_df, (1, 2, 3, 4), store, aug)  # 第 1-4 折全部玩家的列
        log("final embx: folds 1-4 + old official pools", flush=True)  # 回報
        m = train({**rows, **old}, {**arows, **old}, ecfg, log)  # 加上舊官方題池的玩家一起訓練
        dest = Path(a.final_model_out) if Path(a.final_model_out).is_absolute() else ROOT / a.final_model_out  # 相對於儲存庫根目錄，與文件記錄的一致
        dest.parent.mkdir(parents=True, exist_ok=True)  # 建立資料夾
        torch.save(m.state_dict(), dest)  # 寫出嵌入權重
        log(f"wrote {dest}")  # 回報輸出位置
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
