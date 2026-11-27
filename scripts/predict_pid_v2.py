#!/usr/bin/env python
"""The 11-08 v2 release -> one submission of 1,600 rows (METHOD_NOTES 2.29).

The original pools (pub_ / priv_, byte-identical to the 10-30 files) keep the rows of an existing submission (--old-submission:
predict_pid.py's output on those files). The added pools (pub2_ / priv2_: 600 candidates x 10 games, questions of 1-5 games,
either colour) are scored here with aicup_pid.v2.components and the fusion document of experiments/pid_v2_sim.py; --assign auto
applies the one-to-one assignment only when aicup_pid.checks' rule v2 says the pool looks one-to-one. Games that do not replay
(aicup.submission.check_game) or have no trunk row are left out; a question without a usable game takes the first five
candidates and is listed. The result is checked against the sample submission before it is written.
"""
import argparse  # 解析命令列參數
import json  # 讀融合文件、寫檢查報告
import sys  # 設定匯入路徑與結束碼
import time  # 計算執行時間
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 分數矩陣運算
import pandas as pd  # 讀寫提交檔與範例檔

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（scripts/ 的上一層）
for d in (ROOT / "src", ROOT / "scripts"):  # 讓本儲存庫的套件與腳本可以匯入
    if str(d) not in sys.path:  # 避免重複加入
        sys.path.insert(0, str(d))  # 放在搜尋路徑最前面
import aicup_pid  # noqa: E402,F401  匯入時會把 AI_CUP_2026-1 加進 sys.path
from aicup.submission import check_game  # noqa: E402  檢查棋局能否重播
from aicup_pid import pools, v2  # noqa: E402  官方題池工具與新增池的成分
from aicup_pid.assign import assign_sinkhorn, assign_soft, assign_topk  # noqa: E402  三種配對方法
from aicup_pid.checks import top1_collisions  # noqa: E402  Top-1 碰撞統計（含規則 v2 的一對一判斷）
from aicup_pid.components import parse_all  # noqa: E402  平行解析棋譜
from aicup_pid.embed import PlayerEmbedding  # noqa: E402  每局嵌入模型
from aicup_pid.fusion import Fusion, topk  # noqa: E402  線性融合與取前 k 名
from aicup_pid.trunkstyle import load_store  # noqa: E402  讀取 256 維的每局主幹列


def predict_pool(cand_path, q_path, doc, emb, store, assign, workers, log=print, seq=None, big_lam=None, big_tau=None, lend=None, prior3=None):  # 對一個新增池計分、配對並產生每題前 5 名
    cands, questions, qids, sgfs = v2.official_pool(cand_path, q_path)  # 讀官方候選檔與題目檔
    colour = {k: c for games in (*cands.values(), *questions) for k, c in games}  # 每局棋中要辨識那一方的執色
    need_rows = "emb" in doc["components"]  # 融合含嵌入成分時才需要主幹列
    bad = sorted(k for k in sgfs if not check_game(sgfs[k], colour[k])[0] or (need_rows and k not in store) or (seq is not None and k not in (seq[0] if isinstance(seq, list) else seq)))  # 無法重播、缺主幹列或缺序列網路向量（只查第一個網路）的棋局
    if bad:  # 有要排除的棋局時
        log(f"  {len(bad)} games left out (do not replay or have no trunk row): {bad[:5]}", flush=True)  # 回報排除的局數與例子
    cands = {p: [(k, c) for k, c in g if k not in bad] for p, g in cands.items()}  # 候選人的參考棋去掉排除的棋局
    empty_c = sorted(p for p, g in cands.items() if not g)  # 沒有任何可用棋局的候選人
    if empty_c:  # 有候選人完全沒有棋局可比
        raise ValueError(f"{cand_path}: candidates without a usable game: {empty_c[:5]}")  # 無法計分，直接中止
    q_use = [[(k, c) for k, c in q if k not in bad] for q in questions]  # 題目的棋局去掉排除的棋局
    usable = [i for i, q in enumerate(q_use) if q]  # 至少還剩一局的題目索引
    moves = parse_all({k: s for k, s in sgfs.items() if k not in bad}, workers)  # 平行解析可用棋局的落子
    S = v2.components(cands, [q_use[i] for i in usable], moves, sgfs, doc["cfg"], store, emb, which=doc["components"], seq=seq)  # 融合文件列出的各成分分數矩陣
    fus = Fusion()  # 建立融合器
    fus.w = np.asarray(doc["weights"], dtype=float)  # 直接套用融合文件的權重（不重新擬合）
    F = fus.scores([S[c] for c in doc["components"]])  # 融合後的 [題數, 候選人數] 分數
    players = sorted(cands)  # 候選人依 id 排序，與分數矩陣的欄一致
    n = np.array([len(q_use[i]) for i in usable])  # 每題實際可用的局數
    col = top1_collisions(F, n)  # Top-1 碰撞統計，供 auto 模式判斷與報告
    applied = assign in ("on", "big") or (assign == "auto" and bool(col["one_to_one_likely_v2"])) or (assign == "soft" and doc.get("soft_lam") is not None)  # 這個題池是否套用配對
    base = topk(F, pools.TOP_K)  # 不配對時的逐題前 5 名
    if assign == "big":  # METHOD_NOTES 2.37：2 局以上的題目構成一個一對一集合；1 局題維持逐題處理
        idx, rows_big = base.copy(), np.flatnonzero(n > 1)  # 從逐題結果出發，取出多局題的列
        if big_tau is not None:  # METHOD_NOTES 2.122：Sinkhorn 價格（搭配 --big-lam：Top-1 取其軟性配對，其後依扣除價格的順序）
            idx[rows_big] = assign_sinkhorn(F[rows_big], big_tau, pools.TOP_K, lam=big_lam)  # 最終設定走這條（λ 2、τ 0.5）
        else:  # 沒有指定 τ 時
            idx[rows_big] = assign_topk(F[rows_big], pools.TOP_K) if big_lam is None else assign_soft(F[rows_big], big_lam, pools.TOP_K)  # 硬性或軟性一對一配對
    else:  # 其他配對模式
        idx = assign_soft(F, doc["soft_lam"], pools.TOP_K) if (applied and assign == "soft") else assign_topk(F, pools.TOP_K) if applied else base  # 全部題目一起配對，或維持逐題結果
    pri = None  # 1 局題先驗的報告（未啟用時為 None）
    if prior3 is not None:  # METHOD_NOTES 2.123（探針 S3）：1 局題的擁有者（幾乎）從不是 3 局題的擁有者
        if assign != "big" or lend is not None:  # 先驗只定義在 big 配對之後，且不與出借並用
            raise ValueError("--one-prior3 needs --assign big and excludes --lend-margin")  # 參數組合不合法
        w3, temp = prior3  # 先驗機率 w 與溫度 T
        rows3, rows_one = np.flatnonzero(n == 3), np.flatnonzero(n == 1)  # 3 局題與 1 局題的列
        in3 = np.zeros(len(players), dtype=bool)  # 標記被配到 3 局題的候選人
        in3[idx[rows3, 0]] = True  # 3 局題被配到的候選人
        lp = np.where(in3, np.log(w3 / in3.sum()), np.log((1.0 - w3) / (len(players) - in3.sum())))  # 每位候選人的對數先驗：3 局題的候選人共分 w
        new = topk(F[rows_one] + temp * lp[None, :], pools.TOP_K)  # 1 局題加上 T 倍對數先驗後重排
        pri = {"w": w3, "T": temp, "candidates_of_3_game_questions": int(in3.sum()), "one_game_top1_in_them_before": int(in3[idx[rows_one, 0]].sum()),  # 先驗報告：設定與套用前的落點
               "one_game_top1_in_them_after": int(in3[new[:, 0]].sum()), "one_game_top1_changed": int((new[:, 0] != idx[rows_one, 0]).sum()),  # 套用後的落點與 Top-1 改變的題數
               "one_game_top5_changed": int((new != idx[rows_one]).any(axis=1).sum())}  # 前 5 名有變的題數
        idx[rows_one] = new  # 寫回 1 局題的新排序
    lent = None  # 出借棋局的報告（未啟用時為 None）
    if lend is not None:  # METHOD_NOTES 2.123：有把握配對的多局題把棋局借給其 Top-1 候選人；1 局題重新計分
        if assign != "big":  # 出借需要 big 配對的結果
            raise ValueError("--lend-margin needs --assign big")  # 參數組合不合法
        rows_big, rows_one = np.flatnonzero(n > 1), np.flatnonzero(n == 1)  # 多局題與 1 局題的列
        top2 = np.sort(F[rows_big], axis=1)[:, -2:]  # 多局題的第二高與最高分
        ok = (top2[:, 1] - top2[:, 0]) >= lend  # 領先幅度達門檻的多局題
        aug = {p: list(g) for p, g in cands.items()}  # 複製候選人的參考棋清單
        for r in rows_big[ok]:  # 逐一處理有把握的多局題
            aug[players[idx[r, 0]]] = aug[players[idx[r, 0]]] + list(q_use[usable[r]])  # 題目棋併入其 Top-1 候選人的參考棋
        S2 = v2.components(aug, [q_use[usable[r]] for r in rows_one], moves, sgfs, doc["cfg"], store, emb, which=doc["components"], seq=seq)  # 用擴充後的參考棋重算 1 局題的成分
        new = topk(fus.scores([S2[c] for c in doc["components"]]), pools.TOP_K)  # 1 局題的新前 5 名
        lent = {"margin": lend, "questions_lent": int(ok.sum()), "one_game_top1_changed": int((new[:, 0] != idx[rows_one, 0]).sum()),  # 出借報告：門檻、借出題數、Top-1 改變數
                "one_game_top5_changed": int((new != idx[rows_one]).any(axis=1).sum())}  # 前 5 名有變的題數
        idx[rows_one] = new  # 寫回 1 局題的新排序
    rows, j = [], 0  # 輸出列與可用題目的游標
    for i, qid in enumerate(qids):  # 依官方題目順序輸出
        if i in set(usable):  # 有可用棋局的題目
            rows.append((qid, [players[c] for c in idx[j]]))  # 候選人索引換成 id
            j += 1  # 前進到下一個可用題目
        else:  # 沒有可用棋局的題目
            rows.append((qid, players[:pools.TOP_K]))  # 退回依 id 排序的前五位候選人
    top = np.sort(F, axis=1)  # 每題分數由小到大，用來算第一名的領先幅度
    rep = {"candidates": str(cand_path), "questions": str(q_path), "n_candidates": len(players), "n_questions": len(qids), "left_out_games": bad,  # 報告：輸入檔、規模與排除的棋局
           "fallback_questions": [qids[i] for i in range(len(qids)) if i not in set(usable)], "games_by_colour": pd.Series(list(colour.values())).value_counts().to_dict(),  # 退回的題目與執色分布
           "assign": {"mode": assign, "applied": applied, "big_lam": big_lam, "big_tau": big_tau, "n_top1_changed": int((idx[:, 0] != base[:, 0]).sum()),  # 配對設定與 Top-1 改變的題數
                      "top1_changed_by_n": {str(m): int((idx[:, 0] != base[:, 0])[n == m].sum()) for m in sorted(set(n.tolist()))}, "top1_collisions": col},  # 依局數的改變數與碰撞統計
           "median_top1_margin": float(np.median(top[:, -1] - top[:, -2])), "margin_by_n": {str(m): float(np.median((top[:, -1] - top[:, -2])[n == m])) for m in sorted(set(n.tolist()))},  # 第一名領先幅度的中位數（整體與依局數）
           "distinct_top1": int(len(set(base[:, 0].tolist()))), "lend": lent, "prior3": pri}  # 配對前的不同 Top-1 數，以及出借與 1 局題先驗的報告
    if pri is not None:  # 有套用 1 局題先驗時
        log(f"  prior on the 1-game questions (w {w3:g}, T {temp:g}): {pri['candidates_of_3_game_questions']} candidates hold a 3-game question; 1-game Top-1 among them "  # 回報先驗的效果
            f"{pri['one_game_top1_in_them_before']} -> {pri['one_game_top1_in_them_after']}; 1-game questions with a new Top-1 {pri['one_game_top1_changed']}, with a new Top-5 {pri['one_game_top5_changed']}", flush=True)  # 套用前後的落點與改變題數
    if lent is not None:  # 有出借棋局時
        log(f"  lending (margin >= {lend:g}): {lent['questions_lent']} multi-game questions lent their games; 1-game questions with a new Top-1 {lent['one_game_top1_changed']}, "  # 回報出借的效果
            f"with a new Top-5 {lent['one_game_top5_changed']}", flush=True)  # 前 5 名有變的題數
    log(f"{Path(q_path).name}: {len(players)} candidates, {len(qids)} questions ({len(rep['fallback_questions'])} fallback); distinct Top-1 {rep['distinct_top1']}; "  # 題池摘要：規模、退回題數與不同 Top-1 數
        f"confident-half distinct {col['confident_distinct']:.3f}; assignment {'applied' if applied else 'not applied'} (changed {rep['assign']['n_top1_changed']}); "  # 有把握一半的不同比例與配對結果
        f"median margin by n {({k: round(v, 2) for k, v in rep['margin_by_n'].items()})}", flush=True)  # 依局數的領先幅度中位數
    return rows, rep, set(players)  # 回傳每題前 5 名、報告與候選人集合


def main(argv=None):  # 命令列進入點：預測兩個新增池後與原題池的列合併
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明文字取模組 docstring 的第一行
    ap.add_argument("--tests-dir", required=True)  # 官方題目、候選與範例檔的資料夾
    ap.add_argument("--fusion", required=True, help="experiments/pid_v2_sim.py --fusion-out")  # 融合文件（成分、權重、向量檔）
    ap.add_argument("--trunk-store", nargs="*", default=[], help="pid_trunk_extract.py --official2 output (needed when the fusion names emb)")  # 每局主幹列所在的資料夾
    ap.add_argument("--old-submission", required=True, help="800 rows for the original pools (predict_pid.py on the 10-30 files)")  # 原題池 800 列的來源
    ap.add_argument("--assign", choices=["off", "auto", "on", "soft", "big"], default="off",  # 配對模式；最終流程用 big
                    help="soft = aicup_pid.assign.assign_soft with the document's soft_lam (off when it has none); big = one-to-one over the questions of two or more games only (2.37)")  # 各模式的說明
    ap.add_argument("--big-lam", type=float, default=None, help="--assign big: soft assignment with this penalty instead of the hard one")  # 軟性配對的懲罰 λ（最終為 2）
    ap.add_argument("--big-tau", type=float, default=None, help="--assign big: Sinkhorn prices with this temperature (aicup_pid.assign.assign_sinkhorn; METHOD_NOTES 2.122); with --big-lam the Top-1 stays the soft assignment's")  # Sinkhorn 溫度 τ（最終為 0.5）
    ap.add_argument("--one-prior3", type=float, default=None, help="--assign big: prior of the 1-game questions (METHOD_NOTES 2.123, probe S3): the candidates assigned to the 3-game questions "  # 1 局題先驗的 w（最終為 0.05）
                    "share this probability of owning a 1-game question (independent draws would give them 1/6; estimated 0.014 [0.001, 0.062]); default: off")  # 說明文字的後半段
    ap.add_argument("--prior-T", type=float, default=1.6, help="--one-prior3: temperature of the fused scores (fitted on the pseudo-pools' 1-game questions: 1.64 / 1.56)")  # 先驗的溫度 T
    ap.add_argument("--lend-margin", type=float, default=None, help="--assign big: every multi-game question whose Top-1 leads its second choice by at least this much (fused score) lends its games "  # 出借棋局的領先門檻（未採用，預設關閉）
                    "to its assigned candidate, and the 1-game questions are re-scored against the enlarged reference sets (METHOD_NOTES 2.123); default: off")  # 說明文字的後半段
    ap.add_argument("--workers", type=int, default=12)  # 解析棋譜的行程數
    ap.add_argument("--out", required=True)  # 輸出的提交檔路徑
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 記下開始時間，報告用來算執行秒數
    T = Path(a.tests_dir)  # 官方檔所在的資料夾
    doc = json.loads(Path(a.fusion).read_text())  # 讀融合文件
    emb = store = None  # 預設不用嵌入與主幹列
    if "emb" in doc["components"]:  # 融合含每局嵌入成分時
        import torch  # 只在需要時載入 PyTorch
        names = doc["emb_model"] if isinstance(doc["emb_model"], list) else [doc["emb_model"]]  # 一個或多個嵌入權重檔
        embs = [PlayerEmbedding.from_state(torch.load(str(ROOT / m), map_location="cpu", weights_only=False)) for m in names]  # 在 CPU 上載入每個嵌入模型
        emb = embs[0] if len(embs) == 1 else embs  # 單一模型或模型清單
        wide = getattr(embs[0], "d_in", 256) == 640  # emb3：含各階段平均的列（pid_trunk_extract.py 帶有 `phase` 的分段檔）
        store = v2.load_store_phase(*a.trunk_store) if wide else load_store(*a.trunk_store)  # 依模型輸入維度讀 640 或 256 維的主幹列
        print(f"embedding {names} ({embs[0].players_trained} training players, {640 if wide else 256}-dim rows); {len(store)} trunk rows", flush=True)  # 回報嵌入模型與主幹列數
    seq = None  # 預設沒有序列網路向量
    if "seq" in doc["components"]:  # 融合含序列網路成分時
        paths = doc["seq_official"] if isinstance(doc["seq_official"], list) else [doc["seq_official"]]  # 各網路的向量檔清單
        seqs = []  # 每個網路一個 {棋局鍵: 向量} 字典
        for path in paths:  # 每個網路一個項目；一個項目可列出多個檔案（每個題池一個：對題池調適後的向量）
            one = {}  # 這個網路的向量
            for f in (path if isinstance(path, list) else [path]):  # 逐一讀這個網路的向量檔
                z = np.load(f)  # 載入向量檔（key：棋局鍵，emb：向量）
                one.update(zip(z["key"].astype(str), z["emb"].astype(np.float32)))  # 以棋局鍵對應向量
            seqs.append(one)  # 收進清單
        seq = seqs[0] if len(seqs) == 1 else seqs  # 單一網路或網路清單
        print(f"sequence identity vectors: {len(paths)} network(s), {len(seqs[0])} games", flush=True)  # 回報網路數與局數
    new, reports, cand_ids = {}, [], {}  # 新增池的預測、題池報告與各題池的候選人 id
    for tag, prefix in (("public", "pub2_"), ("private", "priv2_")):  # 兩個新增池各自預測
        rows, rep, players = predict_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv", doc, emb, store, a.assign, a.workers, seq=seq, big_lam=a.big_lam, big_tau=a.big_tau, lend=a.lend_margin,  # 傳入檔案、融合文件與配對設定
                                          prior3=None if a.one_prior3 is None else (a.one_prior3, a.prior_T))  # 1 局題先驗的 (w, T)
        new.update({q: top for q, top in rows})  # 記下每題前 5 名
        reports.append(rep)  # 收集題池報告
        cand_ids[prefix] = players  # 記下這個題池的候選人
    old = pd.read_csv(a.old_submission, dtype=str)  # 讀原題池的既有提交檔
    sample = pd.read_csv(T / "sample_submission_player_v2.csv", dtype=str)  # 官方範例提交檔，決定列與順序
    old_rows = {r.question_id: [getattr(r, f"top_{i}") for i in range(1, 6)] for r in old.itertuples()}  # 既有提交檔每題的前 5 名
    for tag, prefix in (("public", "pub_"), ("private", "priv_")):  # 讀原題池的候選人 id
        cand_ids[prefix] = set(pd.read_csv(T / f"player_identification_candidates_{tag}.csv", dtype=str, usecols=["player_id"])["player_id"])  # 只讀 player_id 欄
    out_rows, problems = [], []  # 輸出列與檢查出的問題
    for qid in sample.question_id:  # 依範例檔的順序組出每一列
        top = new.get(qid) or old_rows.get(qid)  # 新增池用新預測，原題池用既有的列
        if top is None:  # 兩邊都沒有這一題
            problems.append(f"no prediction for {qid}")  # 記錄缺漏
            continue  # 換下一題
        prefix = qid[:qid.index("_") + 1]  # 題目 id 的題池前綴
        if len(set(top)) != 5 or any(p not in cand_ids[prefix] for p in top):  # 前 5 名須互不相同且都在自己的題池內
            problems.append(f"{qid}: Top-5 not five distinct candidates of its own pool")  # 記錄格式問題
        out_rows.append([qid, *top])  # 加入輸出列
    sub = pd.DataFrame(out_rows, columns=list(sample.columns))  # 依範例檔的欄位組成提交表
    if len(sub) != len(sample) or list(sub.question_id) != list(sample.question_id) or sub.isna().any().any():  # 列數、順序或缺值與範例不符
        problems.append("rows / order differ from the sample submission")  # 記錄問題
    counts = sub.question_id.str.extract(r"^([a-z0-9]+)_")[0].value_counts().to_dict()  # 各題池前綴的列數
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    report = {"submission": str(a.out), "rows": len(sub), "rows_by_prefix": counts, "problems": problems, "fusion": doc, "old_submission": str(a.old_submission), "pools": reports,  # 檢查報告的內容
              "seconds": time.time() - t0}  # 執行秒數
    Path(a.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")  # 不論成敗都寫出報告
    if problems:  # 有任何問題就不寫提交檔
        print(f"NOT written: {len(problems)} problems, e.g. {problems[:3]}", file=sys.stderr)  # 回報問題
        return 1  # 以失敗結束
    sub.to_csv(a.out, index=False)  # 寫出 1,600 列的提交檔
    print(f"wrote {a.out}: {len(sub)} rows {counts}; format check against the sample passed ({time.time() - t0:.0f} s)")  # 回報完成與格式檢查通過
    return 0  # 成功結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
