#!/usr/bin/env python
"""Official player-identification files -> submission_player.csv (CPU components: opening model + style model + fusion).

For each (candidates, questions) pair -- public first, then private -- the candidates' games fit the opening and style
models of that pool, every question's games are scored against the 400 candidates, the components are fused with the
weights of --fusion (scripts/pid_cv.py --fusion-out, or pid_local_cv.py --fusion-out for the three-component fusion3; the
components may be a subset, e.g. opening alone as the fallback) and the
best five distinct candidates are written; --assign on|auto replaces the Top-1 by the one-to-one assignment (line B3). The submission is
validated (header, order, pool membership, distinct ids, --expect-rows) before exit; a report JSON goes next to it.
Exit codes: 2 input/format problems, 1 validation failure.

  python -I scripts/predict_pid.py --candidates tests/player_identification_candidates_public.csv tests/player_identification_candidates_private.csv \\
      --questions tests/player_identification_test_public.csv tests/player_identification_test_private.csv \\
      --fusion results/pid_fusion.json --out results/submission_player.csv --expect-rows 800 --workers 12
"""
import argparse  # 解析命令列參數
import json  # 讀取融合文件
import sys  # 設定 sys.path 與結束碼
import time  # 計時
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取 CSV

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
if str(ROOT / "src") not in sys.path:  # 尚未加入本儲存庫的 src 時
    sys.path.insert(0, str(ROOT / "src"))  # 讓 aicup_pid 可以匯入
import aicup_pid  # noqa: E402  匯入時設定 AI_CUP_2026-1 的路徑
TRAIN_DIR = aicup_pid.TRAIN_DIR  # 共用的資料位置（也會把 AI_CUP_2026-1 放上 sys.path）
from aicup.ladder import to_json  # noqa: E402  輸出 JSON 的格式化工具
from aicup.submission import check_game  # noqa: E402  判斷棋局能否使用（19 路、能合法重播、該方至少一手）
from aicup_pid import checks, pools  # noqa: E402  配對規則的統計量；讀寫官方檔與提交檔
from aicup_pid.assign import assign_blocks, assign_topk, assignment_report  # noqa: E402  一對一配對（整池或分區塊）
from aicup_pid.components import ALL_COMPONENTS, COMPONENTS, fit_hd, fit_local, fit_models, fit_opening_raw, fit_orient, fit_ref, parse_all, score_components, score_ref  # 各統計成分的擬合與計分
from aicup_pid.paths import FOLDS_CSV  # 玩家分折表（參考玩家成分用）
from aicup_pid.ref import reference_players  # 挑選參考玩家
from aicup_pid.trunkstyle import fit_trunk, load_store, score_trunk  # noqa: E402  主幹風格成分
from aicup_pid.fusion import Fusion, topk  # noqa: E402  融合與取前 k 名
from aicup_pid.provenance import pid_environment  # noqa: E402  在報告中記錄執行環境


def predict_pool(cand_path, q_path, fus, cfg, workers, comps=COMPONENTS, assign="off", log=print, ref_folds=(1, 2, 3, 4), trunk_dirs=(), emb_path=None, b3_dir=None, orig_keys=None):  # 預測一個題池的前五名
    """[(question_id, top5 ids)], report dict for one pool. orig_keys: the game tuples of an earlier release's questions; when given, the
    assignment rule and the assignment run separately on the questions found there (block "orig") and on the others ("added")."""
    t0 = time.time()  # 計時起點
    cands = pools.read_candidates(cand_path)  # 讀候選檔（候選棋局一律執黑，否則報錯）
    questions = pools.read_questions(q_path)  # 讀題目檔
    sgfs = {f"c:{i}:{r.game_id}": r.sgf for i, r in enumerate(cands.itertuples())}  # 以列號定位：game_id 不必唯一
    by_player = {}  # 候選人 → 其棋局的鍵
    for i, r in enumerate(cands.itertuples()):  # 逐列處理候選棋局
        by_player.setdefault(r.player_id, []).append(f"c:{i}:{r.game_id}")  # 歸到擁有者名下
    q_keys, dropped = [], []  # 每題可用棋局的鍵；被丟掉的棋局
    for q in questions:  # 逐題處理
        keys = []  # 這題可用的棋局
        for i, (sgf, colour) in enumerate(q["games"]):  # 題目中的每一局
            ok, info = check_game(sgf, colour or "B")  # 檢查這局能否使用；未標顏色時以黑方檢查
            if not ok:  # 不能用的局
                dropped.append({"question_id": q["question_id"], "game": i + 1, "error": info["error"]})  # 記下原因
                continue  # 換下一局
            key = f"q:{q['question_id']}:{i}"  # 題目棋局的鍵
            sgfs[key] = sgf  # 存棋譜
            keys.append(key)  # 收進這一題
        q_keys.append(keys)  # 這題的可用棋局
    log(f"{Path(cand_path).name}: {len(by_player)} candidates, {len(cands)} games; {Path(q_path).name}: {len(questions)} questions, "  # 回報候選人與題目的規模
        f"{sum(len(k) for k in q_keys)} games ({len(dropped)} dropped)", flush=True)  # 以及題目棋局與丟掉的局數
    opening, style, moves, feats = fit_models(by_player, sgfs, cfg, workers=workers)  # 擬合開局與風格模型（同時解析所有棋譜）
    local = fit_local(by_player, moves, cfg) if "local" in comps else None  # 局部棋形成分
    hd = fit_hd(by_player, sgfs, cfg) if "hd" in comps else None  # 讓子旗標成分
    orient = fit_orient(by_player, moves, cfg) if "orient" in comps else None  # 方向成分
    opening_raw = fit_opening_raw(by_player, moves, cfg) if "opening_raw" in comps else None  # 原始座標開局成分
    usable = [k for k in q_keys if k]  # 至少有一局可用的題目
    S_ref = None  # 參考玩家成分的分數
    if "ref" in comps:  # 以訓練玩家（文件指定的折）當匿名的參考基底
        t1 = time.time()  # 計時
        from aicup.scoring import load_games  # 讀訓練棋局
        folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 玩家分折表
        refs = reference_players(pools.black_games(aicup_pid.TRAIN_DIR), folds_df, ref_folds, min_black=cfg.get("ref_min_black", 30), cap=cfg.get("ref_cap", 40), seed=cfg.get("ref_seed", 0))  # 挑選參考玩家與其棋局
        ref_sgfs = {k: v[0] for k, v in load_games(sorted({k for ks in refs.values() for k in ks}), aicup_pid.TRAIN_DIR).items()}  # 參考玩家的棋譜
        ref_moves = parse_all(ref_sgfs, workers)  # 平行解析
        ref_model = fit_ref(refs, ref_moves, cfg)  # 擬合參考模型
        S_ref = score_ref(ref_model, by_player, usable, moves)  # 題目對候選人的相似度
        log(f"reference basis: {len(refs)} training players from folds {ref_folds}, {len(ref_sgfs)} games, {ref_model.logp.shape[1]} tokens in {time.time() - t1:.0f} s", flush=True)  # 回報
    S_trunk = None  # 主幹風格成分的分數
    if "trunk" in comps:  # 官方棋局的每局主幹列（scripts/pid_trunk_extract.py --official），鍵與這裡的 sgfs 相同
        t1 = time.time()  # 計時
        store = load_store(*trunk_dirs)  # 讀每局主幹列
        S_trunk = score_trunk(fit_trunk(by_player, store, cfg), usable, store)  # 擬合並計分
        log(f"trunk-style component from {len(store)} stored game rows in {time.time() - t1:.0f} s", flush=True)  # 回報
    S_emb = None  # 每局嵌入成分的分數
    if "emb" in comps:  # 支線 B8：在同樣的主幹列上學出的玩家嵌入（由 scripts/pid_embed_cv.py --final-model-out 訓練）
        t1 = time.time()  # 計時
        import torch  # 只有這個成分需要
        from aicup_pid.embed import PlayerEmbedding, score_embed  # 嵌入模型與計分
        if not emb_path:  # 融合文件沒給模型
            raise ValueError("the fusion names 'emb' but gives no emb_model_official")  # 當成輸入錯誤
        store = load_store(*trunk_dirs)  # 讀每局主幹列
        emb_model = PlayerEmbedding.from_state(torch.load(str(ROOT / emb_path), map_location="cpu", weights_only=False))  # 載入嵌入模型
        S_emb = score_embed(emb_model, by_player, usable, store)  # 題目對候選人計分
        log(f"embedding component ({emb_path}, {emb_model.players_trained} training players) in {time.time() - t1:.0f} s", flush=True)  # 回報
    S = score_components([c for c in comps if c != "b3"], opening, style, usable, moves, feats, local, hd, sgfs, orient, opening_raw, S_ref, S_trunk, S_emb)  # 各成分的分數矩陣
    b3_info = None  # B3 成分的資訊
    if "b3" in comps:  # 支線 B3（METHOD_NOTES 2.26）：由 experiments/pid_b3_official.py 擷取的 ct9 zq；只用於不配對的備案
        from aicup_pid.b3 import score_b3  # B3 的計分
        if not b3_dir:  # 沒給資料夾
            raise ValueError("the fusion names 'b3' but no --b3-dir was given")  # 當成輸入錯誤
        prefix = questions[0]["question_id"].split("_")[0]  # 由題號取出 pub 或 priv
        z = np.load(Path(b3_dir) / prefix / "b3.npz")  # 擷取下來的 zq
        cmap = pd.read_csv(Path(b3_dir) / prefix / "inputs" / "cands_map.csv", dtype=str, keep_default_na=False)  # 候選人與其棋局切段（每段至多 20 局，當成一題）的對照
        if set(cmap.player_id) != set(by_player):  # 擷取時的候選人不同
            raise ValueError(f"{b3_dir}/{prefix}: the captured candidates differ from {cand_path}")  # 停下
        chunks = cmap.groupby("player_id")["question_id"].apply(list).to_dict()  # 每位候選人的各段編號
        S["b3"], b3_info = score_b3(z["zq"], {q: i for i, q in enumerate(z["qids"])}, chunks, list(opening.players),  # B3 的分數矩陣
                                    [q["question_id"] for q, keys in zip(questions, q_keys) if keys])  # 只算有可用棋局的題目
        b3_info = {k: (len(v) if isinstance(v, list) else v) for k, v in b3_info.items()}  # 清單只記長度
        log(f"b3 component ({b3_dir}/{prefix}): {b3_info}", flush=True)  # 回報
    S_f = fus.scores([S[c] for c in comps]) if usable else np.zeros((0, len(opening.players)))  # 融合後的分數
    n_games = np.array([q["declared_games"] for q, keys in zip(questions, q_keys) if keys], dtype=int)  # 每題宣告的局數
    collisions = checks.top1_collisions(S_f, n_games)  # 第一名的重複程度（配對規則用）
    if orig_keys is None:  # 整池一起決定是否配對
        applied = assign == "on" or (assign == "auto" and bool(collisions["one_to_one_likely_v2"]))  # 強制配對，或 auto 時依規則第二版判斷
        idx = assign_topk(S_f, pools.TOP_K) if applied else topk(S_f, pools.TOP_K)  # 配對後或直接取前五名
        assign_rep = {"mode": assign, "applied": applied, "top1_collisions": collisions, **({k: v for k, v in assignment_report(S_f, idx).items() if k == "n_top1_changed"} if applied else {"n_top1_changed": 0})}  # 配對報告
    else:  # 11-11 版（METHOD_NOTES 2.27）：原有題與新增題是同一批候選人上的兩組題
        labels = np.array(["orig" if tuple(sgf for sgf, _c in q["games"]) in orig_keys else "added" for q, keys in zip(questions, q_keys) if keys])  # 以棋局內容判斷每題屬於 orig 或 added
        base, blocks, apply = topk(S_f, pools.TOP_K), {}, {}  # 不配對的前五名；各區塊的報告與決定
        for b in sorted(set(labels.tolist())):  # 每個區塊
            m = labels == b  # 該區塊的題目
            col = checks.top1_collisions(S_f[m], n_games[m])  # 區塊內的統計量
            apply[b] = assign == "on" or (assign == "auto" and bool(col["one_to_one_likely_v2"]))  # 區塊內是否配對
            blocks[b] = {"applied": apply[b], "top1_collisions": col}  # 記錄
        idx = assign_blocks(S_f, labels, apply, pools.TOP_K) if len(labels) else base  # 逐區塊配對
        for b in blocks:  # 每個區塊
            blocks[b]["n_top1_changed"] = int((idx[labels == b, 0] != base[labels == b, 0]).sum())  # 第一名被改的題數
        applied = any(apply.values())  # 是否有任一區塊配對
        assign_rep = {"mode": assign, "applied": applied, "top1_collisions": collisions, "n_top1_changed": int((idx[:, 0] != base[:, 0]).sum()) if len(labels) else 0, "blocks": blocks}  # 配對報告
        log("assignment by block: " + ", ".join(f"{b}: {'applied' if v['applied'] else 'not applied'} (distinct Top-1 {v['top1_collisions']['distinct_top1']}/{v['top1_collisions']['n_questions']}, "  # 每個區塊是否配對與統計量
                                                 f"changed {v['n_top1_changed']})" for b, v in blocks.items()), flush=True)  # 以及改動的題數
    rows, j, fallback = [], 0, []  # 提交列；可用題的索引；備案題
    for q, keys in zip(questions, q_keys):  # 依原順序逐題
        if keys:  # 有可用的棋局
            rows.append((q["question_id"], [opening.players[i] for i in idx[j]]))  # 寫出前五名的候選人編號
            j += 1  # 下一個可用題
        else:  # 沒有可用的棋局：取前五位候選人（記錄下來；這種題的分數靠運氣）
            rows.append((q["question_id"], opening.players[:pools.TOP_K]))  # 備案的前五名
            fallback.append(q["question_id"])  # 記錄
    margins = (np.sort(S_f, axis=1)[:, -1] - np.sort(S_f, axis=1)[:, -2]).tolist() if usable else []  # 第一名與第二名的分數差
    report = {"candidates": str(cand_path), "questions": str(q_path), "n_candidates": len(by_player), "n_candidate_games": int(len(cands)),  # 題池報告：檔案與規模
              "n_candidate_games_used": int(sum(len(k) for k in by_player.values())), "candidate_game_ids_unique": bool(cands["game_id"].is_unique),  # 候選棋局的使用情形
              "n_questions": len(questions), "dropped_games": dropped, "fallback_questions": fallback, "median_top1_margin": float(np.median(margins)) if margins else None,  # 題目、丟棄、備案與領先幅度
              "assign": assign_rep, "b3": b3_info,  # 配對與 B3 的資訊
              "seconds": time.time() - t0}  # 耗時
    return rows, report, set(by_player)  # 提交列、報告與候選人集合


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--candidates", nargs="+", required=True, help="candidate CSVs, public then private")  # 候選檔
    ap.add_argument("--questions", nargs="+", required=True, help="test CSVs in the same order")  # 題目檔
    ap.add_argument("--fusion", required=True, help="pid_cv.py --fusion-out JSON")  # 融合文件
    ap.add_argument("--out", required=True)  # 提交檔路徑
    ap.add_argument("--expect-rows", type=int, default=None)  # 預期的列數
    ap.add_argument("--workers", type=int, default=1)  # 平行行程數
    ap.add_argument("--trunk-store", nargs="*", default=None, help="trunk rows of these official files (default: the fusion document's trunk_store_official); a re-released set has its own store")  # 每局主幹列的資料夾
    ap.add_argument("--assign-blocks", nargs="*", default=None, help="test CSVs of an earlier release: the questions found there (same games) form block 'orig', the rest 'added'; "  # 分區塊配對用的舊題目檔
                    "the assignment rule and the assignment then run block by block")  # 說明文字的後半
    ap.add_argument("--b3-dir", default=None, help="experiments/pid_b3_official.py work folder (<dir>/pub, <dir>/priv) for a fusion that names 'b3'")  # B3 的工作資料夾
    ap.add_argument("--assign", choices=["off", "auto", "on"], default="off",  # 一對一配對的模式
                    help="line B3: one-to-one assignment of the Top-1 (auto = only when the Top-1 picks name >= 87.5%% distinct candidates, or >= 95%% "  # 說明文字
                         "distinct among the more confident half with no more questions than candidates: aicup_pid.checks.top1_collisions)")  # 說明文字的後半
    a = ap.parse_args(argv)  # 解析參數
    if len(a.candidates) != len(a.questions):  # 候選檔與題目檔必須成對
        ap.error("--candidates and --questions must pair up")  # 參數錯誤
    try:  # 輸入或格式問題以結束碼 2 回報
        doc = json.loads(Path(a.fusion).read_text())  # 讀融合文件
        comps = list(doc["components"])  # 成分清單
        if not comps or len(set(comps)) != len(comps) or any(c not in (*ALL_COMPONENTS, "b3") for c in comps):  # 成分必須非空、不重複且都是已知成分
            raise ValueError(f"{a.fusion}: components {comps} must be a non-empty subset of {[*ALL_COMPONENTS, 'b3']}")  # 文件有誤
        if len(doc["weights"]) != len(comps):  # 權重數要與成分數相同
            raise ValueError(f"{a.fusion}: {len(doc['weights'])} weights for {len(comps)} components")  # 文件有誤
        fus = Fusion()  # 融合物件
        fus.w = np.asarray(doc["weights"], dtype=float)  # 直接套用文件的權重
        rows, reports, pool_players = [], [], {}  # 全部提交列、各題池報告、各題池的候選人
        orig_keys = None  # 較早版本題目的棋局組合
        if a.assign_blocks:  # 要分區塊配對
            orig_keys = {tuple(sgf for sgf, _c in q["games"]) for f in a.assign_blocks for q in pools.read_questions(f)}  # 以棋局內容辨認原有題
        for cand_path, q_path in zip(a.candidates, a.questions):  # 依序處理各題池（Public 先）
            r, rep, players = predict_pool(cand_path, q_path, fus, doc.get("cfg", {}), a.workers, comps=comps, assign=a.assign, ref_folds=tuple(doc.get("ref_folds_official", (1, 2, 3, 4))),  # 預測一個題池
                                           trunk_dirs=tuple(a.trunk_store) if a.trunk_store else (str(ROOT / doc.get("trunk_store_official", "data/trunk/official")),),  # 每局主幹列的資料夾
                                           emb_path=doc.get("emb_model_official"), b3_dir=a.b3_dir, orig_keys=orig_keys)  # 嵌入模型、B3 與分區塊的設定
            prefix = next((p for p in pools.PREFIXES if r and r[0][0].startswith(p)), None)  # 由題號判斷是 pub_ 或 priv_
            if prefix is not None:  # 判斷得出時
                pool_players[prefix] = players  # 記下該題池的候選人，供檢查用
            rows += r  # 接上這個題池的列
            reports.append(rep)  # 收集報告
    except (FileNotFoundError, ValueError, KeyError) as e:  # 輸入或格式問題
        print(e, file=sys.stderr)  # 印出原因
        return 2  # 結束碼 2
    pools.write_submission(rows, a.out)  # 寫出提交檔
    problems = pools.validate_submission(a.out, [q for q, _ in rows], pool_players, a.expect_rows)  # 檢查提交檔
    report = {"submission": str(a.out), "rows": len(rows), "pools": reports, "fusion": doc, "problems": problems, **pid_environment("cpu")}  # 報告內容（含融合文件與執行環境）
    Path(a.out).with_suffix(".report.json").write_text(to_json(report, indent=1) + "\n")  # 報告寫在提交檔旁
    if problems:  # 檢查不通過
        print("\n".join(problems), file=sys.stderr)  # 列出問題
        return 1  # 結束碼 1
    print(f"wrote {a.out}: {len(rows)} rows, {sum(len(r['fallback_questions']) for r in reports)} fallback questions; assignment "  # 回報列數與備案題數
          + ", ".join(f"{Path(r['questions']).name}: {'applied' if r['assign']['applied'] else 'not applied'} (distinct Top-1 {r['assign']['top1_collisions']['distinct_top1']}/{r['assign']['top1_collisions']['n_questions']}, changed {r['assign']['n_top1_changed']})" for r in reports))  # 以及各題池的配對情形
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
