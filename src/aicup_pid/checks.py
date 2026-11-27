"""The 11/04 official-file checks as pure functions (docs/records/specs/2026-11-03-pid-fork-plan-design.md section 3).

Every function takes in-memory inputs (hashes, parsed moves, style feature rows) and returns a JSON-serialisable dict:
duplicates against the training index and between files (checks 1-2), length quantiles (4), legality (5), the opening-
distribution shift with a split-half null (6), the within-library self-identification (7), Top-1 collisions of the main
line (8), num_games counts (9) and record properties (10).
"""
import math  # 以 2 為底的對數（JS 散度）
from collections import Counter  # 計數

import numpy as np  # 陣列運算

from aicup.sgf import handicap, properties  # SGF 的讓子數與屬性讀取工具
from aicup.submission import check_game  # 檢查棋局能否重播
from aicup_pid import pools  # 題池與官方分數工具
from aicup_pid.components import COMPONENTS, fit_from_parsed, score_questions  # 主線成分的擬合與計分
from aicup_pid.fusion import Fusion, topk  # 融合與取前 k 名
from aicup_pid.opening import canonical  # 開局的標準方向

QUANTILES = {"p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}  # 要回報的分位數
ONE_TO_ONE_FRACTION = 0.875  # 400 個 Top-1 中有 350 位不同候選人
SHIFT_FACTOR = 1.5  # 開局偏移的標記倍數（相對於虛無分布的最大值）
MAX_EXAMPLES = 20  # 最多列出的錯誤例子數
MAX_PLAYERS_LISTED = 50  # 報告中最多列出的玩家數


# ---------------------------------------------------------------- 重複棋譜（檢查 1-2）

def duplicates_against_index(hashes, index):  # 找出與訓練集重複的棋局
    """{key: [{player_id, rank, game_key}]} for the keys whose hash is in the training index."""
    out = {}  # 棋局鍵 -> 重複的訓練棋局清單
    for k, h in hashes.items():  # 逐局查詢
        if h is None:  # 沒有落子的棋譜沒有雜湊
            continue  # 略過
        rows = index.lookup(h)  # 訓練索引中雜湊相同的列
        if rows:  # 有重複
            m = index.meta.iloc[rows]  # 取出那些訓練棋局的資訊
            out[k] = [{"player_id": r.player_id, "rank": r.rank, "game_key": r.game_key} for r in m.itertuples()]  # 記下重複的訓練玩家與棋局
    return out  # 回傳重複清單


def summarise_overlap(dups, owner):  # 彙整每位玩家的重複情形
    """Per owning player (owner: {key: player_id}) how many games hit which training players."""
    by_player = {}  # 每位玩家命中的訓練玩家計數
    for k, hits in dups.items():  # 逐局處理重複結果
        c = by_player.setdefault(owner[k], Counter())  # 取得該局擁有者的計數器
        for h in hits:  # 每個命中的訓練棋局
            c[h["player_id"]] += 1  # 累計命中的訓練玩家
    single = sum(1 for c in by_player.values() if len(c) == 1)  # 只對應到一位訓練玩家的人數
    listed = sorted(by_player)[:MAX_PLAYERS_LISTED]  # 報告中列出的玩家
    return {"n_games_checked": len(owner), "n_dup_games": len(dups), "n_players_with_dups": len(by_player),  # 檢查局數、重複局數、有重複的玩家數
            "players_single_training_id": single, "players_multiple_training_ids": len(by_player) - single,  # 對應單一或多位訓練玩家的人數
            "same_id_players": sum(1 for p, c in by_player.items() if p in c),  # id 與命中的訓練玩家相同的人數
            "by_player": {p: dict(by_player[p]) for p in listed}, "players_not_listed": max(len(by_player) - MAX_PLAYERS_LISTED, 0)}  # 列出的明細與未列出的人數


def duplicates_between(hashes_a, hashes_b):  # 找兩個檔案之間的重複棋局
    """[(key_a, key_b)] sharing a hash (None never matches), in hashes_a order."""
    by_hash = {}  # 雜湊 -> hashes_b 的鍵
    for k, h in hashes_b.items():  # 建立 hashes_b 的反查表
        if h is not None:  # 略過沒有雜湊的棋局
            by_hash.setdefault(h, []).append(k)  # 同雜湊的鍵放在一起
    return [(ka, kb) for ka, h in hashes_a.items() if h is not None for kb in by_hash.get(h, ())]  # 依 hashes_a 順序列出相符的棋局對


def duplicates_within(hashes):  # 找同一個檔案內的重複棋局
    """[(key_i, key_j)] pairs of distinct keys sharing a hash, insertion order."""
    groups = {}  # 雜湊 -> 鍵清單
    for k, h in hashes.items():  # 依雜湊分組
        if h is not None:  # 略過沒有雜湊的棋局
            groups.setdefault(h, []).append(k)  # 加入同一組
    return [(ks[i], ks[j]) for ks in groups.values() if len(ks) > 1 for i in range(len(ks)) for j in range(i + 1, len(ks))]  # 每組內兩兩配對


# ---------------------------------------------------------------- 長度、合法性（檢查 4-5）

def _quantiles(x):  # 分位數與平均
    x = np.asarray(x, dtype=float)  # 輸入可為清單，統一成浮點陣列
    if len(x) == 0:  # 沒有資料
        return None  # 無法計算
    out = {k: float(np.percentile(x, q)) for k, q in QUANTILES.items()}  # 各分位數
    out["mean"] = float(x.mean())  # 加上平均
    return out  # 回傳字典


def length_quantiles(n_moves, n_black):  # 棋局長度的分位數（檢查 4）
    n_moves, n_black = np.asarray(n_moves), np.asarray(n_black)  # 總手數與黑棋手數
    return {"n": int(len(n_moves)), "moves": _quantiles(n_moves), "black_moves": _quantiles(n_black)}  # 局數與兩種長度的分位數


def legality(sgfs, colour="B"):  # 棋譜合法性（檢查 5）
    """check_game over every record: counts, the first MAX_EXAMPLES errors, HD counts and the usable keys in order."""
    n_bad, errors, hd, usable = 0, [], Counter(), []  # 不合法局數、錯誤例子、HD 計數、可用鍵
    for k, sgf in sgfs.items():  # 逐局檢查
        ok, info = check_game(sgf, colour)  # 依指定的執色重播
        hd[str(info["handicap"]) if info["handicap"] is not None else "none"] += 1  # 累計 HD 值（沒有 HD 記為 none）
        if ok:  # 可以重播
            usable.append(k)  # 記為可用
        else:  # 重播失敗
            n_bad += 1  # 累計不合法局數
            if len(errors) < MAX_EXAMPLES:  # 例子數尚未達上限
                errors.append({"key": k, "error": info["error"]})  # 記下錯誤原因
    return {"n": len(sgfs), "n_bad": n_bad, "errors": errors, "handicap_counts": dict(hd), "usable_keys": usable}  # 回傳統計結果


# ---------------------------------------------------------------- 開局分布偏移（檢查 6）

def black_prefix(moves, k):  # 黑棋開局前 k 手的座標序列
    """Canonical coordinates of Black's first k moves among the first 2k plies (the opening model's token path)."""
    cs, _ = canonical(moves, 2 * k)  # 以前 2k 手決定標準方向
    return tuple(cs[i] for i, (c, *_) in enumerate(moves[:2 * k]) if c == "B")[:k]  # 取黑棋的座標，最多 k 手


def js_divergence(count_p, count_q):  # 兩個分布的 JS 散度
    """Jensen-Shannon divergence in bits (0-1) between two count dicts."""
    np_, nq = sum(count_p.values()), sum(count_q.values())  # 兩邊的總數
    if np_ == 0 or nq == 0:  # 任一邊沒有資料
        return float("nan")  # 無法計算
    total = 0.0  # 累加散度
    for key in set(count_p) | set(count_q):  # 走過兩邊出現過的每個項目
        p, q = count_p.get(key, 0) / np_, count_q.get(key, 0) / nq  # 兩邊的機率
        m = 0.5 * (p + q)  # 混合分布
        if p > 0:  # 0 log 0 視為 0
            total += 0.5 * p * math.log2(p / m)  # P 對 M 的 KL 部分
        if q > 0:  # 0 log 0 視為 0
            total += 0.5 * q * math.log2(q / m)  # Q 對 M 的 KL 部分
    return float(total)  # 回傳散度


def opening_shift(q_moves, c_moves, k, n_null=20, seed=0, factor=SHIFT_FACTOR):  # 題目與候選人的開局分布是否偏移
    """JS divergence between the query games' and the candidate games' Black-prefix distributions, against a null of
    random candidate splits of the same size; flag = observed > factor * max(null)."""
    tq = [black_prefix(m, k) for m in q_moves]  # 題目棋局的黑棋前綴
    tc = [black_prefix(m, k) for m in c_moves]  # 候選人棋局的黑棋前綴
    out = {"k": int(k), "n_query": len(tq), "n_candidate": len(tc), "observed": None, "null": [], "null_mean": None, "null_max": None,  # 先填入預設值
           "ratio_to_null_max": None, "flag": None}  # 資料不足時維持 None
    if not tq or len(tc) < 2:  # 資料不足以比較
        return out  # 回傳預設值
    out["observed"] = js_divergence(Counter(tq), Counter(tc))  # 觀察到的散度
    nq = min(len(tq), len(tc) // 2)  # 虛無切分的一組取與題目相同的局數，但不超過候選棋局的一半
    rng = np.random.RandomState(seed)  # 固定種子的亂數
    for _ in range(n_null):  # 產生虛無分布
        perm = rng.permutation(len(tc))  # 隨機打散候選棋局
        out["null"].append(js_divergence(Counter(tc[i] for i in perm[:nq]), Counter(tc[i] for i in perm[nq:])))  # 候選棋局隨機分成 nq 局與其餘兩組，兩組之間的散度
    out["null_mean"], out["null_max"] = float(np.mean(out["null"])), float(np.max(out["null"]))  # 虛無分布的平均與最大值
    out["ratio_to_null_max"] = float(out["observed"] / out["null_max"]) if out["null_max"] > 0 else None  # 觀察值相對虛無最大值的倍數
    out["flag"] = bool(out["observed"] > factor * out["null_max"])  # 超過倍數門檻就標記為偏移
    return out  # 回傳結果


# ---------------------------------------------------------------- 候選庫內的自我辨識（檢查 7）

def self_identification(by_player, moves, feats, cfg, weights, seed=0, model_games=50, sizes=pools.SIZES):  # 用候選人自己的棋局模擬辨識（檢查 7）
    """Each candidate's games: model_games fit the models, the rest are cut into questions of `sizes` (ascending, while
    games remain); candidates with fewer than model_games + min(sizes) games are skipped. Returns the official score of
    every component and the fusion (weights = pid_fusion.json), the fusion margin quantiles and how many candidates have
    at least one question whose Top-1 is someone else."""
    sizes = sorted(int(m) for m in sizes)  # 題目局數由小到大
    model_keys, q_rows, skipped = {}, [], 0  # 擬合用的棋局、題目列、略過的人數
    for p in sorted(by_player):  # 依 id 逐位候選人
        keys = sorted(by_player[p])  # 該候選人的棋局鍵
        if len(keys) < model_games + sizes[0]:  # 棋局不夠切出最小的一題
            skipped += 1  # 計入略過
            continue  # 換下一位
        keys = [keys[i] for i in pools._player_rng(seed, p).permutation(len(keys))]  # 以玩家 id 衍生的種子打散
        model_keys[p] = keys[:model_games]  # 前 model_games 局擬合模型
        rest = keys[model_games:]  # 其餘拿來出題
        for m in sizes:  # 每種局數出一題
            if len(rest) < m:  # 剩餘棋局不夠
                break  # 停止出題
            q_rows.append((p, m, rest[:m]))  # 記下題目（擁有者、局數、棋局）
            rest = rest[m:]  # 移除用掉的棋局
    out = {"n_candidates": len(model_keys), "n_skipped": skipped, "n_questions": len(q_rows), "model_games": int(model_games),  # 規模資訊
           "scores": None, "fusion_margin_quantiles": None, "candidates_misidentified": None}  # 結果欄位先填 None
    if not model_keys or not q_rows:  # 沒有候選人或沒有題目
        return out  # 回傳空結果
    opening, style = fit_from_parsed(model_keys, moves, feats, cfg)  # 擬合開局與風格模型
    S = score_questions(opening, style, [ks for _, _, ks in q_rows], moves, feats)  # 開局與風格兩個成分的分數矩陣
    fus = Fusion()  # 建立融合器
    fus.w = np.asarray(weights, dtype=float)  # 套用給定的融合權重
    S_f = fus.scores([S[c] for c in COMPONENTS])  # 融合分數
    truth = [p for p, _, _ in q_rows]  # 每題真正的擁有者
    n_games = np.array([m for _, m, _ in q_rows])  # 每題局數
    scores = {}  # 各成分與融合的官方分數
    for name, M in (*S.items(), ("fusion", S_f)):  # 逐一評分
        tops = [[opening.players[j] for j in row] for row in topk(M, pools.TOP_K)]  # 前 5 名換成玩家 id
        scores[name] = pools.summarise(tops, truth, n_games)  # 官方分數與輔助指標
    srt = np.sort(S_f, axis=1)  # 融合分數由小到大
    margins = srt[:, -1] - srt[:, -2] if S_f.shape[1] >= 2 else np.zeros(len(q_rows))  # 第一名領先第二名的幅度
    top1 = [opening.players[j] for j in np.argmax(S_f, axis=1)]  # 融合後的 Top-1
    out.update({"scores": scores, "fusion_margin_quantiles": _quantiles(margins),  # 填入分數與領先幅度的分位數
                "candidates_misidentified": len({p for p, t in zip(truth, top1) if p != t})})  # 至少有一題 Top-1 認錯的候選人數
    return out  # 回傳結果


# ---------------------------------------------------------------- Top-1 碰撞（檢查 8）、num_games（9）、棋譜屬性（10）

CONFIDENT_FRACTION, CONFIDENT_DISTINCT = 0.5, 0.95  # 規則 v2（METHOD_NOTES 2.27），11-10 依生日問題的碰撞率在跑模擬之前定下


def confident_distinct(S_f, frac=CONFIDENT_FRACTION):  # 有把握的題目中不同 Top-1 的比例
    """Share of distinct Top-1 candidates among the `frac` most confident questions (largest Top-1 margin). One question per
    candidate -> about 1 whatever the difficulty (confident picks are right, so they do not collide); questions drawn with
    replacement over C candidates -> at most the birthday rate C (1 - (1 - 1/C)^m) / m (0.79 for m = 200, C = 400)."""
    S = np.asarray(S_f, dtype=float)  # 輸入可為清單，統一成浮點陣列
    if len(S) < 2 or S.shape[1] < 2:  # 題目或候選人太少
        return 1.0  # 視為完全不碰撞
    top = np.sort(S, axis=1)  # 每題分數由小到大
    keep = np.argsort(-(top[:, -1] - top[:, -2]), kind="stable")[:int(np.ceil(frac * len(S)))]  # 領先幅度最大的 frac 比例題目
    return float(len(set(S[keep].argmax(axis=1).tolist())) / len(keep))  # 這些題目中不同 Top-1 的比例


def top1_collisions(S_f, n_games, fraction=ONE_TO_ONE_FRACTION):  # 判斷題池是否像一人一題
    """How many distinct candidates the questions' Top-1 picks name, overall and per size, against the random expectation
    C (1 - (1 - 1/C)^Q); one_to_one_likely when distinct >= fraction * Q; one_to_one_likely_v2 adds the confident-half clause
    (predict_pid.py --assign auto uses v2; the file-check flags keep the first rule)."""
    S = np.asarray(S_f, dtype=float)  # 輸入可為清單，統一成浮點陣列
    n_games = np.asarray(n_games)  # 每題局數
    Q, C = S.shape  # 題數與候選人數
    if Q == 0:  # 沒有題目
        return {"n_questions": 0, "n_candidates": int(C), "distinct_top1": 0, "expected_random": 0.0, "by_n": {}, "one_to_one_likely": None,  # 回傳空報告
                "confident_distinct": None, "one_to_one_likely_v2": None}  # 判斷欄位為 None
    t1 = np.argmax(S, axis=1)  # 每題的 Top-1
    distinct = len(set(t1.tolist()))  # 不同 Top-1 的數目
    return {"n_questions": int(Q), "n_candidates": int(C), "distinct_top1": int(distinct),  # 規模與不同 Top-1 數
            "expected_random": float(C * (1 - (1 - 1 / C) ** Q)) if C > 0 else 0.0,  # 每題隨機抽人時的期望不同數
            "by_n": {str(n): len(set(t1[n_games == n].tolist())) for n in sorted(set(n_games.tolist()))},  # 各局數的不同 Top-1 數
            "one_to_one_likely": bool(distinct >= fraction * Q), "confident_distinct": (conf := confident_distinct(S)),  # 第一條規則與有把握一半的統計量
            # v2：87.5% 規則，或題目不多於候選人且較有把握的一半中 >= 95% 的 Top-1 互不相同（困難的區塊）
            "one_to_one_likely_v2": bool(distinct >= fraction * Q or (Q <= C and conf >= CONFIDENT_DISTINCT))}  # 規則 v2 的判斷


def num_games_counts(questions):  # 各宣告局數的題數（檢查 9）
    c = Counter(int(q["declared_games"]) for q in questions)  # 計數每題宣告的局數
    return {str(n): int(c[n]) for n in sorted(c)}  # 依局數排序輸出


def record_properties(sgfs, game_ids):  # 棋譜屬性與 game_id 的檢查（檢查 10）
    """SGF property identifiers (games containing each), game_id examples, whether the ids are all numeric, sorted and unique."""
    props = Counter()  # 每個屬性出現的局數
    for sgf in sgfs.values():  # 逐局讀屬性
        props.update(set(properties(sgf)))  # 每局每個屬性只算一次
    ids = list(game_ids)  # game_id 清單
    numeric = bool(ids) and all(str(g).isdigit() for g in ids)  # 是否全為數字
    if not ids:  # 沒有 id
        sorted_flag = None  # 無法判斷排序
    elif numeric:  # 數字 id 依數值比較
        vals = [int(g) for g in ids]  # 轉成整數，排序依數值而非字串比較
        sorted_flag = bool(all(a <= b for a, b in zip(vals, vals[1:])))  # 是否非遞減
    else:  # 非數字 id 依字串比較
        sorted_flag = bool(all(str(a) <= str(b) for a, b in zip(ids, ids[1:])))  # 是否依字串非遞減
    return {"properties": dict(props), "game_id_examples": [str(g) for g in ids[:5]], "game_id_all_numeric": numeric, "game_id_sorted_in_file": sorted_flag,  # 屬性計數、例子、數字與排序旗標
            "game_id_unique": bool(len(set(map(str, ids))) == len(ids)), "n_game_ids": len(ids)}  # 是否唯一與 id 數
