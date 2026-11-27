"""Candidate pools, questions, the official score and the submission file (player identification).

A pool is 400 candidate players with 100 Black games each; a question is 5 / 10 / 15 / 20 Black games of one of them
(docs/TASK.md). make_pool simulates pools from the training set (docs/records/ARCHITECTURE.md section 3): players of the given
folds with at least 100 + 5 Black games, each one's Black games shuffled with a seed derived from the player id, the
first 100 as the candidate's games and the rest cut into one question per size in ascending order. read_candidates /
read_questions read the official files; question_score is e^-(r-1) for the true player at Top-5 rank r (0 otherwise);
write_submission / validate_submission produce and check submission_player.csv (question_id, top_1..top_5; 800 rows,
five distinct ids from the question's own pool, public before private).
"""
import hashlib  # 由玩家 id 衍生亂數種子
import math  # 指數函數（官方分數）
from collections import Counter  # 找出重複或缺少的題目 id
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列與亂數
import pandas as pd  # 讀寫 CSV

from aicup.metric import LEVELS  # 10 個棋力等級
from aicup.submission import read_test  # 讀官方題目檔
from aicup_pid.paths import TRAIN_DIR  # 訓練 CSV 的預設位置

SIZES = (5, 10, 15, 20)  # 原題池的題目局數
N_CANDIDATES, CAND_GAMES, TOP_K = 400, 100, 5  # 候選人數、每位候選人的局數、輸出名次數
PREFIXES = ("pub_", "priv_")  # 原題池的題目 id 前綴
COLUMNS = ["question_id", *[f"top_{i}" for i in range(1, TOP_K + 1)]]  # 提交檔的欄位


def black_games(train_dir=TRAIN_DIR):  # 列出訓練集的所有黑棋棋局
    """(player_id, rank, game_key) of every Black game of the training set, game_key 'RANK:game_id' as aicup.scoring."""
    frames = []  # 各等級的資料表
    for r in LEVELS:  # 逐等級讀檔
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"], dtype=str)  # 只讀需要的欄位
        df = df[df["color"] == "B"].copy()  # 只留執黑的棋局
        df["game_key"] = df["rank"] + ":" + df["game_id"]  # 組出棋局鍵
        frames.append(df[["player_id", "rank", "game_key"]])  # 保留三個欄位
    return pd.concat(frames, ignore_index=True)  # 合併所有等級


def eligible_players(games, folds_df, folds, min_black=CAND_GAMES + min(SIZES)):  # 可以入選題池的玩家
    """Players of `folds` with >= min_black Black games in `games`: (player_id, rank, n_black), sorted by rank and id."""
    folds = [int(f) for f in folds]  # 折編號轉成整數
    pl = folds_df[folds_df["fold"].isin(folds)][["player_id", "rank"]].astype({"player_id": str})  # 指定折中的玩家
    counts = games.groupby("player_id").size().rename("n_black")  # 每位玩家的黑棋局數
    out = pl.merge(counts, left_on="player_id", right_index=True, how="inner")  # 併入局數
    return out[out["n_black"] >= min_black].sort_values(["rank", "player_id"], kind="stable").reset_index(drop=True)  # 局數足夠者，依等級與 id 排序


def _player_rng(seed, player_id):  # 每位玩家固定的亂數產生器
    return np.random.RandomState(int(hashlib.sha256(f"{seed}:{player_id}".encode()).hexdigest()[:8], 16))  # 由種子與玩家 id 的雜湊決定


ORDERS = ("shuffle", "file")  # 切分棋局的兩種順序


def _player_keys(keys, order, seed, player_id):  # 依指定順序排列一位玩家的棋局
    """A player's game keys in the split order: 'shuffle' (seeded by the player id) or 'file' (the training files' order, i.e.
    sorted game_key: the rank prefix is constant per player and the ids are zero-padded)."""
    if order not in ORDERS:  # 順序不合法
        raise ValueError(f"order must be one of {ORDERS}, not {order!r}")  # 拒絕執行
    keys = sorted(keys)  # 先排序，讓結果不受輸入順序影響
    if order == "file":  # 依訓練檔的順序
        return keys  # 排序後即為檔案順序
    return [keys[i] for i in _player_rng(seed, player_id).permutation(len(keys))]  # 以玩家的種子打散


def make_pool(games, folds_df, folds, n_players=N_CANDIDATES, cand_games=CAND_GAMES, sizes=SIZES, seed=0, pool_id="sim", order="shuffle"):  # 從訓練集模擬一個題池
    """A simulated pool: (candidates: player_id, rank, game_key; questions: question_id, player_id, rank, n_games, game_keys;
    info). Up to n_players eligible players (RandomState(seed)); per player the Black games are shuffled with a seed from
    the player id, the first cand_games are the candidate's, the rest give one question per size (ascending) while games
    remain, disjoint. order 'file' takes the player's first cand_games games in the training files' order as the candidate's
    and the later ones as questions (a stress test for drift). ValueError when no player is eligible."""
    sizes = sorted(int(m) for m in sizes)  # 題目局數由小到大
    elig = eligible_players(games, folds_df, folds, cand_games + sizes[0])  # 至少能出最小一題的玩家
    if len(elig) == 0:  # 沒有合格玩家
        raise ValueError(f"folds {list(folds)}: no player has >= {cand_games + sizes[0]} Black games")  # 無法建立題池
    rng = np.random.RandomState(seed)  # 挑選玩家用的亂數
    chosen = elig.iloc[np.sort(rng.permutation(len(elig))[:n_players])]  # 隨機挑出玩家，保持原本的排序
    by_player = games.groupby("player_id")["game_key"].apply(sorted).to_dict()  # 每位玩家排序後的棋局鍵
    cand_rows, q_rows = [], []  # 候選棋局列與題目列
    for p in chosen.itertuples():  # 逐位入選玩家
        keys = _player_keys(by_player[p.player_id], order, seed, p.player_id)  # 依切分順序排列棋局
        cand_rows += [{"player_id": p.player_id, "rank": p.rank, "game_key": k} for k in keys[:cand_games]]  # 前 cand_games 局作為候選棋局
        rest = keys[cand_games:]  # 其餘拿來出題
        for m in sizes:  # 每種局數出一題
            if len(rest) < m:  # 剩餘棋局不夠
                break  # 停止出題
            q_rows.append({"question_id": f"{pool_id}_q_{len(q_rows) + 1:04d}", "player_id": p.player_id, "rank": p.rank,  # 題目 id 與擁有者
                           "n_games": m, "game_keys": rest[:m]})  # 局數與棋局
            rest = rest[m:]  # 移除用掉的棋局，題目之間互不重疊
    cands = pd.DataFrame(cand_rows, columns=["player_id", "rank", "game_key"])  # 候選棋局表
    qs = pd.DataFrame(q_rows, columns=["question_id", "player_id", "rank", "n_games", "game_keys"])  # 題目表
    info = {"folds": [int(f) for f in folds], "seed": int(seed), "eligible": int(len(elig)), "players": int(len(chosen)),  # 題池資訊：折、種子、合格與入選人數
            "cand_games": int(cand_games), "questions": int(len(qs)), "questions_by_n": {str(m): int((qs["n_games"] == m).sum()) for m in sizes},  # 候選局數、題數與各局數的題數
            "players_by_rank": {lv: int((chosen["rank"] == lv).sum()) for lv in LEVELS}, "order": order}  # 各等級人數與切分順序
    return cands, qs, info  # 回傳三者


def read_candidates(path):  # 讀原題池的官方候選檔
    """The official candidates CSV as (player_id, game_id, sgf, color); ValueError on missing columns or a non-Black game."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # 全部以字串讀入，空字串不轉成 NaN
    missing = [c for c in ("game_id", "player_id", "sgf_content", "color") if c not in df.columns]  # 缺少的欄位
    if missing:  # 有缺欄位
        raise ValueError(f"{path}: missing columns {missing}")  # 中止
    bad = df[df["color"].str.upper() != "B"]  # 非黑棋的棋局
    if len(bad):  # 原題池應該只有黑棋
        raise ValueError(f"{path}: {len(bad)} candidate games are not Black (the pools hold Black games only)")  # 中止
    return df.rename(columns={"sgf_content": "sgf"})[["player_id", "game_id", "sgf", "color"]].reset_index(drop=True)  # 統一欄位名稱後回傳


def read_questions(path):  # 讀官方題目檔
    """The official test CSV: [{question_id, declared_games, games: [(sgf, colour), ...]}] (aicup.submission.read_test)."""
    return read_test(path)  # 沿用棋力競賽的讀取函式


def question_score(top, truth):  # 一題的官方分數
    """e^-(r-1) when `truth` is the r-th of `top` (1-based), else 0."""
    top = list(top)  # 轉成清單以便查名次
    return math.exp(-top.index(truth)) if truth in top else 0.0  # 名次越後分數越低，不在前 5 名為 0


def summarise(tops, truth, n_games):  # 彙整官方分數與輔助指標
    """Official score and auxiliaries of Top-5 lists: {score, top1, top5, mrr, n, by_n: {size: {...}}}."""
    tops, truth, n_games = list(tops), list(truth), np.asarray(n_games)  # 統一型別
    s = np.array([question_score(t, y) for t, y in zip(tops, truth)])  # 每題的官方分數
    top1 = np.array([len(t) > 0 and t[0] == y for t, y in zip(tops, truth)], dtype=float)  # 第一名是否正確
    hit = np.array([y in t for t, y in zip(tops, truth)], dtype=float)  # 正解是否在前 5 名內
    rr = np.array([1.0 / (list(t).index(y) + 1) if y in t else 0.0 for t, y in zip(tops, truth)])  # 名次的倒數

    def block(m):  # 一組題目的平均指標
        return {"score": float(s[m].mean()), "top1": float(top1[m].mean()), "top5": float(hit[m].mean()), "mrr": float(rr[m].mean()), "n": int(m.sum())}  # 各指標在遮罩內取平均

    out = block(np.ones(len(s), dtype=bool))  # 全部題目
    out["by_n"] = {str(m): block(n_games == m) for m in sorted(set(n_games.tolist()))}  # 各局數分開統計
    return out  # 回傳結果


def write_submission(rows, path):  # 寫出提交檔
    """rows: [(question_id, [top_1..top_5])] in the inputs' order -> CSV with COLUMNS."""
    table = pd.DataFrame([{"question_id": q, **{f"top_{i + 1}": p for i, p in enumerate(top)}} for q, top in rows], columns=COLUMNS)  # 組成提交表
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    table.to_csv(path, index=False)  # 寫出 CSV
    return table  # 回傳表格


def validate_submission(path, question_ids, pool_players, expect_rows=None):  # 檢查提交檔的格式
    """Check a submission against the questions (ids in input order, public then private) and the pools' player ids
    ({prefix: set of ids}): header, every id once in order, no priv_ before a pub_, five distinct predictions all from the
    question's own pool, and the expected row count. Returns the list of problems ([] when valid)."""
    problems = []  # 收集問題
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # 以字串讀入
    if list(df.columns) != COLUMNS:  # 標頭不符
        problems.append(f"header {list(df.columns)} != {COLUMNS}")  # 記錄問題
        return problems  # 標頭錯誤就不再往下檢查
    ids = df["question_id"].tolist()  # 提交檔的題目 id
    if ids != list(question_ids):  # 與輸入的 id 或順序不同
        counts = Counter(ids)  # 計數每個 id
        dup = [q for q, c in counts.items() if c > 1]  # 重複的 id
        missing = [q for q in question_ids if q not in counts]  # 缺少的 id
        extra = [q for q in counts if q not in set(question_ids)]  # 多出來的 id
        problems.append(f"question ids differ from the inputs' order: {len(dup)} duplicated, {len(missing)} missing, {len(extra)} unknown")  # 記錄問題
    seen_priv = False  # 是否已出現 private 題
    for q in ids:  # 檢查 public 都在 private 之前
        if q.startswith("priv_"):  # 遇到 private 題
            seen_priv = True  # 記下已進入 private 段
        elif q.startswith("pub_") and seen_priv:  # private 之後又出現 public
            problems.append(f"{q}: a public question after a private one")  # 記錄問題
            break  # 回報一次即可
    for r in df.itertuples(index=False):  # 逐列檢查預測
        top = [getattr(r, f"top_{i}") for i in range(1, TOP_K + 1)]  # 該列的五個預測
        prefix = next((p for p in PREFIXES if r.question_id.startswith(p)), None)  # 題目所屬的題池
        if len(set(top)) != TOP_K or any(not t for t in top):  # 必須是五個不同的非空 id
            problems.append(f"{r.question_id}: the five predictions are not distinct non-empty ids")  # 記錄問題
        if prefix is None:  # 沒有已知的前綴
            problems.append(f"{r.question_id}: no pub_/priv_ prefix")  # 記錄問題
        elif prefix in pool_players:  # 有該題池的玩家清單時
            outside = [t for t in top if t not in pool_players[prefix]]  # 不在該題池的預測
            if outside:  # 有越界的預測
                problems.append(f"{r.question_id}: {len(outside)} predictions are not in the {prefix} pool (e.g. {outside[0]!r})")  # 記錄問題
    if expect_rows is not None and len(df) != expect_rows:  # 列數不符預期
        problems.append(f"{len(df)} rows, expected {expect_rows}")  # 記錄問題
    return problems  # 回傳問題清單


def make_pool_one_per_player(games, folds_df, folds, n_players=N_CANDIDATES, cand_games=CAND_GAMES, per_size=100, sizes=SIZES, seed=0, pool_id="sim", order="shuffle"):  # 一人一題的模擬題池
    """Like make_pool, but every chosen player gets exactly one question (the official design: 400 questions for 400
    candidates, 100 per size). Sizes are handed out largest first, at most per_size players each, only to players with
    >= cand_games + size Black games; a player left over takes the largest size they can afford. info adds
    one_per_player and per_size. ValueError when no player is eligible."""
    sizes = sorted(int(m) for m in sizes)  # 題目局數由小到大
    elig = eligible_players(games, folds_df, folds, cand_games + sizes[0])  # 至少能出最小一題的玩家
    if len(elig) == 0:  # 沒有合格玩家
        raise ValueError(f"folds {list(folds)}: no player has >= {cand_games + sizes[0]} Black games")  # 無法建立題池
    rng = np.random.RandomState(seed)  # 挑選玩家用的亂數
    chosen = elig.iloc[np.sort(rng.permutation(len(elig))[:n_players])]  # 隨機挑出玩家，保持原本的排序
    ranked = chosen.sort_values(["n_black", "player_id"], ascending=[False, True], kind="stable")  # 黑棋局數多的玩家優先分配大題
    assigned = {}  # 玩家 -> 題目局數
    for m in sorted(sizes, reverse=True):  # 局數由大到小分配
        slots = int(per_size)  # 這種局數的名額
        for p in ranked.itertuples():  # 依黑棋局數由多到少
            if slots == 0:  # 名額用完
                break  # 換下一種局數
            if p.player_id in assigned or p.n_black < cand_games + m:  # 已分配，或棋局不夠
                continue  # 略過
            assigned[p.player_id] = m  # 分配這種局數
            slots -= 1  # 名額減一
    for p in ranked.itertuples():  # 處理沒分到的玩家
        if p.player_id not in assigned:  # 尚未分配
            assigned[p.player_id] = max(m for m in sizes if p.n_black >= cand_games + m)  # 取負擔得起的最大局數
    by_player = games.groupby("player_id")["game_key"].apply(sorted).to_dict()  # 每位玩家排序後的棋局鍵
    cand_rows, q_rows = [], []  # 候選棋局列與題目列
    for p in chosen.itertuples():  # 逐位入選玩家
        keys = _player_keys(by_player[p.player_id], order, seed, p.player_id)  # 依切分順序排列棋局
        cand_rows += [{"player_id": p.player_id, "rank": p.rank, "game_key": k} for k in keys[:cand_games]]  # 前 cand_games 局作為候選棋局
        m = assigned[p.player_id]  # 這位玩家的題目局數
        q_rows.append({"question_id": f"{pool_id}_q_{len(q_rows) + 1:04d}", "player_id": p.player_id, "rank": p.rank,  # 題目 id 與擁有者
                       "n_games": m, "game_keys": keys[cand_games:cand_games + m]})  # 緊接在候選棋局之後的 m 局
    cands = pd.DataFrame(cand_rows, columns=["player_id", "rank", "game_key"])  # 候選棋局表
    qs = pd.DataFrame(q_rows, columns=["question_id", "player_id", "rank", "n_games", "game_keys"])  # 題目表
    info = {"folds": [int(f) for f in folds], "seed": int(seed), "eligible": int(len(elig)), "players": int(len(chosen)),  # 題池資訊：折、種子、合格與入選人數
            "cand_games": int(cand_games), "questions": int(len(qs)), "questions_by_n": {str(m): int((qs["n_games"] == m).sum()) for m in sizes},  # 候選局數、題數與各局數的題數
            "players_by_rank": {lv: int((chosen["rank"] == lv).sum()) for lv in LEVELS}, "one_per_player": True, "per_size": int(per_size), "order": order}  # 各等級人數與一人一題的設定
    return cands, qs, info  # 回傳三者
