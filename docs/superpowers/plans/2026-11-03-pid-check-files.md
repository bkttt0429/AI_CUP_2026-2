# 11/04 官方檔案檢查腳本（`pid_check_files.py`）— 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一支 CPU 腳本，在 11/04 官方候選庫與測試題到手後、任何建模與上傳之前，把分岔計畫第 3 節的檢查 1、2、4–10 一次算完寫成 JSON，並且在 11/04 前用模擬的官方檔演練一遍（同時給出第 7 項的參考值）。

**Architecture:** 三層。(1) `aicup_pid.dedup`：整局著手序列在 8 重對稱下的正規形做 128-bit 雜湊，訓練集 100 萬局分塊讀、多進程雜湊，建成「雜湊 → 訓練局」索引。(2) `aicup_pid.checks`：每一項檢查是一個純函式，吃已解析的著手與特徵、回 dict；候選庫內自我辨識重用 `components` 的模型與 `fusion` 的權重。(3) `scripts/pid_check_files.py` 把官方檔讀進來、跑所有檢查、印摘要、寫 `--out`；`scripts/pid_mock_official.py` 從模擬候選庫寫出官方格式的檔案（每人一題）供演練，順帶給支線 B3 用的「每人一題」候選庫。

**Tech Stack:** Python 3.12、numpy、pandas、torch（Fusion）、hashlib.blake2b、multiprocessing（fork）；AI_CUP_2026-1 的 `aicup.sgf.parse_moves`／`properties`、`aicup.submission.check_game`／`read_test`、`aicup.ladder.environment`／`to_json`。不加新套件。全部 CPU；計算一律 `sbatch scripts/nano4/run.sbatch`（從 AI_CUP_2026-1 根目錄送件）。

**Spec:** [docs/superpowers/specs/2026-11-03-pid-fork-plan-design.md](../specs/2026-11-03-pid-fork-plan-design.md) 第 3 節（檢查 1–10、A1／A2／A3 的判定）、第 4 節工程項（每人一題模式、provenance）、第 7 節成本；[docs/ARCHITECTURE.md](../../ARCHITECTURE.md) §4。

## Global Constraints

- 雜湊：`canonical(moves, len(moves))`（`aicup_pid.opening`，和 `pid_audit.whole_game` 同一個正規形）後，每手 3 bytes（顏色 1 byte：B=0、W=1；座標 2 bytes big-endian：x·19+y，pass = 380）串起來的 `blake2b(digest_size=16)`。同一局的 8 個對稱像必須同雜湊；改一手就不同。
- 開局位移的 token：正規化後黑棋前 k 手座標的 tuple（k = 4、8，`opening.tokens` 的 Black-only 前綴路徑）；距離 = Jensen–Shannon 散度（以 2 為底，0–1）；虛無參考 = 候選局隨機切成 |Q| 與其餘兩半、`RandomState(seed)`、20 次；A3(i) 旗標 = k=4 的觀測值 > 1.5 × 虛無最大值。
- 自我辨識：每位候選人的局以 `pools._player_rng(seed, player_id)` 打亂，前 `model_games`=50 局當模型、其餘依 5 → 10 → 15 → 20 切題（局數不夠就停，少於 50+5 局的候選人跳過並計數）；模型 = `OpeningModel`／`StyleModel`（`cfg` 同 `pid_fusion.json` 的 `cfg`）、融合權重 = `pid_fusion.json` 的 `weights`；分數 = `pools.summarise`；margin = 融合分數每列最大減次大。
- Top-1 碰撞：`distinct_top1 / n_questions >= 0.875`（400 題 ≥ 350）才立 `one_to_one_likely`；隨機期望 = C·(1 − (1 − 1/C)^Q)。
- 訓練集索引：只讀 `train_<rank>.csv` 的 `player_id, game_id, rank, color, sgf_content`，`chunksize` 20,000，`mp.get_context("fork").Pool(workers)`；每局保留 `(player_id, rank, game_key, color, n_moves, n_black)`；`--max-train-games` 只給冒煙測試。
- 解析失敗（`parse_moves` 回空或 `check_game` 不通過）的局不進開局位移與自我辨識，但要計數並列前 20 個例子；腳本不因單局錯誤中止。
- 每個結果 JSON 都帶 `aicup_pid.provenance.pid_environment("cpu")`（AI_CUP_2026-1 與本倉庫兩個 commit）。
- fold 0 不進任何模擬候選庫（`--pools` 含 0 直接拒絕）。
- 測試只用 `tests/synth.py` 的合成資料；真實資料的演練在 nano4 以 sbatch 跑並把 JSON 進 git（`results/checks/`）。
- git：分支 `nano4/pid-fork-plan`（從 `main` `6b107cf`），每個 task 一個 commit，PR 開在 `main`，合併等使用者。

## Review Focus

1. **候選人局數不是 100、候選人數不是 400**：自我辨識與每人一題模式都要照「有多少用多少」處理並在輸出記下實際數字，不得假設 100／400（Task 1 `test_one_per_player_fills_sizes_greedily`、Task 4 `test_self_identification_skips_short_candidates`）。
2. **測試檔的局不合法或 SGF 壞掉**：`check_game` 丟掉的局要在第 5 項計數，開局位移與第 8 項只用剩下的局，題目全壞時第 8 項的那題記 `None` 而不是崩（Task 6 `test_cli_tolerates_bad_games`）。
3. **pass 與對稱**：pass 在 8 個對稱下座標不變，雜湊必須把一局與它的 8 個對稱像視為同一局，且 B/W 顏色互換的「同座標序列」不算同一局（Task 3 `test_hash_symmetry_and_colour`）。
4. **跨庫重複**：public 與 private 的候選局若有同一局，要算進第 2 項（Task 6 `test_cli_cross_pool_duplicates`）。
5. **訓練集太大**：索引必須分塊、多進程，記憶體只存雜湊與六欄中繼資料；冒煙測試走 `--max-train-games`，真實演練量時間寫進 METHOD_NOTES（Task 3 `test_index_streams_chunks`、Task 8）。

---

## 檔案結構

| 檔案 | 責任 |
| --- | --- |
| `src/aicup_pid/pools.py`（改） | `make_pool_one_per_player`：每位候選人恰一題、各題長盡量各 `per_size` 題 |
| `src/aicup_pid/components.py`（改） | `fit_from_parsed`（已解析的著手與特徵 → 兩個模型）；`fit_models` 改呼叫它 |
| `src/aicup_pid/dedup.py`（新） | `game_hash`、`hash_games`（多進程）、`TrainingIndex`、`build_training_index` |
| `src/aicup_pid/checks.py`（新） | 檢查 1、2、4–10 的純函式 |
| `src/aicup_pid/provenance.py`（新，取自 PR #4 `b152ca0`） | `pid_environment` |
| `scripts/pid_check_files.py`（新） | CLI：官方檔 → 全部檢查 → JSON + 摘要 |
| `scripts/pid_mock_official.py`（新） | CLI：模擬候選庫 → 官方格式的候選庫／測試檔／真值檔 |
| `tests/test_pools.py`（改）、`tests/test_dedup.py`、`tests/test_checks.py`、`tests/test_check_files.py`、`tests/test_mock_official.py`、`tests/test_provenance.py`（新） | 測試 |
| `docs/NANO4.md` §4、`docs/ARCHITECTURE.md` §4、`METHOD_NOTES.md` §2.4、`README.md`（改） | 指令、檢查項 8–10、分岔計畫與演練結果、文件連結 |

---

### Task 1: `pools.make_pool_one_per_player`

**Files:**
- Modify: `src/aicup_pid/pools.py`（`make_pool` 之後）
- Test: `tests/test_pools.py`

**Interfaces:**
- Produces: `make_pool_one_per_player(games, folds_df, folds, n_players=N_CANDIDATES, cand_games=CAND_GAMES, per_size=100, sizes=SIZES, seed=0, pool_id="sim") -> (cands, qs, info)`，欄位與 `make_pool` 相同；`info` 多 `per_size` 與 `one_per_player: True`。每位被選的玩家恰好一題：題長由大到小分配，每個題長最多 `per_size` 位、只給執黑局數 ≥ `cand_games + size` 的人；分不到任何大小的玩家給他負擔得起的最大題長；仍分不到（局數 < cand_games + min(sizes)）的人不會被選（eligible 已排除）。

- [ ] **Step 1: 寫失敗的測試**（接在 `tests/test_pools.py` 既有測試後）

```python
def test_one_per_player_fills_sizes_greedily():
    frame, folds_df = training_frame(folds=(1, 2), players_per_fold=6, games_per_player=60)  # 30 Black games each
    cands, qs, info = pools.make_pool_one_per_player(black_table(frame), folds_df, [1, 2], n_players=8, cand_games=10, per_size=2, sizes=(5, 10, 15, 20), seed=0)
    assert info["players"] == 8 and info["one_per_player"] and info["per_size"] == 2
    assert len(qs) == 8 and qs["player_id"].is_unique
    assert qs["n_games"].value_counts().to_dict() == {20: 2, 15: 2, 10: 2, 5: 2}
    assert (cands.groupby("player_id").size() == 10).all()
    for r in qs.itertuples():
        assert len(r.game_keys) == r.n_games and not set(r.game_keys) & set(cands[cands["player_id"] == r.player_id]["game_key"])


def test_one_per_player_short_players_get_the_largest_affordable_size():
    frame, folds_df = training_frame(folds=(1,), players_per_fold=3, games_per_player=32)  # 16 Black games
    cands, qs, info = pools.make_pool_one_per_player(black_table(frame), folds_df, [1], n_players=3, cand_games=10, per_size=1, sizes=(5, 10, 15, 20), seed=0)
    assert len(qs) == 3 and sorted(qs["n_games"]) == [5, 5, 5]  # 6 spare games: only a 5-game question fits
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cd ~/AICUP_2026-1 && sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -t 00:20:00 -J pidt1 scripts/nano4/run.sbatch python -I -m pytest -q ../BK_AICUP_2026-2/tests/test_pools.py`
Expected: FAIL，`AttributeError: ... has no attribute 'make_pool_one_per_player'`

- [ ] **Step 3: 實作 `make_pool_one_per_player`**：重用 `eligible_players` 與 `_player_rng`；被選玩家依 `n_black` 降冪、同局數依 `player_id`，對 `sorted(sizes, reverse=True)` 逐一分配。

- [ ] **Step 4: 跑測試確認通過**（同一指令）Expected: PASS

- [ ] **Step 5: Commit** `git add src/aicup_pid/pools.py tests/test_pools.py && git commit -m "pools: make_pool_one_per_player -- one question per candidate, sizes filled largest first up to per_size each (spec section 4 engineering item; used by the self-identification check and line B3)"`

### Task 2: `components.fit_from_parsed`

**Files:**
- Modify: `src/aicup_pid/components.py:34-45`
- Test: `tests/test_scripts.py`（既有 `test_pid_cv_cli` 覆蓋 `fit_models`）、新測試在 `tests/test_checks.py` 由 Task 4 使用

**Interfaces:**
- Produces: `fit_from_parsed(cand_keys_by_player, moves, feats, cfg) -> (OpeningModel, StyleModel)`；`fit_models` 改為 `parse_all` + `features_all` 後呼叫它並回傳 `(opening, style, moves, feats)` 不變。

- [ ] **Step 1: 寫失敗的測試**（`tests/test_checks.py` 開頭）

```python
def test_fit_from_parsed_matches_fit_models():
    frame, _ = training_frame(folds=(1,), players_per_fold=4, games_per_player=12, seed=1)
    sgfs = {k: s for k, (s, c) in sgf_lookup(frame).items() if c == "B"}
    by_player = black_table(frame).groupby("player_id")["game_key"].apply(list).to_dict()
    o1, s1, moves, feats = fit_models(by_player, sgfs, {"depth": 8})
    o2, s2 = fit_from_parsed(by_player, moves, feats, {"depth": 8})
    assert o1.players == o2.players and o1.state() == o2.state() and np.allclose(s1.mu, s2.mu)
```

- [ ] **Step 2: 跑測試確認失敗** Expected: `ImportError: cannot import name 'fit_from_parsed'`
- [ ] **Step 3: 實作並讓 `fit_models` 呼叫它**（`StyleModel` 的均值屬性名以 `style.py` 為準，測試改用實際名稱）
- [ ] **Step 4: 跑 `tests/test_checks.py tests/test_scripts.py` 確認通過**
- [ ] **Step 5: Commit** `git commit -m "components: fit_from_parsed (models from already parsed games); fit_models delegates to it"`

### Task 3: `aicup_pid.dedup`

**Files:**
- Create: `src/aicup_pid/dedup.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `game_hash(moves) -> bytes`（16 bytes；空局回 `blake2b(b"").digest()`）
  - `hash_games(sgfs: dict[key, sgf], workers=1) -> dict[key, (hash, n_moves, n_black)]`（`parse_moves` 失敗或空局 → `(None, 0, 0)`）
  - `class TrainingIndex`: 屬性 `meta`（DataFrame：`player_id, rank, game_key, color, n_moves, n_black`）、`by_hash: dict[bytes, list[int]]`（meta 的列號）；方法 `lookup(h) -> list[int]`、`black_lengths() -> (n_moves array, n_black array) 只取 color == "B"`
  - `build_training_index(train_dir, workers=1, chunksize=20000, max_games=None, log=print) -> TrainingIndex`（讀 `LEVELS` 順序的 `train_<rank>.csv`）

- [ ] **Step 1: 寫失敗的測試**

```python
def test_hash_symmetry_and_colour():
    moves = parse_moves(sgf_of([("B", 15, 3), ("W", 3, 15), ("B", 16, 16), ("W", 2, 2), ("B", None, None), ("W", 9, 9)]))
    hs = {game_hash(_apply_symmetry(moves, k)) for k in range(8)}   # helper in the test: apply opening._sym to every move
    assert len(hs) == 1 and len(hs.pop()) == 16
    swapped = [("W" if c == "B" else "B", x, y) for c, x, y in moves]
    assert game_hash(swapped) != game_hash(moves)
    assert game_hash(moves[:-1]) != game_hash(moves)


def test_hash_games_counts_and_tolerates_garbage():
    out = hash_games({"a": sgf_of([("B", 3, 3), ("W", 15, 15), ("B", 3, 15)]), "b": "(;SZ[19])", "c": "not an sgf"})
    assert out["a"][1:] == (3, 2) and out["b"] == (None, 0, 0) and out["c"] == (None, 0, 0)


def test_index_streams_chunks(tmp_path):
    frame, _ = training_frame(folds=(1, 2), players_per_fold=3, games_per_player=10, seed=2)
    for r in LEVELS:
        sub = frame[frame["rank"] == r]
        if len(sub):
            sub[["player_id", "game_id", "rank", "color", "sgf_content"]].to_csv(tmp_path / f"train_{r}.csv", index=False)
    idx = build_training_index(tmp_path, workers=1, chunksize=7, log=lambda *a, **k: None)
    assert len(idx.meta) == len(frame) and set(idx.meta["rank"]) == set(frame["rank"])
    some = frame.iloc[5]
    rows = idx.lookup(game_hash(parse_moves(some["sgf_content"])))
    assert [idx.meta.iloc[i]["game_key"] for i in rows] == [f"{some['rank']}:{some['game_id']}"]
    n_moves, n_black = idx.black_lengths()
    assert len(n_moves) == (frame["color"] == "B").sum() and (n_black * 2 <= n_moves + 1).all()
    assert len(build_training_index(tmp_path, workers=1, chunksize=7, max_games=4, log=lambda *a, **k: None).meta) == 4
```

- [ ] **Step 2: 跑測試確認失敗** Expected: `ModuleNotFoundError: aicup_pid.dedup`
- [ ] **Step 3: 實作 `dedup.py`**：`game_hash` 依 Global Constraints 的 3 bytes 編碼；`hash_games` 用 `mp.get_context("fork").Pool`（workers > 1 時）對 sgf 列表 map；`build_training_index` 對每個 rank 檔 `pd.read_csv(..., usecols=[...], dtype=str, chunksize=chunksize)`，每塊雜湊後把列附加到 list，最後 `pd.concat`；`by_hash` 只收 hash 非 None 的列。`max_games` 到了就停止讀取。
- [ ] **Step 4: 跑測試確認通過**
- [ ] **Step 5: Commit** `git commit -m "dedup: canonical whole-game hash (8 symmetries, colours kept) and a chunked multiprocess index of the training set"`

### Task 4: `aicup_pid.checks`

**Files:**
- Create: `src/aicup_pid/checks.py`
- Test: `tests/test_checks.py`

**Interfaces:**
- Consumes: Task 2 `fit_from_parsed`、Task 3 `TrainingIndex`；`aicup_pid.opening.canonical`／`tokens`、`aicup_pid.fusion.Fusion`／`standardise`／`topk`、`pools.summarise`／`question_score`／`_player_rng`、`aicup.submission.check_game`、`aicup.sgf.properties`。
- Produces（全部回 JSON 可序列化的 dict）:
  - `duplicates_against_index(hashes: dict[key, hash|None], index) -> dict[key, list[dict(player_id, rank, game_key)]]`（只含有命中的 key）
  - `summarise_overlap(dups, owner: dict[key, player_id]) -> {n_games_checked, n_dup_games, n_players_with_dups, players_single_training_id, players_multiple_training_ids, same_id_players, by_player: {pid: {train_pid: n}}（最多 50 位，其餘計數）}`
  - `duplicates_between(hashes_a, hashes_b) -> list[(key_a, key_b)]`、`duplicates_within(hashes) -> list[(key_a, key_b)]`
  - `length_quantiles(n_moves, n_black) -> {n, moves: {p05,p25,p50,p75,p95,mean}, black_moves: {...}}`
  - `legality(sgfs: dict[key, sgf], colour="B") -> {n, n_bad, errors: [{key, error}] (≤ 20), handicap_counts: {hd: n}}`
  - `black_prefix(moves, k) -> tuple`
  - `js_divergence(counter_p, counter_q) -> float`
  - `opening_shift(q_moves: list, c_moves: list, k, n_null=20, seed=0, factor=1.5) -> {k, n_query, n_candidate, observed, null: [..], null_mean, null_max, ratio_to_null_max, flag}`
  - `self_identification(by_player: dict[pid, list[key]], moves, feats, cfg, weights, seed=0, model_games=50, sizes=SIZES) -> {n_candidates, n_skipped, n_questions, scores: {opening, style, fusion: pools.summarise 格式}, fusion_margin_quantiles: {p05,p25,p50,p75,p95}, candidates_misidentified: int}`
  - `top1_collisions(S_f, n_games, fraction=0.875) -> {n_questions, n_candidates, distinct_top1, expected_random, by_n: {n: distinct}, one_to_one_likely}`
  - `num_games_counts(questions) -> {str(n): count}`
  - `record_properties(sgfs: dict[key, sgf], game_ids: list[str]) -> {properties: {prop: n_games}, game_id_examples: [≤5], game_id_all_numeric: bool, game_id_sorted_in_file: bool|None}`

- [ ] **Step 1: 寫失敗的測試**（合成資料；每個函式至少一條）

```python
def test_overlap_summary_maps_candidates_to_training_players(tmp_path):
    # training dir from training_frame; candidates = 2 Black games of player A (exact + one symmetric copy) and 1 novel game
    ...  # build index as in test_index_streams_chunks
    dups = duplicates_against_index(hashes, idx)
    s = summarise_overlap(dups, owner)
    assert s["n_dup_games"] == 2 and s["n_players_with_dups"] == 1 and s["players_single_training_id"] == 1
    assert s["same_id_players"] == 0 and list(s["by_player"]["pub_p_A"].values()) == [2]

def test_duplicates_between_and_within():
    assert duplicates_between({"q1": h1, "q2": h2}, {"c1": h2, "c2": h3}) == [("q2", "c1")]
    assert duplicates_within({"a": h1, "b": h1, "c": None}) == [("a", "b")]

def test_length_quantiles_and_legality():
    q = length_quantiles(np.array([10, 20, 30, 40]), np.array([5, 10, 15, 20]))
    assert q["n"] == 4 and q["moves"]["p50"] == 25.0 and q["black_moves"]["mean"] == 12.5
    leg = legality({"ok": sgf_of([("B", 3, 3), ("W", 15, 15)]), "bad": sgf_of([("W", 3, 3), ("B", 15, 15)]), "junk": "(;SZ[9];B[aa])"})
    assert leg["n"] == 3 and leg["n_bad"] == 2 and {e["key"] for e in leg["errors"]} == {"bad", "junk"}

def test_opening_shift_flags_a_different_habit_set():
    same habit set split: flag False and observed <= null_max * 1.5; questions drawn from players with other habits (seed 7): flag True at k=4

def test_self_identification_recovers_planted_habits_and_skips_short_candidates():
    6 players x 30 Black games (training_frame games_per_player=60) → model_games=10, sizes (5, 10) → every player gets questions;
    one player truncated to 12 games → counted in n_skipped; scores["fusion"]["top1"] >= 0.5; margin quantile keys present

def test_top1_collisions():
    S = np.eye(6)[[0, 1, 1, 2, 3, 3]]  # 6 questions, 6 candidates, two collisions
    r = top1_collisions(S, np.array([5, 5, 10, 10, 20, 20]))
    assert r["distinct_top1"] == 4 and r["by_n"] == {"5": 2, "10": 1, "20": 1} and not r["one_to_one_likely"]
    assert abs(r["expected_random"] - 6 * (1 - (1 - 1 / 6) ** 6)) < 1e-9
    assert top1_collisions(np.eye(6), np.full(6, 5))["one_to_one_likely"]

def test_num_games_and_record_properties():
    assert num_games_counts([{"declared_games": 5}, {"declared_games": 5}, {"declared_games": 20}]) == {"5": 2, "20": 1}
    p = record_properties({"a": "(;SZ[19]HD[2];B[pd];W[dp])", "b": "(;SZ[19];B[pd])"}, ["g_000002", "g_000001"])
    assert p["properties"] == {"SZ": 2, "HD": 1, "B": 2, "W": 1} and p["game_id_all_numeric"] is False and p["game_id_sorted_in_file"] is False
```

- [ ] **Step 2: 跑測試確認失敗** Expected: `ModuleNotFoundError: aicup_pid.checks`
- [ ] **Step 3: 實作 `checks.py`**。`opening_shift` 的 token 用 `canonical(moves, 2*k)` 後取前 `2*k` 手中的黑棋座標（不足 k 手的局用現有的）；`self_identification` 的題目切法與 Global Constraints 一致，模型用 `fit_from_parsed`，融合用 `Fusion()` 並設 `w = weights`，分數用 `pools.summarise`，`candidates_misidentified` = 至少一題 Top-1 不是自己的候選人數；`record_properties` 的 `game_id_all_numeric` 用 `str.isdigit`，`game_id_sorted_in_file` 在全數字時比較整數序、否則字串序。
- [ ] **Step 4: 跑測試確認通過**
- [ ] **Step 5: Commit** `git commit -m "checks: the 11/04 file checks as pure functions -- duplicates against the training index / between files, length quantiles, legality, opening-distribution shift with a split-half null, within-library self-identification, Top-1 collisions, num_games, record properties"`

### Task 5: provenance（取自 PR #4）

**Files:**
- Create: `src/aicup_pid/provenance.py`、`tests/test_provenance.py`（`git checkout origin/claude/pid-rank-overlap -- src/aicup_pid/provenance.py tests/test_provenance.py`）

- [ ] **Step 1: 取檔並跑 `tests/test_provenance.py`** Expected: PASS（內容不改；commit 訊息註明來源 `b152ca0`）
- [ ] **Step 2: Commit** `git commit -m "provenance: pid_environment records this repository's commit next to AI_CUP_2026-1's (lifted from PR #4 b152ca0, unchanged)"`

### Task 6: `scripts/pid_check_files.py`

**Files:**
- Create: `scripts/pid_check_files.py`
- Test: `tests/test_check_files.py`

**Interfaces:**
- Consumes: Tasks 2–5；`pools.read_candidates`／`read_questions`；`components.parse_all`／`features_all`／`score_questions`。
- CLI: `--candidates PUB PRIV --questions PUB PRIV --fusion results/pid_fusion.json --out results/checks/<name>.json [--train-dir DIR] [--no-training] [--max-train-games N] [--workers N] [--seed 0] [--n-null 20] [--model-games 50]`。退出碼：2 讀檔／格式錯誤，否則 0（檢查只報告）。
- 輸出 JSON：`{"pools": [{"prefix", "candidates": path, "questions": path, "n_candidates", "n_candidate_games", "n_questions", "num_games", "legality": {"candidates", "questions"}, "lengths": {"candidates", "questions"}, "training_overlap": {"candidates": summarise_overlap, "questions": summarise_overlap} | None, "question_vs_candidate_dups", "question_vs_question_dups", "opening_shift": {"4", "8"}, "self_identification", "main_line": {"top1_collisions", "margin_quantiles", "top1": [[question_id, candidate]]}, "records": {"candidates", "questions"}, "seconds"}], "cross_pool_candidate_dups": int, "training": {"n_games", "lengths_black", "seconds"} | None, "flags": {"A1_training_overlap", "A3i_opening_shift", "one_to_one_likely": {prefix: bool}}, "fusion": doc, **pid_environment}`。
- 摘要列印：每庫一行重複數、一行合法性、一行開局位移（k=4：觀測／虛無最大／旗標）、一行自我辨識（score／top1／n=5）、一行 Top-1 碰撞；最後三個旗標。

- [ ] **Step 1: 寫失敗的測試**（重用 `tests/test_scripts._official_files`，把它搬到 `tests/synth.py` 成 `official_files(tmp_path, frame, prefix, n_q)` 並讓 `test_scripts.py` 改 import）

```python
def test_cli_end_to_end_with_planted_overlap(tmp_path, capsys):
    train frame (folds 1,2; 6 players x 40 games) written as train_<rank>.csv in tmp_path/"train";
    pub files from the SAME frame (so every candidate game is in training), priv files from another seed (no overlap);
    fusion json {"components": ["opening", "style"], "weights": [1.0, 0.5], "cfg": {"depth": 8}}
    rc = pid_check_files.main([..., "--train-dir", str(train), "--out", str(out), "--model-games", "6"])
    assert rc == 0
    res = json.loads(out.read_text())
    pub, priv = res["pools"]
    assert pub["training_overlap"]["candidates"]["n_dup_games"] == pub["n_candidate_games"] and priv["training_overlap"]["candidates"]["n_dup_games"] == 0
    assert res["flags"]["A1_training_overlap"] is True
    assert pub["self_identification"]["n_questions"] > 0 and set(pub["opening_shift"]) == {"4", "8"}
    assert len(pub["main_line"]["top1"]) == pub["n_questions"] and "pid_git_head" in res
    assert "A1" in capsys.readouterr().out

def test_cli_no_training_and_bad_games(tmp_path):
    one question's games replaced by garbage, another's first game by a White-first game →
    res["pools"][0]["legality"]["questions"]["n_bad"] == 3 (2 garbage + 1 illegal); main_line top1 entry for the all-garbage question is [qid, None];
    res["training"] is None and res["flags"]["A1_training_overlap"] is None

def test_cli_cross_pool_duplicates(tmp_path):
    the same frame for pub and priv → res["cross_pool_candidate_dups"] == n_candidate_games of one pool

def test_cli_refuses_unpaired_files(): SystemExit on 2 candidates vs 1 questions
```

- [ ] **Step 2: 跑測試確認失敗** Expected: `ModuleNotFoundError: pid_check_files`
- [ ] **Step 3: 實作腳本**：結構照 `predict_pid.py`（ROOT／sys.path 區塊、`main(argv)`）；每庫：讀檔 → `check_game` 全部局 → 可用局 `parse_all`／`features_all` → 雜湊（`hash_games`）→ 檢查 → 主線評分（`fit_from_parsed` 於全部候選局、`score_questions`、`Fusion` 權重、`topk`）→ 第 8 項。訓練索引建一次給兩庫用。
- [ ] **Step 4: 跑 `tests/test_check_files.py tests/test_scripts.py` 確認通過**
- [ ] **Step 5: Commit** `git commit -m "pid_check_files.py: the 11/04 official-file checks (spec section 3) -- training overlap, duplicates, legality, lengths, opening shift, self-identification, Top-1 collisions, num_games, record properties; flags A1 / A3(i) / one-to-one"`

### Task 7: `scripts/pid_mock_official.py`

**Files:**
- Create: `scripts/pid_mock_official.py`
- Test: `tests/test_mock_official.py`

**Interfaces:**
- Consumes: Task 1 `make_pool_one_per_player`、`aicup.scoring.load_games`。
- CLI: `--pools 1,2 3,4 --out-dir data/mock_official --seed 0 [--n-players 400] [--cand-games 100] [--per-size 100] [--folds-csv]`；第 i 個 pool 用 `PREFIXES[i]`（最多兩個）。寫 `player_identification_candidates_<public|private>.csv`（`game_id, player_id, sgf_content, color`；`player_id` = `<prefix>p_<訓練 player_id>`、`game_id` = `<prefix>g_<序號:06d>`）、`player_identification_test_<public|private>.csv`（`question_id, num_games, sgf_1..20, color_1..20`，`question_id` = `<prefix>q_<序號:04d>`）、`truth_<public|private>.csv`（`question_id, player_id, train_player_id, rank, n_games`）、`mock_info.json`（兩庫 `info` + `pid_environment`）。fold 0 拒絕（退出碼 2）。

- [ ] **Step 1: 寫失敗的測試**

```python
def test_mock_official_writes_official_format(tmp_path, monkeypatch):
    frame, folds_df = training_frame(folds=(1, 2, 3, 4), players_per_fold=5, games_per_player=60)
    monkeypatch black_games / load_games as tests/test_scripts._patch_data does for pid_mock_official
    rc = pid_mock_official.main(["--pools", "1,2", "3,4", "--out-dir", str(tmp_path / "mock"), "--n-players", "8", "--cand-games", "10", "--per-size", "2", "--folds-csv", ...])
    assert rc == 0
    cands = pools.read_candidates(tmp_path / "mock" / "player_identification_candidates_public.csv")
    qs = pools.read_questions(tmp_path / "mock" / "player_identification_test_public.csv")
    assert cands["player_id"].str.startswith("pub_p_").all() and len(cands) == 80 and len(qs) == 8
    truth = pd.read_csv(tmp_path / "mock" / "truth_public.csv", dtype=str)
    assert set(truth["player_id"]) <= set(cands["player_id"]) and truth["question_id"].tolist() == [q["question_id"] for q in qs]
    assert all(q["declared_games"] == len(q["games"]) for q in qs)
    assert pid_mock_official.main(["--pools", "0,1", "--out-dir", str(tmp_path / "x"), "--folds-csv", ...]) == 2
```

- [ ] **Step 2: 跑測試確認失敗** Expected: `ModuleNotFoundError: pid_mock_official`
- [ ] **Step 3: 實作腳本**（`read_test` 要求只有前 `num_games` 欄非空，其餘填空字串）
- [ ] **Step 4: 跑測試確認通過**
- [ ] **Step 5: Commit** `git commit -m "pid_mock_official.py: official-format candidate / test / truth files from a one-question-per-player simulated pool (11/04 rehearsal; line B3 evaluation set)"`

### Task 8: 文件、nano4 演練、結果進 git

**Files:**
- Modify: `docs/NANO4.md`（新 §4「11/04 流程」：mock 生成、檢查腳本於 mock、11/04 當天的三道指令：檢查 → predict → 對帳；§1 測試指令不變）、`docs/ARCHITECTURE.md` §4（加第 8–10 項與腳本名、連到 spec）、`METHOD_NOTES.md` §2.4（分岔計畫一段 + mock 演練的數字：訓練索引秒數、自我辨識參考值、Top-1 碰撞、開局位移虛無值）、`README.md`（連到 spec 與 plan）
- Results: `results/checks/mock_official_checks.json`、`results/checks/mock_official_info.json`（mock 的 CSV 在 `data/`，不進 git）

- [ ] **Step 1: 全套測試**

Run: `cd ~/BK_AICUP_2026-1 && sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -t 00:30:00 -J pidtest scripts/nano4/run.sbatch python -I -m pytest -q ../BK_AICUP_2026-2/tests`
Expected: 全部 passed（原 13 + 新增）

- [ ] **Step 2: 產生 mock 官方檔**

Run: `sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 01:00:00 -J pidmock scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_mock_official.py --pools 1,2 3,4 --out-dir ../BK_AICUP_2026-2/data/mock_official --seed 0`
Expected: 兩庫各 400 位候選人、每人一題；`mock_info.json` 的 `questions_by_n` 盡量各 100

- [ ] **Step 3: 在 mock 上跑檢查腳本（含訓練索引）**

Run: `sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidchk scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_check_files.py --candidates ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_private.csv --questions ../BK_AICUP_2026-2/data/mock_official/player_identification_test_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_test_private.csv --fusion ../BK_AICUP_2026-2/results/pid_fusion.json --out ../BK_AICUP_2026-2/results/checks/mock_official_checks.json --workers 12`
Expected（正控制）：兩庫候選局與測試局 100% 在訓練集、`A1_training_overlap` True、每位候選人對應恰一個訓練 `player_id`；測試局不在候選局；開局位移旗標 False；自我辨識分數（這是第 7 項的參考值）與 Top-1 碰撞的 `distinct_top1`（一人一題的 mock 應 ≥ 350）。記下秒數。

- [ ] **Step 4: 寫文件、結果進 git、Commit**

`git add docs METHOD_NOTES.md README.md results/checks && git commit -m "docs + results: 11/04 procedure in NANO4, checks 8-10 in ARCHITECTURE, fork plan and the mock rehearsal numbers in METHOD_NOTES 2.4; mock_official_checks.json"`

- [ ] **Step 5: push、開 PR（base `main`）、在 issue #1 貼摘要（`[nano4 Claude]`），並為分岔計畫開一個新 issue 當這條線的頻道**。合併等使用者。

### Task 9（執行中追加）: `scripts/pid_reconcile.py`

**Files:** Create `scripts/pid_reconcile.py`、Test `tests/test_reconcile.py`

**Interfaces:** `reconcile(checks: dict, submission: DataFrame, truth: DataFrame | None) -> dict`（`n_compared`、`n_top1_equal`、`n_top1_differ`、`differences`、`checks_without_top1`、`missing_in_submission`、`missing_in_checks`、`truth_score`）；CLI `--checks --submission [--truth ...] --out`，退出碼 1 = Top-1 有不同或有缺題（11/04 上傳前的門），2 = 讀檔錯誤。

- [x] 測試（同意、不同意、讀檔錯誤三條）→ 紅 → 實作 → 綠（job 684690 之後的全套）
- [x] 在 mock 上與 `predict_pid.py` 串跑（job 684685）：800/800 相同，對真值 0.9474

## 執行紀錄（2026-11-03）

- Task 1–7 依序完成，每個 task 一個 commit（provenance 的檔案因先 staged 而併入 Task 1 的 commit）。
- 兩個計畫外的發現：(a) nano4 一張 GPU 的工作上限 `-c 12`，所有 `-c 24` 指令改 12；(b) `predict_pid.py` 以 `game_id` 當候選局 key 的隱患（合成資料的重複 id 暴露），改位置 key並加測試。
- mock 演練數字在 METHOD_NOTES §2.4；第 7 項參考值 0.936 寫回 spec §3.1。

