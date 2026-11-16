# 選擇機會與情境偏好探針 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本計畫建議同一執行者逐項實作，避免多個 worker 同時改動評估與切分定義；需要時再做獨立審查。

**Goal:** 驗證現行 P6／完整定案之外，逐局面的動作選項 covariance 和價值×偏好交互作用是否具有可重現的互補訊號。

**Architecture:** 凍結 humanv0 與既有 g/x/e 投影，用小量重播產生每局 U/F，再由候選人的 reference 局估低維個人參數。R、A0、A1、A2、A2perm 共享同一抽取與 baseline；僅 A2 是主要候選，所有比較經每個 outer seed 內的玩家 cross-fit 與重新選參數的 bootstrap。先通過輸入／近似稽核，再跑完整候選擬題池。

**Tech Stack:** R1 既有 Python venv、PyTorch、NumPy、SciPy、KataGo Python replay、Slurm。無新套件、無全模型訓練。

**Spec:** `docs/superpowers/specs/2026-11-16-choice-signal-design.md`（執行者必須先讀，尤其 §2 舊實驗去重、§5 近似 gate、§6 同 seed 資料隔離）。

## Global Constraints

- 上限：同時 **2 GPU**；第一輪含檢查的總配額 **6 GPU 小時**、新增磁碟 **1 GiB**。
- quota 門檻：`/work/u2507115` 可用至少 **6 GiB**、`/home/u2507115` 至少 **3 GiB**。
- 全部計算／測試走 production Slurm，使用 R1 venv/run.sbatch；登入節點只編檔、查檔、送件與讀結果。
- fold 0 不用；官方 Public／Private 不混用；只讀 candidate CSV，不讀官方 question CSV 或答案；不寫 submission／upload。
- 不修改現行 trainer／extractor／fusion registry，不碰正在跑的 o26、m161r、choice64 輸出，不刪既有資料。
- 外部來源與 source manifest 不符即停；缺鍵、非法棋、非有限數不得靜默補零／略過。
- 保留既有 Python replay 的 `board_simple_ko_no_superko` 限制並明記；不得把空點 mask 宣稱完整合法判斷。
- query 只計分，不用來估自己的候選 z、投影、尺度或選自己所用的超參數；同 seed 的所有 held-out 局也不得作其他候選人的 reference。
- 陰性只限制本次投影／近似／資料量；區間太寬要報 inconclusive，不能稱沒有訊號。

## Review Focus

1. 兩個 pool 相同 c:key 或同玩家不同 seed：namespace 及來源隔離不能串錯；Task 1/4 必須測。
2. pass、提子、simple ko、執白／讓子：動作集合、落子前局面與價值方向必須正確；Task 2/3 必須測。
3. α=0、零資訊局、F 近奇異：數值有限且不憑空改排名；Task 2/4 必須測。
4. 中斷寫檔／磁碟不足／其他 session 改檔：可安全續算且不讀半檔；Task 1/3/5 必須測。
5. 負候選引用查詢資料、調參多重選擇與同局反覆出現：test half 必須與選參數隔離，bootstrap 必須重選 grid；Task 4 必須測。

## 執行順序與交付物

`Task 1 manifest → Task 2 數學 → Task 3 pilot／校準 → Task 4 評估 → Task 5 正式研究與回報`。

這份文件只規劃 A 的機制探針。B（事件反應）、C（候選分歧）、更大型少樣本 policy 及舊題池出檔驗證都在 spec §9 的後續範圍，不把它們混入本輪 budget。

### Task 1：鎖定來源、棋局與每個切分的資料角色

**Files:**
- Create: `scripts/pid_choice_manifest.py`
- Create: `tests/test_choice_pipeline.py`
- Reference only: `src/aicup_pid/v2.py::candidate_holdout/training_games`、`paths.py::FOLDS_CSV`、`scripts/pid_probe_bottleneck.py::one_seed`

**Interfaces:**
- `build_manifest(run_root: Path, *, fusion: Path, projection: Path, checkpoint: Path) -> Path` 回傳新 run 目錄。
- `read_manifest(run_dir: Path) -> dict` 驗 schema、hash、所有唯一 key，不做容錯替換。
- `seed_split(manifest: dict, pool: str, seed: int) -> dict` 回傳 `reference: dict[owner,list[str]]`、`heldout: dict[owner,list[str]]`、`halves: dict[owner,int]`；key 為 `pool|original_key`。
- CLI：`python -I scripts/pid_choice_manifest.py --run-root ROOT --fusion FILE --projection FILE --checkpoint FILE`；預設 candidate 檔為 `data/tests_1111/player_identification_candidates2_{public,private}.csv`。不得推導或打開 question 檔。

- [ ] **Step 1: 寫失敗測試** `test_choice_pipeline.py`：
  - `test_pool_namespace_preserves_original_key`：兩池同原始 key 留兩筆、不得互覆。
  - `test_holdout_matches_existing_and_excludes_all_query_games`：7/3 與現有 `candidate_holdout` 完全相同；任何 held key 不在同 seed 任一 candidate reference。
  - `test_calibration_has_100_players_ten_games_and_no_fold0`：固定 hash 次序、每人 10 局，7/3 分開；不足 100 人報錯，不放寬。
  - `test_wrong_projection_or_changed_source_refused`：blocks/fit_folds/shape 與 SHA 有任一不符就失敗。
  - `test_source_symlink_resolves_to_recorded_target`：記錄 realpath；同內容兩路徑若 inode 相同不誤算副本，改 target 後拒絕 resume。
- [ ] **Step 2: Slurm 跑上述測試，確認因缺新介面失敗**，不是 import path 或環境錯誤。
- [ ] **Step 3: 實作 manifest／split**，固定 spec §4/§6 的 hash salts、100×10 校準、600×10 兩池與 seed 0–5。manifest 存實際 SGF hash、player/color、CSV row index、parts、依賴檔 hash；設定 body SHA 前 12 字作 run_id，manifest 不自我 hash。run_id/source 相同才可續用。
- [ ] **Step 4: 重跑測試通過，生成唯讀 manifest**；確認只產生 manifest，不跑模型。不足輸入即報 blocking precondition，不自動補抽外部資料。
- [ ] **Step 5: 在隔離的研究分支只提交 Task 1 檔案**，不提交 data、其他 session 改動或自動改動 METHOD_NOTES。

### Task 2：實作可對照的條件選擇數學

**Files:**
- Create: `src/aicup_pid/choice_signal.py`
- Create: `tests/test_choice_signal.py`

**Interfaces:**
- `choice_moments(logp: torch.Tensor, phi: torch.Tensor, played: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,int]`：logp `[T,A]`、phi `[T,A,D]`、played `[T]`；回 U `[D]`、F `[D,D]`、n。被 mask 動作可為 −inf，但非法實際手要報錯。
- `context_features(psi: torch.Tensor, value: torch.Tensor) -> torch.Tensor`：psi `[T,A,33]`，value `[T]`，回 `[T,A,66]` 的 `[psi,v*psi]`。
- `permute_context(value: np.ndarray, plies: np.ndarray, *, pool: str, key: str) -> np.ndarray`：精確依 spec §4 分期／salt；原陣列不被修改。
- `normalise_moments(U: np.ndarray,F: np.ndarray,scale: np.ndarray) -> tuple[np.ndarray,np.ndarray]`：U/scale、F 外積尺度除法，FP64 評分。
- `fit_candidate(U: np.ndarray,F: np.ndarray,n: int,ridge: float) -> np.ndarray`：n>0、λ>0，solve 而非 inverse。
- `score_candidates(Uq: np.ndarray,Fq: np.ndarray,Z: np.ndarray) -> np.ndarray`：`[Q,D]、[Q,D,D]、[C,D] → [Q,C]`；按 row/candidate 分塊控制暫存。
- `exact_delta(logp: torch.Tensor,phi: torch.Tensor,played: torch.Tensor,z: torch.Tensor) -> torch.Tensor`：逐手 exact log likelihood ratio，logsumexp 穩定計算，不改 p0。

- [ ] **Step 1: 寫失敗測試**：
  - `test_binary_choice_moments_and_context`：p=(.25,.75)、φ=(0,2)、選第二點 → U=.5、F=.75；v=.5 時 U=(.5,.25)、F=((.75,.375),(.375,.1875))。
  - `test_common_action_offset_cancels`：所有動作同加常數向量，U/F 與 exact ratio 不變。
  - `test_forced_move_zero_information`：唯一可選動作 → U=F=0，candidate score 為 0，不因取 log(0) 出 NaN。
  - `test_pass_and_masked_actual_move`：pass 額外 indicator 可辨、illegal logits 不參與分布、實際非法手失敗。
  - `test_fisher_matches_exact_hessian`：小 FP64 合成 logits 上以 autograd 算梯度/Hessian，與 U/F 對應至 1e-8。
  - `test_second_order_matches_small_exact_tilt`：小 z 誤差隨 z 縮半約三次方下降；大 z 不假裝精確，audit 可拒絕。
  - `test_singular_fisher_has_finite_ridge_solution`：零 F／rank-deficient F 在 λ>0 有限；n=0 或 λ≤0 失敗。
  - `test_context_permutation_preserves_phase_multisets`：同 phase v multiset、維度不變；相同 key 重跑一致，不碰其他池。
- [ ] **Step 2: Slurm 跑 `tests/test_choice_signal.py`，確認缺實作造成失敗。**
- [ ] **Step 3: 實作上述純函式**；前向 p0/moment 累積 FP32，解線性方程與最終比較 FP64；有限性檢查、F 對稱及負特徵值容忍度明定為 `min_eig >= -1e-5 * max(trace(F),1)`，超過即拒絕，不投影修正成另一模型。
- [ ] **Step 4: 重跑測試通過**；以合成資料比 R/A0/A1/A2/A2perm 的零訊號情況，確保各臂不靠維度或缺資料產生偏置。
- [ ] **Step 5: 只提交 Task 2 模組與測試**。

### Task 3：獨立抽取、256 局試跑與近似稽核

**Files:**
- Create: `scripts/pid_choice_extract.py`
- Extend: `tests/test_choice_pipeline.py`
- Reference only: `scripts/pid_trunk_extract.py::replay_target`、`pid_val_extract.py::forward_all`、R1 `src/aicup/humansl.py`、`src/aicup/l1b.py::neutral_rows`

**Interfaces:**
- `replay_choices(key: str,sgf: str,color: str) -> dict`：回落子前 bins/globs、完整 board-simple-ko masks、played、mover、原始 ply；只採計前 256 ply 的主角回合，保存失敗原因。
- `extract_game(model, replay: dict, projection: np.ndarray, *, pool: str,key: str) -> dict`：回未縮放 U0/F0/U2/F2/U2perm/F2perm/n。每局暫存必要的 ψ/v 以做 phase permutation，寫完丟棄，不存 361 點 trunk。
- `write_shard(run_dir: Path,part_id: str,arrays: dict,metadata: dict) -> Path`：tmp/fsync/rename；done JSON 含 checksum，存在不相同內容拒絕覆寫。
- CLI：`--run-dir DIR --stage pilot|calibration|official --shard I --nshards N`；pilot 為校準清單前 256 局，calibration 跳過已完成 pilot。
- audit CLI：`--run-dir DIR --audit calibration`、`--run-dir DIR --audit reference --pool pub2|priv2 --seed S`。audit reference 嚴格不載任何該 seed 的 outer-held SGF；依 spec §5 選 norm/hash 候選及 blocks。

- [ ] **Step 1: 補失敗測試**：`test_replay_before_move_and_white_value_sign`、`test_capture_simpleko_suicide_handicap_and_pass`、`test_no_target_moves_fails_coverage`、`test_shard_interrupt_and_changed_resume`；重播結果對現有 reference path，不能拿新函式自己當答案。
- [ ] **Step 2: Slurm 跑 targeted tests 確認缺實作失敗**；再實作 replay 與固定 projection metadata 檢查。不得修改被現行工作讀取的舊 extractor。
- [ ] **Step 3: 實作串流抽取與原子 shard 寫入**，每 shard 64 局；1,024 校準 audit 位置為校準 held-out 三局的候選位置按固定 hash 取前 1,024，另存 p0/ψ/v，不依模型正確率抽。記錄 extractor/source/model SHA、legality、精度、丟失 key。
- [ ] **Step 4: 重跑測試後執行 256 局 pilot**，用 1 GPU、12 CPU、30 分鐘。保存 `pilot.json`：總局／有效手／replay與GPU耗時／warmup分離／bytes與峰值／最慢 shard。這時不讀效能指標選模型。
- [ ] **Step 5: 量測後決定是否在預算內完成**：將剩餘 744 校準局＋12,000 官方候選局、reference audit、評估時間一併估入；估不下就報 budget_exceeded，不自行增資。time limit 依 spec §8 留餘裕，GPU allocation 的 idle/setup 也計入 6 GPUh。
- [ ] **Step 6: 完成校準，寫 `normalizer.npz` 與 `audit_calibration.json`**。fit z 僅每人七局，檢驗其三局上的 exact vs Taylor；A1/A2/A2perm 各過 median/p95 gate，λ 取共同通過集合。無可用 λ 停止，不開始官方大批。
- [ ] **Step 7: 完成正式抽取／reference audit 的介面，先用合成 shard 驗證覆蓋與 gate**。正式 12,000 局與各 seed 的稽核在 Task 5 才執行；raw moments 約 500 MiB，temp+audit+report 峰值仍須 <1 GiB。reference audit 的排序 margin 必須用同 block exact scores 的列標準化分差（0.05），不能用固定 nat 門檻誤殺高 λ；std<1e-12 記無辨識力。不能在官方 audit 後篩 λ 救結果。
- [ ] **Step 8: 只提交 extractor 與測試**；資料不入 git，稽核摘要與 job IDs 寫研究紀錄。

### Task 4：嚴格 baseline、同切分 cross-fit 與程序級 bootstrap

**Files:**
- Create: `src/aicup_pid/choice_eval.py`
- Create: `scripts/pid_choice_eval.py`
- Create: `tests/test_choice_eval.py`

**Interfaces:**
- `GridPoint(arm: str,ridge: float|None,alpha: float)`：R 的 ridge=None，其餘僅 gate 通過的 λ。
- `GridScores` dataclass：`pool:str, seeds:tuple[int,...], owners:tuple[str,...], halves:np.ndarray[C], configs:tuple[GridPoint,...], score:np.ndarray[S,G,C,3], base:np.ndarray[S,C,3]`；最後一維是等權題型 1/2/3，各 owner 的 1 局題先平均三局。
- `choose_grid(scores: np.ndarray[G,C,3],configs:tuple[GridPoint,...],weights:np.ndarray[C],arm:str) -> int`：weights 只可在來源 half 非零；依等權題型平均最大、α較小、λ較大平手規則。
- `crossfit_grid(data: GridScores) -> dict`：每 seed 每方向各自選，輸出 selection lock、selected grid index、per-owner delta；不從別的 seed 讀分數。
- `bootstrap_procedure(pools: dict[str,GridScores],*,seeds:tuple[int,...],draws:int=5000,rng_seed:int=20261116) -> dict`：pool×half owner 分層，跨 seed 同人用同重數，每次重選 grid；回主要 A2−B 與機制 contrasts 的估計/CI/selection frequency。
- `load_baseline_inputs(manifest:dict,pool:str,seed:int) -> dict`：只載 candidate SGF、原文件列出的成分與該 seed 的 reference-only t50；拒絕 missing/nonfinite，不能使用 `Pool` 的 query loader 或 fallback。
- CLI：`--run-dir DIR --stage baseline-check|explore|evaluate|robustness`，分別處理 baseline seed 0–3、機制 seed 0/1、主要 seed 2/3、穩健性 seed 4/5。每 seed 的選參數規則相同，半數玩家的指標在對應 lock 存好後才輸出。

- [ ] **Step 1: 寫失敗測試**：
  - `test_alpha_zero_exactly_preserves_baseline`：每題每候選分數與 rank 完全不變；constant new-score row 標準化為零。
  - `test_same_seed_all_candidates_exclude_outer_held_games`：所有候選含負候選都沒有 held key；任何跨 seed vector 來源不符報錯。
  - `test_test_half_cannot_change_its_selected_grid`：修改被評 half 的 query scores/labels，不得改變它所用的 selection；另一方向可正常受影響。
  - `test_other_seed_never_enters_selection`：只改另一 seed，當前 seed 的 lock／選擇不變。
  - `test_size_balancing_and_tie_order`：n=1 三題不變成三倍權重；平手選低 α、高 λ。
  - `test_bootstrap_reselects_and_shares_owner_weights_across_seeds`：小合成資料手算一組重抽 weights，結果須與重選 grid 相同，且不同於只 bootstrap 已選 delta；同人所有 seed 共用權重。
  - `test_query_duplicate_is_not_a_new_cluster`：同人同棋跨 seed 不新增 cluster；pub2/player1 與 priv2/player1 不合併。
  - `test_missing_component_or_t50_fails_instead_of_fallback`：缺檔、鍵、成分或 NaN 都必須報錯。
- [ ] **Step 2: Slurm 跑 failing tests，確認新介面缺失**；實作數學純評估 helpers，再重跑到通過。
- [ ] **Step 3: 實作 candidate-only baseline adapter**，按 `fuse_v` 同公式與權重重算；原始 keys 只在對應 pool 內查既有向量，研究輸出使用 namespace。對全部 0–3 種子重現 785974 的兩池單局 mean rank-score/Top-1 至 1e-9；存輸入文件與 feature hashes。失敗不可往下試新分數。
- [ ] **Step 4: 實作 R/A0/A1/A2/A2perm score grid 與鎖定 cross-fit**。norm/Fbar 只來自校準訓練資料；A0 的 fixed-F 與 A1 的真 F 除此之外公式相同。處理一 pool×seed 後釋放大矩陣，只存 GridScores/per-question ranks，不將全數 [Q,C]×grid 寫磁碟。
- [ ] **Step 5: 實作 5,000 次重新選參數的 bootstrap**，保存 raw owner grid、seed、draw count、CI 的條件範圍；報各 pool/題型、救回／改錯、被評 owner 數與獨立棋局數。
- [ ] **Step 6: 合成端到端測試通過後只提交 Task 4 檔案**。此步尚不依正式結果改 spec/grid。

### Task 5：有界執行、三態判決與研究交付

**Files:**
- Create: `scripts/nano4/pid_choice_probe.sh`
- Extend: `tests/test_choice_pipeline.py`
- Update only after actual run: `METHOD_NOTES.md`
- Output: `results/final/choice_probe_<run_id>_{audit,selection,eval}.json`

**Interfaces:**
- shell usage：`pid_choice_probe.sh check|pilot|calibration|extract|baseline-check|explore|evaluate|robustness RUN_DIR`；每次只執行一階段，驗前置成功 manifest，不自動跳階。
- `budget.json` 存 run 的全部 job IDs、allocated GPUs、elapsed allocation seconds、已用 GPUh、目前 active allocation、原始／暫存 bytes 與 quota。不能只用 GPU kernel 時間計費；pending 不計 GPUh。
- report verdict：`input_invalid / approximation_invalid / approximation_inconclusive / budget_exceeded / promising / below_practical_threshold / inconclusive`；只有最後三者是效果判決，前四者不是「訊號不存在」。

- [ ] **Step 1: 補 workflow 測試**：quota/budget不足拒絕下一階段；changed manifest 不 resume；父工作 COMPLETED 但 checksum 缺失仍拒絕；query CSV 被意外讀取會失敗；watcher 退出不取消既有 o26。
- [ ] **Step 2: 實作 shell stage wrapper 與 dry-run**，檢查 source hashes、run lock、上一步產物及剩餘預算；每分鐘只記本 run quota／job 狀態，不修改其他監看。
- [ ] **Step 3: 納入 Task 3 已完成 pilot／校準的 job 與耗用，Slurm 執行必要測試後，以已量測 ETA 提交剩餘抽取／評估階段**。已驗證 shard 不重跑。reference audit 隨本階段實際要評估的 seed 執行，4/5 留到穩健性階段。提交命令從 R1 根目錄，保留排除故障節點；每個 sbatch 回傳 job ID 必須記 ledger，不能把送件成功當實驗成功。

```bash
# 時限／GPU 數在 pilot 後依 spec §8 計算；以下僅為 pilot 範例。
sbatch --parsable --export=ALL \
  --exclude='25a-hgpn[001,003-010,039,062,069,155,159,177]' \
  -p 8gpus --gres=gpu:1 -c 12 -t 00:30:00 -J pid_choice_pilot \
  scripts/nano4/run.sbatch bash ../BK_AICUP_2026-2/scripts/nano4/pid_choice_probe.sh pilot RUN_DIR
```

- [ ] **Step 4: 讀主要 seed 2/3 的固定程序結果**。A2−B ≥+0.002、CI下界>0且兩池各正才進 seed4/5；若 CI上界<+0.002 記此版本未達實用幅度；其他記 inconclusive，不再掃 grid。A2−A2perm 未過時即使 A2 有用也不得宣稱已確認情境交互訊號。
- [ ] **Step 5: 有希望且 budget 足夠才執行 seed 4/5**，同 seed cross-fit 重做、演算法/grid不改；仍明記重複玩家／棋局。預算不足記未完成穩健性，不宣稱已通過。
- [ ] **Step 6: 核對 Slurm terminal state、各 shard coverage、輸出與基準 hash**；將實測耗時、peak disk、效果/區間/判決、失敗版本與來源寫 METHOD_NOTES。不得新增 upload CSV 或變更定案。
- [ ] **Step 7: 提交本研究程式與小型證據摘要**；如需完整原型保留或接 B/C，依已記錄判決另行規劃，不把這一輪低維探針冒充大型新網路已完成。

## 計畫自查與執行前狀態

- 本輪完成的只有：現存實驗／程式／結果去重、spec、實作計畫與獨立設計審查。以上 checkbox 全部未執行。
- Spec §2 的舊路線均有明確對照或排除，尤其 P6 **已通過且在用**、PLDA **已做且無融合增益**。
- 新訊號只歸因於實際選項 covariance 或經 A2perm 證實的情境對應，不把重做殘差、更多參數或更好的校正當成新資訊。
- 介面、33/66 維、校準／官方 key namespace、seeds、λ/α grid 與判決門檻已對齊 spec；選參數與 bootstrap 的資料角色在 Task 4 有測試。
- 執行前需以 worktree 隔離現有髒工作區；只提交本計畫產物與新研究檔，保留原 session 的所有未提交變更。
