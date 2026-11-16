# 新訊號研究設計：先驗證「選擇機會」與「情境偏好」

日期：2026-11-16。這是執行前鎖定的設計，以下原規則保留。實作已完成；二階版本因reference數值稽核失敗停止，未評估辨識效果。精確計分後續另見`../plans/2026-11-16-choice-exact-followup.md`，不可把後續結果當作原試驗通過。實際執行證據與結論見METHOD_NOTES及results/final/choice_probe_2993db0b3ed8_*。

使用者先要求從瓶頸研究新訊號，接受「個人選擇模型優先、反應動態另查、候選分歧後接」的研究順序，再要求「開始仔細規劃且看看之前是否有做過」。本文件把第一輪縮成可否證的機制探針，不直接建大型個人化 policy 網路。數值、預算與實作選擇是本次規劃建議，不能寫成使用者逐項指定。

## 1. 已知瓶頸與本輪問題

現行定案為 `upload/pid_v2bgm_d2_a2k_p3.csv`，Public 0.9181331。診斷 785974 已完成，`results/final/v2_probe_bottleneck_v2bgm_d2.json`：兩個新增擬題池單局 Top-1 0.54861／0.56806、Top-5 0.81250／0.82222；失分的 52.10%／52.57% 來自正解第 2–5 名。這是候選棋譜留局診斷，並非官方隱藏測試正確率。

本輪只問：**在既有 P6 特徵之外，同一局面各可選落點的條件共變異，以及偏好與局勢的交互作用，是否提供互補辨識證據？** 多個相近模型失敗不能證明原始棋譜資訊已耗盡；本探針失敗也不能否定所有個人 policy 架構。

## 2. 既有工作稽核：筆記、實作、結果交叉確認

範圍：目前兩倉庫的可讀程式、玩家辨識 METHOD_NOTES、results/final 與相關啟動腳本。沒有查到不等於歷史上絕對沒有人做過；未保留的未記錄實驗不在證據範圍。下表路徑皆相對本倉庫，R1 表示相鄰 `BK_AICUP_2026-1`。

| 提案容易重複的部分 | 已有證據 | 結論與本次處理 |
| --- | --- | --- |
| 實際落點減背景預期 | §2.102／2.107；`scripts/pid_pol_extract.py`、`pid_probe_polrows.py`；`results/final/v2_sim14p.json`、`v2_probe_p_fuse.json` | 已做。p2/p2b 單獨 +0.0140/+0.0079，但融合區間含 0。只能列對照，不能報成新訊號。 |
| 同時放入全盤、落點、預期選項及 policy 純量 | §2.110；`pid_sup_fit.py --blocks g x e`、`pid_sup_ply_extract.py --scalars`；`v2_sim14p6.json`、`v2_probe_p6_fuse.json` | **已做且通過，已在第二套 js 輸入中。** p6s/p6sb 融合 +0.0138/+0.0121（當時較弱基準）。不得再說現行沒有「未走選項的預期」。 |
| 加實際選擇跨手標準差 | `pid_probe_polrows.py` 的 Gs/Xs/Es/Ds；`v2_probe_polrows.json` | 已做。這是跨實際局面的變異；不是同一局面對所有可選動作的 conditional covariance。新增量必須明確區別。 |
| 原型不確定性、共變異、貝葉斯比對 | §2.44；`pid_v2_sim17.py::plda_fit/plda_scores`；`v2_sim17.json` | 已做。取代餘弦 −0.0177，當附加成分約 0。PLDA 是局向量的玩家間／玩家內變異；本次 F 是單一棋盤上未走選項的分布，不重跑 PLDA。 |
| 棋局不同階段 | §2.34–2.36；`pid_v2_phase_rows.py`；`v2_sim9.json`、`v2_sim10.json` | 已做，弱配置曾正向，強配置／Public 未保住相同幅度。僅序中後盤分桶不列新方向。 |
| 對手資訊與上一手回應 | §2.7／2.38／2.117；`src/aicup_pid/local.py`、`pid_probe_opp.py` | 局部位移回應已在 local；對手身分／重複對手未帶來實用融合增益。B 要測事件後多回合條件反應，不能重做對手向量或距離統計。 |
| 勝率、領先與短期變化 | §2.115–2.116；`pid_val_extract.py`；`v2_sim14p7*.json` | 已做，P7 融合為負。A2 的新假說是價值與動作偏好的乘積交互作用；單獨增加價值純量只能是舊對照。 |
| 逐手／逐局配對、困難負例、pair network | §2.56／2.145／2.149／2.151；`pid_v2_latematch.py`、`pid_seq_train.py`、`pid_pair_train.py`；對應結果保留 | 已做，完整融合未通過。C 必須依可選動作的預測分歧定位證據，不能把 top-k 重排本身當新方法。 |
| 對照組校正、熱門候選人偏差 | §2.139／2.154；`v2_probe_asnorm_v2bgm_d2.json` | 已做，約 +0.001、區間含 0。本計畫不再掃欄正規化。 |
| 候選人特定、帶情境的動作分布／conditional Fisher | 檢視上述 extractor、scorer、訓練介面，並搜尋 policy/residual/Fisher/conditional 等 | 尚未找到同等實作或結果。只將「每局面選項 covariance＋情境交互」標成待驗證的新部分。 |

額外工程發現：`pid_trunk_extract.policy_vectors` 只回傳 g/x/e 與五個純量，沒有保存完整空間主幹或選項二階矩；pass 時 x/e 都設為零，其 mask 是空點／非劫點近似。不能從現成 js 壓縮檔精確還原新增 F，必須小量重播。不能把舊 g/x/e 的樣本共變異冒充 F。

## 3. 考慮過的路線與取捨

1. **直接建立可少樣本適應的個人 policy 網路**：最貼近長期方向，但須重新學泛化與少樣本適應，訓練／存檔成本高，且可能重現 pair 記人問題。延後。
2. **凍結主幹的低維選擇模型探針（本輪）**：可逐項加入真正新增的統計量，先驗證訊號存在及互補性，磁碟可控；二階近似及既有投影會限制能力，必須量近似誤差。
3. **事件後反應與動態狀態模型**：較不同，但事件對齊、棋力混淆與有效樣本數尚未量清。另列 B 的先導設計，不與 A 合併訓練，也不拿 A 陰性當 B 的判決。

採用第 2 路線作第一輪；C（候選分歧選手）要等 A 有可校準的個人模型才有意義。

## 4. A 探針的數學定義與固定實驗臂

背景 p0 是凍結 humanv0、`neutral_rows()`、policy channel 0，保留合法盤上動作與 pass，softmax 使用 FP32。重播沿用 `trunk_extract.setup_and_moves/start_state/RULES`；局面須為落子前局面，取前 256 個棋局 ply 中主角自己的回合。沒有用時或勝負標籤；模型估計價值不得稱真實賽果。

投影來源 `/work/u2507115/pid_v2/probep/proj3.npz`：執行前要求 blocks 恰為 g/x/e、fit_folds 為 3/4、W 為 1152×64、sd 為 1152；取 x 區塊前 32 維 `R = W[384:768,:32] / sd[384:768,None]`。不符合就停止並回報，不能偷偷改投影。它是已訓練的線性投影，非原始完整動作表示，陰性只限制此版本。

- 盤上動作 ψ(a) = `[trunk(a) @ R, 0]`；pass ψ = `[0×32, 1]`，共 33 維。
- 用預先選定 1,000 局 fold 3/4 訓練棋的 policy-weighted covariance 對角均值決定各維尺度 `sqrt(max(mean_var, 1e-6))`；均值精確定義為 `sum(F0_raw)/sum(n)` 的對角，各回合等權；縮放後的這個 33 維矩陣就是 A0 的固定 Fbar。全程固定，不用官方題池重估。先存未縮放 U/F，校準完成後於評分時做 `U/d`、`F/(d dᵀ)`；66 維使用重複的 33 維尺度。這樣前 256 局試跑可直接納入校準，不必重抽。
- A2 的 φ(s,a) = `[ψ(a), v(s) ψ(a)]`，66 維；v 是主角下一手之前 humanv0 的 win−loss 機率，範圍 [−1,1]。head 已是 side-to-move 視角，主角執白也不可再反號。A0/A1 用 ψ，33 維。第一輪不加 entropy gating、更多 phase、額外投影維度或多個價值模型。
- 每局存 U = Σ(φ_played−E_p0 φ)、F = Σ Cov_p0(φ)、n = 有效主角回合數。F 在 FP32 累積後對稱化，輸出不做 int8 量化。F 不是跨局樣本共變異。
- 候選參數 `z = solve(F_ref + λ n_ref I, U_ref)`；λ 候選只用 {0.1, 1, 10}。題目分數 `U_queryᵀz − 0.5 zᵀF_query z`，多局的 U/F 相加。這是帶收縮的二階 MAP 近似，**不是完整貝葉斯 posterior predictive**，不使用棋局回合數宣稱獨立樣本數。

| 臂 | 定義 | 用途 |
| --- | --- | --- |
| R | 33 維 U/n 的候選／題目餘弦 | 同次抽取的殘差均值對照，不宣稱逐位元重現歷史 p2。 |
| A0 | F_ref/F_query 分別換成 n_ref/n_query × 訓練集固定 Fbar；其他同 A1 | 區分平均背景幾何與每個局面的實際選項幾何。 |
| A1 | 33 維實際 U/F 的二階分數 | 量每局面選項 covariance 的增益。 |
| A2 | 66 維情境交互作用的實際 U/F | **唯一主要候選**；其他臂是機制對照，不能看完確認結果改選其中一個宣稱本試驗通過。 |
| A2perm | 同 A2 的 66 維，但同局內在 ply <60、60–149、150–255 各區間固定置換 v 與當下動作的對應 | 同維度、同局勢值邊際分布的負對照；使用 `sha256('choice-vperm-v1|'+pool+'|'+key+'|'+phase)` 前 8 bytes little-endian 為 NumPy default_rng 種子。少於 2 手不置換。不用不同 shuffle 種子挑結果。只作機制診斷，不視為能預測未來的 policy。 |

所有臂均與完整定案 B 比較 `B + α standardise(S_arm)`，α 僅 {0, 0.05, 0.1, 0.2, 0.4}；α=0 必須逐項重現 B。所有 600 候選都計分，不把正解是否在 top-k 用作選樣條件。主要問題是 A2 對 B；A1−A0、A2−A1、A2−R 解釋機制。要聲稱情境對應帶來新訊號，另要求 A2−A2perm 的聚類 CI 下界 >0。A2perm 同樣在相同 cross-fit 設定以固定 grid 選 λ/α，不能讓負對照承受較差的調參待遇。

## 5. 輸入正確性與近似有效性是先決條件

- 新 replay 必須持有棋盤狀態，使用現有 `Board.would_be_legal` 的逐點規則判斷（與 `aicup.humansl` 參考路徑一致），固定 pass 可選。已讀 R1 `repo/KataGo/python/katago/game/board.py`：此 Python 版本明確不支援 superko；沿用這個已知限制，metadata 必須記 `legality=board_simple_ko_no_superko`，不宣稱是未知官方規則的完美還原。不能直接重用空點近似 mask。實際手不合法或重播失敗時該局標記失敗，正式評估缺一個應有 key 就停止；不得只在新方法中跳掉難局。
- 手工棋譜測占點、提子、劫、自殺、pass、讓子、主角執白、無主角回合、兩個題池相同原始 key。查詢實際落子之前的 v/φ/合法選項不得含後續棋步。
- 新舊 extractor 在各自共用的盤上合法點定義下，g/x/e 可做對齊驗證；pass 與不同 mask 的差異另記，不能要求假等價。
- 抽 1,024 個校準棋局的固定 hash 位置，僅這些保存 p0/ψ/v 供 exact log-normalizer 驗證。位置分散到不同棋局；記錄抽樣 manifest，不依正確率挑位置。
- 對每個 λ，用校準玩家分開的 reference games 估 z，與 exact `log p_z(a|s)−log p0(a|s)` 比。每手絕對誤差中位數 ≤0.02 nat、95 百分位 ≤0.10 nat，且有限值／F 最小特徵值數值檢查通過，才容許該 λ 進選擇表。A1/A2/A2perm 分別檢查，取共同通過的 λ；**A0/R 只驗有限值及公式，A0 刻意替換 F，不應被原始 policy 的 Taylor gate 剔除。** 全部 λ 不通過即標記 approximation_invalid 並停；不能以此宣稱訊號不存在，也不能事後用查詢答案調 clipping 救回。
- 官方 reference-only 稽核：對每個 pool×seed 及通過校準的 λ，只用該 seed 的 reference 估 A1/A2/A2perm 的 z。每臂各取 ||z|| 最大的 16 人加 hash 最前的 16 位其他人；每人抽自己 reference 的一局（key hash 最小）及另一位候選人的 reference 一局（owner/key hash 最小且不同人），各取 ply hash 最前的 8 個主角回合。直接重播這些 reference，檢查 exact 與二階分數的逐手誤差及成段差：每手 median/p95 沿用 0.02/0.10 nat；對同一 8 手 block 的 32 位候選 exact 分數做列標準化後，以該人與其餘 31 位候選人分別比較，標準化分差 |Δ|≥0.05 時，exact/二階分數的正負順序一致率須 ≥95%。exact 分數 std<1e-12 的 block 記無辨識力，不用除法放大雜訊。至少 100 個 block×candidate-pair 達到此 margin 才可宣稱排序近似已驗證（不是 100 個獨立樣本）；不足標記 approximation_inconclusive。使用標準化 margin 是為了不讓高 λ 的純分數縮放被誤當失敗。任一臂／λ 失敗則該次 pool×seed 研究停，不再剔 λ 或換模型；不能根據官方資料反覆篩掉不利設定。稽核不讀 outer query，也不改候選參數。這是有限範圍數值稽核，不是全部局面均校準的保證；C 仍需獨立評估。
- 儲存每局 U/F/n 而非全盤主幹。需要重新抽取新增的 F，不覆寫任何既有 npz 或訓練輸出。

## 6. 資料、參數選擇與不誇大證據

1. 校準資料：`training_games` 與 `splits/player_folds.csv` 合併；只用 fold 3/4，先以主角至少 10 回合的棋局篩出至少 10 局的玩家，再依 `sha256('choice-player-v1|' + player_id)` 選 100 人，各依 `sha256('choice-v1|' + game_key)` 選 10 局，共 1,000 局。固定每人前 7 局供近似審查的 reference，後 3 局供 held-out audit；不夠 100 人直接回報，不改抽法。抽取排程按全體 key hash 排序，256 局 pilot 是其中前 256 局。這批用於尺度、Fbar、吞吐與近似審查，不用來宣稱未見玩家準確率。
2. 官方新增候選棋：各池 600×10 局、只讀 candidate CSV，**不讀 official question CSV、答案或 submission**。保存 key 為 `(pool, original_key)`，同一池維持原 CSV row 順序以相容現有 holdout。
3. 7 參考／3 留局完全沿用 `v2.candidate_holdout(h=3,seed)`。題型 n=1（三局各一題）、n=2（前兩局一題）、n=3（三局一題）。兩池永遠分開估候選模型與校準參數。
4. 現行 B 需每個 seed 的 **reference-only t50** 向量 `offsim_{pool}_h3s{seed}_..._t50.npz`；不能改用看過全部 10 局的 official t50。重用 `fuse_v` 的公式與原權重，資料載入不重用會載官方 query 且容許 fallback 的 `Pool`；缺檔／缺成分直接停止。先重現 785974 的 seed 0–3 單局分數至 1e-9（同 FP64 評分公式），否則不得繼續比較。
5. 每池 600 人按 `sha256('choice-half-v1|' + pool + '|' + player_id)` 排序，前 300／後 300 固定為 H0/H1；所有 seed 共用這個分組。**每一 outer seed 獨立做雙向 cross-fit**：只用該 seed 的 H0 query 選 (λ,α)，評該 seed 的 H1 query；反向亦然。所有 600 位候選人的模型與基準 FT 都只能使用該 seed 的 7 局 reference；不得把別的 seed 的 reference/query 帶入選參數。候選庫始終為全部 600 人。僅不使用測試半的 query 標籤仍不夠：那些局連負候選的參考模型也不能進入。
6. 選擇指標為 n=1/2/3 等權平均的官方 rank score，平手優先較小 α，再較大 λ。每臂各自選值僅供對照；A2 是事前唯一主要候選。R 只選 α。每個 pool×seed×被評 half 各有自己的 lock JSON，鎖定選參數資料 hash 後才計被評 half 的指標。seed 0/1 作機制探索、2/3 作凍結程序的主要評估；跨 seed 凍結的是 grid／演算法／主要臂，**不是把另一 seed 所選數值搬來**。
7. 初判有希望再用 seed 4/5 驗穩健性，每個 seed 仍以同一事先鎖定的 cross-fit 程序選參數，不能讀跨 seed 的 reference 或評估結果。這些玩家及棋局與其他 seed 重疊，**只能稱對留局切分穩健，不是新獨立資料集**；舊基準已用過的種子亦如實記錄。
8. 指標按 pool×half 分層、以 owner 為單位 bootstrap 5,000 次，同玩家所有種子／重複棋局／題型共用重抽權重；兩池最後等權合併。保存每個固定 grid 的 per-owner／題型得分，**每次 bootstrap 在來源 half 重選 λ/α，再於另一 half 的重抽 owner 上評分**，而非只重抽已選結果。不重跑主幹。區間涵蓋這個 grid 選擇的變異，但仍條件於固定候選庫、已訓練基準與投影，不能稱新玩家母體的完整不確定性。除了平均，報 n 分層、兩池、救回／改錯數、正解原排名層、按顏色與手數，不靠其中最佳子集採用。

主要前進門檻：A2−B 的等權總分 ≥+0.002，玩家聚類 95% CI 下界 >0，兩池各自差值 >0，seed 4/5 同方向。只有較弱模型變好或只有 R/A0 變好，不能宣稱新增 F／交互機制成立。

判決分三類：滿足門檻為 promising；95% CI 上界 <+0.002 為此版本未達實用幅度；其餘為 inconclusive。inconclusive 不加權重、不換提交、不無限重抽到顯著；保留可偵測幅度與區間，有限預算用盡即報研究未分出來。通過也只是值得下一階段：舊題池 1/2/3/5 局確認與完整生成式模型另立計畫，**本輪不產出提交 CSV**。

## 7. 工程邊界、介面與產物

新增隔離模組 `src/aicup_pid/choice_signal.py`（數學）及 `choice_eval.py`（cross-fit／bootstrap）；新增 `scripts/pid_choice_manifest.py`、`pid_choice_extract.py`、`pid_choice_eval.py`、`scripts/nano4/pid_choice_probe.sh`、三個 test 檔。不修改現行 trainer／extractor／fusion registry，不建立新的虛擬環境。

- manifest：schema_version、run_id、pool、original_key、owner、color、SGF hash、split source、part IDs；包含 humanv0/proj3/fusion/source hashes 與真實路徑。與檔案無關的 RNG 使用固定 salt，不能用不穩定的 Python hash。
- shard npz：namespaced keys、U0[N,33]、F0[N,33,33]、U2[N,66]、F2[N,66,66]、U2perm[N,66]、F2perm[N,66,66]、n[N]、failures；metadata 另 JSON；FP32。相同來源 manifest 才可 resume，先 tmp+fsync，再 rename；existing 不同 hash 拒絕覆寫。13,000 局的這些原始陣列約 500 MiB；temp、稽核與 grid 摘要亦計入 1 GiB，不能把未壓縮峰值漏算。
- 中間目錄：`/work/u2507115/pid_v2/choice_probe_20261116/<run_id>`，run_id 含設定 manifest SHA 前 12 字。結果 `results/final/choice_probe_<run_id>_{audit,selection,eval}.json`。只保存 per-question rank/delta 與必要摘要，不存所有網路與所有臂的全分數大矩陣。
- 每一步寫成功 manifest 後才排下游；Slurm afterok 不代替產物、checksum、缺鍵檢查。錯誤／逾時保留成功 shard，重送明確同 run_id 續算。禁止靜默補零或 fallback。

## 8. 成本與時間回報

- 上限：同時 **2 GPU**；第一輪含檢查的總配額 **6 GPU 小時**、新增磁碟 **1 GiB**。這是研究上限，不是執行時間承諾。只用 production Slurm，R1 既有 venv/run.sbatch。
- 檢查與256局吞吐試跑：1 GPU、30 分鐘時限；256 局為校準清單前 256 局，包含在 1,000 局內，不重算。
- 啟動大批前：依試跑實測有效局／秒、位置／秒、bytes／局、replay 與 GPU 耗時計算剩餘抽取時間及 peak disk；CPU 使用量另列。正式抽取最多 2 GPU，12 CPU/GPU，排程時限採預估時間×1.5＋15分鐘且不超剩餘總 budget；不能用加長 time limit 超越 budget。
- quota 門檻：`/work/u2507115` 可用至少 **6 GiB**、`/home/u2507115` 至少 **3 GiB**，且預估本輪 peak <1 GiB；不足即停在已驗證 shard，不清別的實驗。檢查 symlink/realpath；不碰 o26、m161r、choice64 的資料和輸出。
- 以每階段結束回報「實際耗時、剩餘上限、開始/完成時間範圍（排隊另計）」。在測吞吐之前不報幾點出新訊號結果。

## 9. B 與 C 的後續界線

**B 事件反應**：獨立先導，尚不實作。選事件前要鎖定「前一個對手回合的背景 surprisal」等不含主角未來動作的定義；觀察後續第 1/2/4/8 個主角回合的偏好殘差／區域轉換。以同階段、顏色、價值與難度配對事件對照；打亂事件對齊但保留無序分布作負對照，玩家分組驗證、事件邊界不足者明確 censor。先量每人／每局有效事件數與跨局穩定性，資料不足就不能訓練大動態模型。這是可觀察的反應差，不是情緒或因果心理診斷。

**C 候選分歧**：A2 有融合增益、近似校準可用後才規劃。僅依參考棋與落子前局面比較候選 policy，選高分歧且低參數不確定性的位置，再讀實際落子。對照全部手、隨機等量、背景 entropy；同時報 top-k 外的覆蓋上限與原正確題改錯。若只得到近似 U/F 而沒有可靠的逐局面 policy，不得直接以 z 距離冒充 action-distribution 分歧。它是證據／算力分配，不是新的獨立資訊來源。

## 10. 文獻定位

沿用 §2.160 已核對的 [Maia4All](https://arxiv.org/html/2507.21488v1)、[Fisher kernel](https://proceedings.neurips.cc/paper/1998/hash/db1915052d15f7815c8b88e879465a1e-Abstract.html)、[ToMnet](https://proceedings.mlr.press/v80/rabinowitz18a.html) 作方法靈感。前者是西洋棋個人化、後者是 gridworld 行為模型，都不是本資料有效性的證明。本輪的主要依據仍是既有實驗去重與可直接驗證的新增統計量。
