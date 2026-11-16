# 精確條件選擇分數：近似失敗後的獨立後續

2026-11-16 17:50。使用者已授權依瓶頸證據決定方法；這是實作中的方法修正，由執行者決定，並非使用者指定數學細節。

## 前次結果與界線

原試驗2993db0b3ed8的二階版本判決為`approximation_invalid`。seed2公共A2 λ1 p95=.100226nat、排序一致率=.948613；私人A2 p95=.135216nat、一致率=.921883，低於事前門檻。λ10雖通過，不能事後刪λ1來宣稱原試驗有效。效果工作786587已取消，沒有讀任何新方法的留局辨識指標。失敗報告原樣保留。

## 選擇與數學

考慮：(1)只留λ10，會事後改grid，拒絕；(2)訓練新大模型，無法直接解釋這次失敗，延後；(3)保留同一個候選參數估計器，直接算完整動作正規化，採用。

保留凍結humanv0/proj3、33/66維特徵、reference-only `z=solve(F+λnI,U)`，以及已由校準固定的λ=[1,10]。z現在明確稱「以二階目標估計的參數」，不稱精確MAP。動作模型則完整定義為 `p_z(a|s) ∝ p0(a|s)exp(phi(s,a)·z)`，分數直接累積 `phi_played·z - logsumexp(logp0+phi·z) + logsumexp(logp0)`，不再用Taylor分數。FP64分塊運算，全部600人都算。

R與A0公式、grid保持原樣；A1/A2/A2perm換精確分數。主要候選仍A2；A2perm仍需相同選參數待遇。A1−A0現在同時改變選項幾何與normalizer，僅為混合對照，不能歸因為純conditional covariance增益；A2−A2perm仍是同維度且同精確公式的情境對應對照。相同seed內cross-fit、5000次重選參數bootstrap、原採用門檻、4/5各池各seed同方向要求皆不變。這是看過reference數值稽核後的新方法版本；不能冒稱原預註冊試驗通過或獨立新資料驗證。

## 實作與驗證

1. 新增`choice_exact.py`，測試多候選分塊分數對獨立scalar logsumexp、零z、共同feature偏移不變、pass/非法mask、強tilt有限。先RED再GREEN，Slurm。
2. 新增`pid_choice_exact.py`：每pool一次讀取完整已校驗moments，分seed只用7ref估z；重播2/3 heldout的聯集，但只對該seed的held棋計分，所有候選的z嚴格同seed來源。每64局保存可續算精確證據，namespaced keys、seed、600owner次序、arm/ridge和code/protocol hash全部驗證。先32/64局pilot，推算預算才續跑。
3. 精確版本獨立protocol含原runid、原失敗audit hashes、固定grid、code hashes。原manifest/extractor/choice_signal.py不動；原失敗audit不覆寫。逐局檢查有限；pilot實際首局各arm/seed的FP64批次結果對獨立逐候選exact_delta，誤差<=1e-8。對照前先在FP64把logp重新正規化，使既有exact_delta省略的logsumexp(logp)=0，避免把原FP32殘差誤當公式錯誤。只驗算術正確，不宣稱統計校準。
4. 評估器明確`--method exact`，只讀完整精確證據與匹配provenance；不得假造Taylor audit通過。輸出加exact前綴，保留失敗版本。R/A0可用既有moments，其餘對每題的各局exact分數相加。
5. 全部測試與獨立審查後，先primary2/3。若達初判門檻才用4/5同法重播，仍在原總6GPUh/2GPU/1GiB限制內。超出則停止並報未完成，不把缺失當陰性。

## 成本

前次約0.7GPUh、552MB。兩池seed2/3聯集約6120局，精確證據FP64最多約208MB；保留原moments及baseline，總預估<850MB。新pilot估算後才排大批；新job全部計同一jobs.txt，沒有重置預算。

若需要4/5，先以實測大小重新預估新增兩池證據、4/5baseline與暫存峰值；接近1GiB時不啟動，或先歸檔已完成2/3的grid/ranks/selection與完整來源清單後，只清理本輪可重算exact-score快取，記錄所移除檔案hash。不得清理原moments、其他實驗或尚未成功評估的證據。

獨立code review未發現科學正確性缺陷；指出中斷在NPZ rename與JSON commit之間會安全拒絕resume。此版source/protocol已凍結，不為非阻斷問題重寫版本。若實際遇到，先確認該part沒有成功sidecar，將僅該孤立NPZ移至本run的quarantine（仍計磁碟），再同protocol重算並核對陣列；不把孤立檔標為成功，也不刪其他part。既有tmp同樣不讀取且計磁碟。完整已commit shard可直接resume。
