#!/usr/bin/env python
"""A fusion document for pseudo-pool / old-pool READINGS whose new families have no real-pool vectors on /home yet
(METHOD_NOTES 2.201; /home has no room for them before a candidate has passed).

As data/day_scripts/build_v2h.py for families of my official networks (tags <family>s<k>e), with one difference: a family
named after the composition counts the members that have a linked checkpoint (/work/u2507115/pid_v2/seqnet/
ckpt_-1_<family>s<k>e.pt) instead of the members that have a real-pool vector file. The readings only use the file NAMES of
`seq_official` (scripts/pid_v2_offsim_par.py, experiments/pid_v2_oldsim.py); a submission file needs the real-pool vectors and
a document built by build_v2h.py.

  pid_build_doc_names.py <name> <family>[*k][+<family>[*k]...] [<new family> ...]
"""
import json  # 讀寫融合文件（JSON）
import re  # 檢查家族名稱的格式
import sys  # 讀取命令列參數，錯誤時結束程式
from pathlib import Path  # 以路徑物件處理檔案位置

R, SN = Path("/home/u2507115/BK_AICUP_2026-2"), Path("/work/u2507115/pid_v2/seqnet")  # 叢集上的儲存庫根目錄與序列網路檢查點資料夾
name, spec, new = sys.argv[1], sys.argv[2].split("+"), set(sys.argv[3:])  # 文件名稱、組成（以 + 分開的 家族[*比重]）、新家族
ref = json.loads((R / "results/pid_fusion_v2ge.json").read_text())  # 參考融合文件 v2ge：權重等其餘欄位沿用它
F = lambda tag: [str(R / f"data/seq_ft/official_{pool}_net-1{tag}_t50.npz") for pool in ("pub2", "priv2")]  # noqa: E731  一個網路在 pub2、priv2 兩個新增池微調 50 步後的向量檔路徑
groups = {}  # 家族 → 各成員的向量檔
for g in {x.split("*")[0] for x in spec}:  # 逐一處理組成中的家族（去掉比重）
    if not re.fullmatch(r"o\d\w*", g):  # 只接受自己官方網路的家族名稱（o 加數字開頭）
        sys.exit(f"{g}: only families of my official networks")  # 名稱不符就中止
    if g in new:  # 新家族：真題池向量可能還不存在
        groups[g] = [F(f"_{g}s{k}e") for k in range(24) if (SN / f"ckpt_-1_{g}s{k}e.pt").exists()]  # 依有連結的檢查點計算成員
    else:  # 既有家族
        groups[g] = [F(f"_{g}s{k}e") for k in range(24) if Path(F(f"_{g}s{k}e")[0]).exists()]  # 依 pub2 的真題池向量檔是否存在計算成員
    if len(groups[g]) < 7:  # 少於 7 個成員的家族不收
        sys.exit(f"family {g}: only {len(groups[g])} members ({'linked checkpoints' if g in new else 'real-pool vectors'})")  # 中止並說明依據
files = [x for g in spec for _ in range(int(g.split("*")[1]) if "*" in g else 1) for x in groups[g.split("*")[0]]]  # 依比重重複各家族成員的向量檔，組成 seq_official
doc = {**ref, "seq_official": files, "arm": "ft50", "composition": spec,  # 以 v2ge 為底，換上新組成的向量檔清單
       "seq_model": f"composition {'+'.join(spec)} ({len(files)} networks), each after 50 steps of the pool fine-tune; mean of the row-standardised matrices; weights of v2ge",  # 記錄組成、網路數與做法
       "note": "pseudo-pool / old-pool reading only (scripts/pid_build_doc_names.py): the real-pool vector files of "  # 註明只供擬題池／舊題池讀數
               + (" ".join(sorted(new & set(groups))) or "-") + " may not exist"}  # 列出可能缺真題池向量的新家族
(R / f"results/pid_fusion_{name}.json").write_text(json.dumps(doc, indent=1) + "\n")  # 寫出融合文件
print(name, spec, "networks", len(files), "| members:", " ".join(f"{g}={len(v)}" for g, v in sorted(groups.items())))  # 回報網路總數與各家族的成員數
