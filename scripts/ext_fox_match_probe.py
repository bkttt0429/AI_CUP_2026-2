#!/usr/bin/env python
"""What a hit of the overlap gate is (METHOD_NOTES 2.50): for every competition training game (fold 0 excluded, test sets
never read) whose first --prefix moves equal an external game's up to a board symmetry, describe the pair WITHOUT any account:
which rank file and which external folder, both lengths, how many moves agree from the start (under the symmetry that aligns
the prefix) and whether the whole record is the same, the external record's year and the two ranks written in it, and how many
external records share that prefix. Account keys are not read."""
import argparse  # 解析命令列參數
import json  # 寫出配對描述
import sys  # 設定匯入路徑與結束碼
from pathlib import Path  # 路徑處理

import numpy as np  # 讀精簡分段檔
import pandas as pd  # 讀分折檔與競賽訓練 CSV

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的 ext_fox_ingest 可以匯入
from ext_fox_ingest import SYMS, moves_of, prefix_hash  # noqa: E402  沿用閘門的對稱、解析與雜湊，結果才一致


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--parts", nargs="+", required=True, help="name=dir")  # 外部資料夾名稱與它的精簡分段檔位置
    ap.add_argument("--prefix", type=int, default=60)  # 與閘門相同的開局手數
    ap.add_argument("--aicup1", default=str(Path.home() / "BK_AICUP_2026-1"))  # AI_CUP_2026-1 儲存庫（訓練資料與分折檔）
    ap.add_argument("--out", required=True)  # 輸出的 JSON 檔
    a = ap.parse_args(argv)  # 解析參數
    ext = {}  # 開局雜湊 -> 共用這個開局的外部棋局
    for spec in a.parts:  # 逐一處理外部資料夾
        name, d = spec.split("=", 1)  # 拆出名稱與路徑
        for f in sorted(Path(d).glob("games_*.npz")):  # 這個資料夾的每個分段檔
            with np.load(f) as z:  # 讀分段檔
                mv, off, ln, year, br, wr, ha = z["moves"], z["off"], z["len"], z["year"], z["br"], z["wr"], z["ha"]  # 只取落子與描述用的欄位，不讀帳號鍵
            for i, (o, n) in enumerate(zip(off, ln)):  # 逐局
                if n >= a.prefix:  # 夠長才能比對開局
                    h = prefix_hash([(int(v) // 512, int(v) % 512) for v in mv[o:o + a.prefix]], a.prefix)  # 開局的對稱不變雜湊
                    if h is not None:  # 開局內沒有虛手
                        ext.setdefault(h, []).append((name, [(int(v) // 512, int(v) % 512) for v in mv[o:o + n]], int(year[i]), int(br[i]), int(wr[i]), int(ha[i])))  # 記下整局落子與描述欄位
        print(f"{name}: {len(ext)} distinct prefixes so far", flush=True)  # 回報進度
    folds = pd.read_csv(Path(a.aicup1) / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]  # 競賽玩家所屬的折
    out = []  # 收集每一對的描述
    for f in sorted((Path(a.aicup1) / "data/training").glob("train_*.csv")):  # 逐一讀各級別的訓練檔
        c = pd.read_csv(f, usecols=["player_id", "sgf_content"], dtype=str)  # 只讀玩家編號與棋譜
        c = c[c.player_id.map(folds).fillna(-1) != 0]  # 排除第 0 折的玩家
        for s in c.sgf_content:  # 逐局
            mv = moves_of(s.encode())  # 解析競賽棋局的落子
            h = prefix_hash(mv, a.prefix)  # 開局雜湊
            if h is None or h not in ext:  # 沒有命中
                continue  # 下一局
            for name, em, year, br, wr, ha in ext[h]:  # 命中的每一局外部棋局
                best = 0  # 從開頭起最多相同的手數
                for sym in SYMS:  # 逐一試八種對稱，找出使兩份棋譜對齊的那一個
                    t = [(c_, 361 if p == 361 else 19 * sym(p // 19, p % 19)[0] + sym(p // 19, p % 19)[1]) for c_, p in em]  # 外部棋局套用這個對稱
                    k = 0  # 相同手數的計數
                    while k < min(len(t), len(mv)) and t[k] == mv[k]:  # 從頭比到第一個不同的手
                        k += 1  # 多一手相同
                    best = max(best, k)  # 取各對稱中最長者
                out.append({"competition_file": f.name, "competition_moves": len(mv), "external_folder": name, "external_moves": len(em), "moves_equal_from_start": best,  # 這一對的描述：檔名、雙方手數、相同手數
                            "whole_record_equal": bool(best == len(mv) == len(em)), "external_year": year, "external_ranks_black_white": [br, wr], "external_handicap": ha,  # 是否整局相同與外部棋譜的年份、段位、讓子
                            "external_records_with_this_prefix": len(ext[h])})  # 共用這個開局的外部棋譜數
                print(out[-1], flush=True)  # 立即印出這一對
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")  # 寫出 JSON
    print(f"{len(out)} pair(s); wrote {a.out}")  # 總結


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值結束
