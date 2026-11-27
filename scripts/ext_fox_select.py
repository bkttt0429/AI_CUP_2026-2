#!/usr/bin/env python
"""External players for the identity metric (METHOD_NOTES 2.46): which accounts and which of their games.
Input: the compact parts of scripts/ext_fox_ingest.py (one directory per rank folder of the Fox Go Dataset). An account key
is a salted hash; no name exists in these files.
  eligible games   19x19 (ingest), no handicap, at least --min-moves moves, no pass inside the first 256 plies
  accounts         at least --need eligible games in the folder and not among the --cap-pct busiest (bots, shared accounts);
                   --accounts of them per folder, drawn at random
  games            --games per account, each game given to ONE of its chosen accounts (the one still missing more), so a
                   record is extracted once and has one subject, as a competition record has
Output <out>/select.csv: game_key (x:<folder>:<record>), account (hex key), color (the subject's), sgf_content in the
competition's own form "(;SZ[19]HD[0];B[..];W[..]...)" cut after 256 plies; <out>/select.json: the counts.
"""
import argparse  # 解析命令列參數
import csv  # 讀先前的選取檔、寫出 select.csv
import json  # 寫出 select.json 的計數
import sys  # 設定匯入路徑與結束碼
import time  # 記錄經過的秒數
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算與亂數

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的 ext_fox_ingest 可以匯入
from ext_fox_ingest import moves_of  # noqa: E402  用閘門同一個解析器驗證寫出的棋譜

T_MAX = 256  # 每局最多使用的手數（與競賽的逐手特徵相同）
ABC = "abcdefghijklmnopqrs"  # SGF 的座標字母


def sgf_of(codes):  # 落子代碼轉成競賽格式的棋譜文字
    return "(;SZ[19]HD[0]" + "".join(f";{'BW'[c // 512]}[{ABC[(c % 512) // 19]}{ABC[(c % 512) % 19]}]" for c in codes) + ")"  # 19 路、無讓子，逐手寫出顏色與座標


def load(d):  # 讀一個資料夾的全部分段檔
    """One folder's parts as arrays in memory (an .npz re-reads an array at every access)."""
    Z = []  # 每個分段檔一個字典
    for f in sorted(Path(d).glob("games_*.npz")):  # 依編號讀分段檔
        with np.load(f) as z:  # 開啟分段檔
            Z.append({k: z[k] for k in ("pb", "pw", "ha", "len", "br", "wr", "year", "moves", "off", "key")})  # 一次讀入需要的欄位
    return Z  # 回傳分段檔清單


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--parts", nargs="+", required=True, help="name=dir of one rank folder's parts, e.g. 6d=/work/.../parts_6d")  # 各段位資料夾的名稱與分段檔位置
    ap.add_argument("--accounts", type=int, default=7500, help="per folder; with --joint: in total")  # 要選的帳號數（每個資料夾；--joint 時為總數）
    ap.add_argument("--games", type=int, default=20)  # 每個帳號取幾局
    ap.add_argument("--need", type=int, default=30)  # 帳號至少要有的合格局數
    ap.add_argument("--min-moves", type=int, default=80)  # 合格對局的最少手數
    ap.add_argument("--cap-pct", type=float, default=0.5)  # 排除局數最多的前百分之幾帳號（機器人、共用帳號）
    ap.add_argument("--joint", action="store_true", help="count an account's games over ALL folders and choose once: an account met in two folders (its rank changed) stays one player")  # 跨資料夾合併計數、只選一次
    ap.add_argument("--by-account", action="store_true", help="write the rows account by account (accounts in random order): any prefix of the file holds whole accounts")  # 依帳號成群寫出，檔案的任何前段都只含完整的帳號
    ap.add_argument("--skip", nargs="*", default=[], help="select.csv files of earlier selections: their accounts and their games are left out")  # 先前的選取檔，避免重複選取
    ap.add_argument("--deepen", action="store_true", help="choose among the accounts of --skip instead of leaving them out: MORE games of accounts that were selected before "  # 改為加深先前選過的帳號
                                                          "(their earlier games stay left out; METHOD_NOTES 2.126)")  # 說明的後半：先前用過的棋局仍然排除
    ap.add_argument("--seed", type=int, default=0)  # 亂數種子
    ap.add_argument("--out", required=True)  # 輸出資料夾
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    rng = np.random.default_rng(a.seed)  # 可重現的亂數產生器
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    skip_acc, skip_games = set(), set()  # 先前選過的帳號與棋局
    csv.field_size_limit(10 ** 9)  # 棋譜欄位很長，放寬 CSV 欄位上限
    for f in a.skip:  # 逐一讀先前的選取檔
        with open(f, newline="") as fh:  # 開啟選取檔
            for r in csv.DictReader(fh):  # 逐列讀取
                skip_acc.add(int(r["account"], 16)); skip_games.add(r["game_key"])  # 記下帳號鍵與棋局鍵
    summary, n_rows = {"args": vars(a), "folders": {}, "skipped_accounts": len(skip_acc), "skipped_games": len(skip_games)}, 0  # 摘要與寫出的總局數
    names = [x.split("=", 1)[0] for x in a.parts]  # 資料夾名稱
    dirs = dict(x.split("=", 1) for x in a.parts)  # 名稱 -> 分段檔位置
    groups = [names] if a.joint else [[n] for n in names]  # --joint 時全部資料夾一起選，否則逐一資料夾選
    with open(out / "select.csv", "w", newline="") as fh:  # 寫出選取檔
        w = csv.writer(fh)  # CSV 寫入器
        w.writerow(["game_key", "account", "color", "sgf_content"])  # 表頭
        for grp in groups:  # 每組選一次
            name = "+".join(grp)  # 組名（例如 3d+1d）
            Z, zname = [], []  # 分段檔與各分段檔所屬的資料夾
            for n in grp:  # 讀入這組的每個資料夾
                zz = load(dirs[n])  # 讀這個資料夾的分段檔
                Z += zz; zname += [n] * len(zz)  # 串接並記下資料夾名稱
            pb, pw = np.concatenate([z["pb"] for z in Z]), np.concatenate([z["pw"] for z in Z])  # 黑白雙方的帳號鍵
            ha, ln = np.concatenate([z["ha"] for z in Z]), np.concatenate([z["len"] for z in Z])  # 讓子數與手數
            br, wr, year = np.concatenate([z["br"] for z in Z]), np.concatenate([z["wr"] for z in Z]), np.concatenate([z["year"] for z in Z])  # 段位與年份（只用於摘要）
            part = np.concatenate([np.full(len(z["len"]), i) for i, z in enumerate(Z)])  # 每局所在的分段檔編號
            local = np.concatenate([np.arange(len(z["len"])) for z in Z])  # 每局在分段檔內的索引
            ok = (ha <= 0) & (ln >= a.min_moves)  # 無讓子且手數足夠
            for i, z in enumerate(Z):  # 要使用的那些手裡不可有虛手
                is_pass = (z["moves"] % 512) == 361  # 標出每一手是否為虛手
                cs = np.concatenate([[0], np.cumsum(is_pass)])  # 虛手數的前綴和
                end = z["off"] + np.minimum(z["len"], T_MAX)  # 每局使用到的結束位置
                ok[part == i] &= (cs[end] - cs[z["off"]]) == 0  # 前 256 手內有虛手就不合格
            idx = np.flatnonzero(ok)  # 合格對局的索引
            acc, cnt = np.unique(np.concatenate([pb[idx], pw[idx]]), return_counts=True)  # 每個帳號的合格局數
            cap = np.percentile(cnt, 100 - a.cap_pct)  # 最活躍帳號的局數門檻
            pool = acc[(cnt >= a.need) & (cnt <= cap)]  # 局數夠、又不是最活躍的帳號
            if a.deepen:  # 加深模式
                pool = pool[np.isin(pool, np.fromiter(skip_acc, np.uint64, len(skip_acc)))]  # 只留先前選過的帳號
            else:  # 一般模式
                pool = pool[~np.isin(pool, np.fromiter(skip_acc, np.uint64, len(skip_acc)))] if skip_acc else pool  # 排除先前選過的帳號
            chosen = rng.choice(pool, min(a.accounts, len(pool)), replace=False)  # 隨機抽出帳號
            need = dict.fromkeys(chosen.tolist(), a.games)  # 每個帳號還缺幾局
            got = dict.fromkeys(chosen.tolist(), 0)  # 每個帳號已拿到幾局
            rows, ranks, years, left, held = 0, [], [], len(chosen) * a.games, []  # 計數、摘要用清單、剩餘名額、暫存的列
            for g in rng.permutation(idx):  # 以隨機順序走過合格對局
                if left <= 0:  # 名額已滿
                    break  # 結束
                b, wkey = int(pb[g]), int(pw[g])  # 這局的黑白帳號
                nb, nw = need.get(b, 0), need.get(wkey, 0)  # 雙方各還缺幾局（未被選中者為 0）
                if nb <= 0 and nw <= 0:  # 雙方都不需要
                    continue  # 略過這局
                z = Z[part[g]]  # 這局所在的分段檔
                key = f"x:{zname[part[g]]}:{Path(str(z['key'][local[g]])).stem}"  # 棋局鍵 x:<資料夾>:<棋譜>
                if key in skip_games:  # 先前已選過這局
                    continue  # 不重複使用
                side = 0 if nb >= nw else 1  # 給還缺比較多局的一方
                who = b if side == 0 else wkey  # 這局的主體帳號
                need[who] -= 1  # 少缺一局
                got[who] += 1  # 多拿一局
                left -= 1  # 剩餘名額減一
                o = int(z["off"][local[g]])  # 落子的起點
                codes = z["moves"][o:o + min(int(ln[g]), T_MAX)].tolist()  # 前 256 手的落子代碼
                text = sgf_of(codes)  # 轉成競賽格式
                if rows < 200:  # 寫出的棋譜讀回來是同樣的落子（重疊閘門與重播都解析這個格式）
                    assert [c * 512 + q for c, q in moves_of(text.encode())] == codes, "record does not read back"  # 抽驗前 200 局
                if a.by_account:  # 依帳號成群寫出
                    held.append((who, key, "BW"[side], text))  # 先暫存
                else:  # 依抽到的順序寫出
                    w.writerow([key, f"{who:016x}", "BW"[side], text])  # 直接寫一列
                rows += 1  # 寫出的局數
                ranks.append(int(br[g] if side == 0 else wr[g])); years.append(int(year[g]))  # 記下主體段位與年份
            if held:  # 有暫存的列
                order = {int(k): i for i, k in enumerate(rng.permutation(chosen))}  # 帳號的隨機順序
                held.sort(key=lambda r: order[r[0]])  # 依帳號排序，同一帳號的列相鄰
                for who, key, col, text in held:  # 逐列寫出
                    w.writerow([key, f"{who:016x}", col, text])  # 寫一列
                del held  # 釋放記憶體
            g_ = np.array(list(got.values()))  # 每個帳號拿到的局數
            summary["folders"][name] = {"games_in_parts": int(len(ok)), "eligible_games": int(len(idx)), "accounts_with_eligible_games": int(len(acc)), "accounts_meeting_need": int(len(pool)),  # 這組的計數：局數、合格局數、帳號數
                                        "busiest_cut_at_games": float(cap), "accounts_chosen": int(len(chosen)), "games_written": rows, "accounts_with_8_or_more": int((g_ >= 8).sum()),  # 活躍門檻、選中帳號與寫出局數
                                        "accounts_with_all_games": int((g_ >= a.games).sum()), "subject_rank_counts": {str(k): int(v) for k, v in zip(*np.unique(ranks, return_counts=True))},  # 拿滿的帳號數與主體段位分布
                                        "year_counts": {str(k): int(v) for k, v in zip(*np.unique(years, return_counts=True))}}  # 年份分布
            n_rows += rows  # 累計總局數
            s_ = summary["folders"][name]  # 這組摘要的簡寫
            print(f"{name}: {s_['games_in_parts']} games, {s_['eligible_games']} eligible; accounts {s_['accounts_with_eligible_games']}, with >= {a.need} games (earlier selections left out) {s_['accounts_meeting_need']}; "  # 回報這組的結果
                  f"chosen {s_['accounts_chosen']} -> {rows} games ({s_['accounts_with_all_games']} accounts with all {a.games}, {s_['accounts_with_8_or_more']} with >= 8); subject ranks {s_['subject_rank_counts']} ({time.time() - t0:.0f} s)", flush=True)  # 回報的後半
            del Z  # 釋放這組的分段檔
    summary["games_written"] = n_rows  # 寫出的總局數
    (out / "select.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")  # 寫出摘要
    print(f"wrote {out / 'select.csv'}: {n_rows} games ({time.time() - t0:.0f} s)")  # 總結


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值結束
