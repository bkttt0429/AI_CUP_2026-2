#!/usr/bin/env python
"""External game records for the identity metric (METHOD_NOTES 2.46): the Fox Go Dataset (github.com/featurecat/go-dataset,
GPL-3.0) -> compact parts, and the count-only overlap gate against the competition's training games.

  --stage ingest --archives <dir with the .7z / .7z.NNN files of one rank folder> --out <dir>
      Every SGF is unpacked to the node's RAM disk, read as data (nothing in it is executed), removed again, and reduced to
      what the competition's own records hold - board size, handicap, the moves - plus the fields used only for filtering
      and grouping: the two ACCOUNT KEYS (a salted hash of PB / PW; the names themselves are never written), ranks, year,
      time settings, result type. Written as <out>/games_NNNN.npz.
  --stage overlap --parts <dirs of ingest outputs ...> --out <json>
      The gate: how many competition training games (fold 0 excluded; test sets never read) have the same first --prefix
      moves as an external game, up to the eight board symmetries. COUNTS ONLY: no account key or name is read for a match.
      Any overlap means the competition data and this source share games: stop and report, do not train on the source.

Moves are stored as uint16 codes colour * 512 + (19 * x + y), pass = 361; a game key is the archive's file name.
Run with python -I (the archives are untrusted data).
"""
import argparse  # 解析命令列參數
import hashlib  # blake2b 雜湊：帳號鍵與開局雜湊
import json  # 寫出重疊閘門的計數結果
import os  # 產生鹽與取得行程編號
import re  # 解析 SGF 的屬性與落子
import shutil  # 刪除解壓用的暫存資料夾
import sys  # 以 main 的回傳值結束
import time  # 記錄經過的秒數
from pathlib import Path  # 路徑處理

import numpy as np  # 精簡分段檔的陣列與 npz 檔

MOVE = re.compile(rb";\s*([BW])\[([a-s]{2})?\]")  # SGF 裡的一手：顏色與座標（空括號為虛手）
PROP = re.compile(rb"([A-Z]{1,2})\[([^\]]*)\]")  # 棋譜檔頭的屬性，例如 PB[...]、SZ[19]
SYMS = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y),  # 棋盤八種對稱的前四種：恆等、兩種鏡射與轉 180 度
        lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]  # 後四種：先轉置，再做同樣的四種變換


def moves_of(raw):  # 取出 SGF 主線的落子序列
    """[(colour 0/1, point 0-360 or 361)] of an SGF's main line (bytes)."""
    out = []  # 收集每一手
    for c, p in MOVE.findall(raw):  # 依序走過主線每一手
        out.append((0 if c == b"B" else 1, 361 if not p else 19 * (p[0] - 97) + (p[1] - 97)))  # 黑 0 白 1；沒有座標記為虛手 361
    return out  # 回傳落子清單


def prefix_hash(mv, k):  # 開局前 k 手的對稱不變雜湊
    """uint64 of the first k moves, the smallest over the eight symmetries; None for a shorter game or a pass inside."""
    if len(mv) < k or any(p == 361 for _, p in mv[:k]):  # 不足 k 手或前段含虛手
        return None  # 不參與比對
    best = None  # 目前最小的位元組序列
    for f in SYMS:  # 逐一套用八種對稱
        b = bytes(v for c, p in mv[:k] for v in (c, *f(p // 19, p % 19)))  # 顏色與對稱後的座標排成位元組
        best = b if best is None or b < best else best  # 保留最小者，使結果與棋盤方向無關
    return int.from_bytes(hashlib.blake2b(best, digest_size=8).digest(), "little")  # 8 位元組雜湊轉成整數


def rank_code(b):  # 段位文字轉成整數代碼
    """Fox rank text -> int (kyu negative: 18k = -18 ... 1k = -1; dan 1-9; pro 10+; unknown 0)."""
    s = b.decode("utf-8", "replace")  # 位元組轉文字，無法解碼的字元以替代字元表示
    m = re.match(r"(\d+)\s*(级|級|k|K)", s)  # 級位的各種寫法
    if m:  # 是級位
        return -int(m.group(1))  # 級位記為負數
    m = re.match(r"(\d+)\s*(段|d|D)", s)  # 業餘段位的寫法
    if m:  # 是段位
        return int(m.group(1))  # 段位記為正數
    m = re.match(r"[Pp职職]?(\d+)\s*[Pp段]?", s)  # 職業段位的寫法
    return 10 + int(m.group(1)) if m and ("P" in s or "p" in s or "职" in s or "職" in s) else 0  # 職業記為 10 以上，其餘為不明 0


def ingest(a):  # 匯入階段：壓縮檔 -> 精簡分段檔
    import py7zr  # 讀 7z 壓縮檔（只有匯入階段需要）
    t0 = time.time()  # 計時起點
    out = Path(a.out)  # 輸出資料夾
    out.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    salt_file = Path(a.salt_file)  # 帳號雜湊用的鹽檔（不進儲存庫）
    if not salt_file.exists():  # 第一次執行時還沒有鹽
        salt_file.write_bytes(os.urandom(32))  # 產生 32 位元組的隨機鹽
        salt_file.chmod(0o600)  # 只有自己可讀寫
    salt = salt_file.read_bytes()  # 讀入鹽
    key_of = lambda name: int.from_bytes(hashlib.blake2b(name, key=salt, digest_size=8).digest(), "little")  # noqa: E731  帳號名換成加鹽雜湊整數，名稱不寫出
    files = sorted(Path(a.archives).iterdir())  # 這個段位資料夾的所有檔案（依名稱排序，下面只取 7z 檔）
    groups = {}  # 壓縮檔名 -> 它的各個分卷
    for f in files:  # x.7z -> 一個壓縮檔；x.7z.001、x.7z.002 ... -> 拆成分卷的一個壓縮檔
        m = re.fullmatch(r"(.+\.7z)(\.\d+)?", f.name)  # 取出不含分卷編號的壓縮檔名
        if m:  # 只處理 7z 檔
            groups.setdefault(m.group(1), []).append(f)  # 同一個壓縮檔的分卷歸成一組
    part, n_all, n_bad = 0, 0, 0  # 分段檔編號、保留局數、略過局數
    for name, vols in groups.items():  # 逐一處理每個壓縮檔
        src = vols[0]  # 預設直接讀單一檔
        if len(vols) > 1 or vols[0].name != name:  # 把分卷合併（直接串接）放在輸出旁邊，之後刪除
            src = out / f"_{name}"  # 合併檔以底線開頭
            with open(src, "wb") as w:  # 寫出合併檔
                for v in sorted(vols):  # 依分卷編號順序
                    w.write(v.read_bytes())  # 逐卷串接
        tmp = Path(a.tmp) / f"extfox_{os.getpid()}_{name}"  # 節點的 RAM 磁碟：五十萬個小檔不寫到網路檔案系統
        try:  # 不論成功與否都要清掉暫存資料夾
            with py7zr.SevenZipFile(src) as z:  # 開啟壓縮檔
                names = [n for n in z.getnames() if n.endswith(".sgf")]  # 只取 SGF 棋譜
                bad = [n for n in z.getnames() if n.startswith("/") or ".." in n.split("/")]  # 會寫到目標資料夾之外的項目
                if bad:  # 壓縮檔內容不安全
                    raise ValueError(f"{name}: entries leave the target folder, e.g. {bad[:3]}")  # 拒絕解壓
                tmp.mkdir(parents=True)  # 建立暫存資料夾
                z.extractall(tmp)  # 解壓到 RAM 磁碟
            print(f"{name}: {len(names)} records unpacked to {tmp} ({time.time() - t0:.0f} s)", flush=True)  # 回報解壓的棋譜數
            for i in range(0, len(names), a.chunk):  # 每 --chunk 份棋譜寫成一個分段檔
                rec = {k: [] for k in ("key", "pb", "pw", "br", "wr", "year", "ha", "tm", "tc", "tt", "res", "len", "off")}  # 這個分段檔各欄位的清單
                mv_all, off = [], 0  # 全部落子的串接與目前的起點
                for n in names[i:i + a.chunk]:  # 逐局處理
                    raw = (tmp / n).read_bytes()  # 以位元組讀入棋譜（只當資料）
                    head = dict(PROP.findall(re.split(rb";\s*[BW]\[", raw, maxsplit=1)[0]))  # 第一手之前的檔頭屬性
                    mv = moves_of(raw)  # 主線落子
                    if head.get(b"SZ", b"19") != b"19" or len(mv) < a.min_moves or b"PB" not in head or b"PW" not in head:  # 不是 19 路、太短或缺帳號
                        n_bad += 1  # 計入略過
                        continue  # 不保留這局
                    num = lambda k: int(head[k]) if head.get(k, b"").lstrip(b"-").isdigit() else -1  # noqa: E731  讀整數屬性，缺少或不是數字時為 -1
                    re_ = head.get(b"RE", b"")  # 對局結果
                    rec["key"].append(n); rec["pb"].append(key_of(head[b"PB"])); rec["pw"].append(key_of(head[b"PW"]))  # 棋局鍵與黑白雙方的帳號鍵
                    rec["br"].append(rank_code(head.get(b"BR", b""))); rec["wr"].append(rank_code(head.get(b"WR", b"")))  # 黑白雙方的段位
                    rec["year"].append(int(head[b"DT"][:4]) if head.get(b"DT", b"")[:4].isdigit() else 0)  # 對局年份，不明為 0
                    rec["ha"].append(min(num(b"HA"), 99)); rec["tm"].append(num(b"TM")); rec["tc"].append(min(num(b"TC"), 30000)); rec["tt"].append(min(num(b"TT"), 30000))  # 讓子與時間設定，截斷以放進整數型別
                    rec["res"].append(1 if b"+R" in re_ else 2 if b"+T" in re_ else 3 if b"+" in re_ else 0)  # 認輸 / 超時 / 計點 / 其他
                    rec["len"].append(len(mv)); rec["off"].append(off)  # 手數與在落子陣列中的起點
                    mv_all.extend(c * 512 + p for c, p in mv)  # 落子編成 uint16 代碼
                    off += len(mv)  # 下一局的起點
                np.savez(out / f"games_{part:04d}.npz", key=np.array(rec["key"]), pb=np.array(rec["pb"], np.uint64), pw=np.array(rec["pw"], np.uint64), br=np.array(rec["br"], np.int8),  # 寫出分段檔：棋局鍵、帳號鍵、黑方段位
                         wr=np.array(rec["wr"], np.int8), year=np.array(rec["year"], np.int16), ha=np.array(rec["ha"], np.int8), tm=np.array(rec["tm"], np.int32), tc=np.array(rec["tc"], np.int16),  # 白方段位、年份、讓子與時間設定
                         tt=np.array(rec["tt"], np.int16), res=np.array(rec["res"], np.int8), len=np.array(rec["len"], np.int32), off=np.array(rec["off"], np.int64), moves=np.array(mv_all, np.uint16))  # 結果、手數、起點與全部落子
                part += 1  # 下一個分段檔編號
                n_all += len(rec["key"])  # 累計保留局數
                print(f"  part {part}: {n_all} games kept, {n_bad} left out ({time.time() - t0:.0f} s)", flush=True)  # 回報進度
        finally:  # 無論成功或失敗
            shutil.rmtree(tmp, ignore_errors=True)  # 刪除解壓出來的棋譜
        if src.name.startswith("_"):  # 這是暫時合併的分卷檔
            src.unlink()  # 刪除合併檔
    print(f"ingest done: {n_all} games in {part} parts, {n_bad} left out (not 19x19, under {a.min_moves} moves, or no account) ({time.time() - t0:.0f} s)")  # 匯入總結


def overlap(a):  # 重疊閘門：只計數的比對
    import multiprocessing as mp  # 平行計算競賽棋局的開局雜湊

    import pandas as pd  # 讀分折檔與競賽訓練 CSV
    t0 = time.time()  # 計時起點
    ext, n_ext = set(), 0  # 外部開局雜湊的集合與外部局數
    for d in a.parts:  # 逐一讀匯入輸出的資料夾
        for f in sorted(Path(d).glob("games_*.npz")):  # 這個資料夾的每個分段檔
            z = np.load(f)  # 讀分段檔
            mv, off, ln = z["moves"], z["off"], z["len"]  # 只用落子、起點與手數（不讀帳號鍵）
            for o, n in zip(off, ln):  # 逐局
                h = prefix_hash([(int(v) // 512, int(v) % 512) for v in mv[o:o + a.prefix]], a.prefix) if n >= a.prefix else None  # 解碼前 --prefix 手並算雜湊
                n_ext += 1  # 累計外部局數
                if h is not None:  # 有可比對的開局
                    ext.add(h)  # 加入集合
        print(f"external {d}: {n_ext} games so far, {len(ext)} distinct prefixes ({time.time() - t0:.0f} s)", flush=True)  # 回報進度
    folds = pd.read_csv(Path(a.aicup1) / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]  # 競賽玩家所屬的折
    res = {"prefix_moves": a.prefix, "external_games": n_ext, "external_distinct_prefixes": len(ext), "by_file": {}, "note": "counts only; fold 0 excluded; test sets not read; no account read for a match"}  # 結果：設定、外部統計與說明
    tot = hit = 0  # 競賽局數與命中數
    for f in sorted((Path(a.aicup1) / "data/training").glob("train_*.csv")):  # 逐一讀各級別的訓練檔
        c = pd.read_csv(f, usecols=["player_id", "sgf_content"], dtype=str)  # 只讀玩家編號與棋譜
        c = c[c.player_id.map(folds).fillna(-1) != 0]  # 排除第 0 折的玩家
        with mp.get_context("fork").Pool(a.workers) as pool:  # 多行程計算雜湊
            hs = pool.map(_h, [(s, a.prefix) for s in c.sgf_content], chunksize=2000)  # 每局的開局雜湊
        hs = [h for h in hs if h is not None]  # 去掉無法比對的局
        m = sum(h in ext for h in hs)  # 與外部開局相同的局數
        res["by_file"][f.name] = {"games_with_prefix": len(hs), "matches": int(m)}  # 依檔案記錄計數
        tot += len(hs); hit += m  # 累計
        print(f"{f.name}: {len(hs)} games with {a.prefix} moves, {m} with an external game of the same first {a.prefix} moves ({time.time() - t0:.0f} s)", flush=True)  # 回報這個檔的結果
    res.update(competition_games=tot, matches=int(hit))  # 總計
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")  # 寫出 JSON
    print(f"OVERLAP: {hit} of {tot} competition training games match one of {len(ext)} external prefixes -> {'STOP: shared games' if hit else 'no shared game in these parts'}; wrote {a.out}")  # 總結：有命中就要停下來檢視


def _h(x):  # 行程池的工作函式：一局棋譜的開局雜湊
    return prefix_hash(moves_of(x[0].encode()), x[1])  # 解析棋譜文字後算雜湊


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--stage", required=True, choices=("ingest", "overlap"))  # 執行哪個階段
    ap.add_argument("--archives")  # 匯入：壓縮檔所在的資料夾
    ap.add_argument("--parts", nargs="*", default=[])  # 閘門：匯入輸出的資料夾
    ap.add_argument("--out", required=True)  # 輸出資料夾或 JSON 檔
    ap.add_argument("--salt-file", default="/work/u2507115/ext_fox/account_salt.bin", help="local secret for the account keys; never in the repository")  # 帳號雜湊的鹽檔
    ap.add_argument("--tmp", default="/dev/shm", help="where an archive is unpacked while it is read (removed afterwards)")  # 解壓位置（RAM 磁碟）
    ap.add_argument("--chunk", type=int, default=50000)  # 每個分段檔最多放幾份棋譜
    ap.add_argument("--min-moves", type=int, default=30)  # 保留的最少手數
    ap.add_argument("--prefix", type=int, default=60)  # 閘門比對的開局手數
    ap.add_argument("--aicup1", default=str(Path.home() / "BK_AICUP_2026-1"))  # AI_CUP_2026-1 儲存庫（訓練資料與分折檔）
    ap.add_argument("--workers", type=int, default=12)  # 計算雜湊的行程數
    a = ap.parse_args(argv)  # 解析參數
    return ingest(a) if a.stage == "ingest" else overlap(a)  # 依階段執行


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值結束
