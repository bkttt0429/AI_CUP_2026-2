#!/usr/bin/env python
"""The whole added-pool pipeline scored on OFFICIAL players: labelled pseudo-pools cut from each official pool's own candidates
(METHOD_NOTES 2.41). The organisers give the owner of every candidate game (600 players x 10 games; candidate games may be
used for calibration, Q&A Q10). Per pool and split seed, aicup_pid.v2.candidate_holdout takes h = 3 games of every candidate
out: the other 7 are the candidate's reference games, the 3 make its questions
  n = 1: each held-out game (1,800 questions)   n = 2: the first two (600)   n = 3: all three (600)
The sequence networks are fine-tuned on the reference games only (scripts/pid_v2_seq_finetune.py --holdout 3), so question
games are as unseen as official question games. Everything else is the official prediction path (predict_pid_v2.predict_pool:
the fusion document's components, embedding model and weights). Pools are never mixed: weights are fitted and scored inside
one pool. Differences from the real pools: 7 reference games instead of 10, no 5-game questions.

This file is experiments/pid_v2_offsim.py with the splits (pool x seed) as separate processes (METHOD_NOTES 2.190): the original
computes its eight splits one after the other in one process (about 75 minutes for four seeds), which is too slow for the
evening's verdict.  --only POOL:SEED --dump DIR  computes one split and writes it to DIR (no statistics);
--from-dumps --dump DIR  reads the splits of --seeds from DIR and prints / writes exactly what the original does (the random
stream of the statistics is the same: one permutation per pool, then the bootstraps).

Reported per pool and for both: the score by question size and their mean ("mix") for the fine-tune steps asked for, with
the document's weights; paired differences between steps (bootstrap over candidates); and the value of calibrating the
fusion weights on the pool's own candidates: weights fitted on the questions of one half of the candidates, scored on the
other half, both ways (fit set: one 1-game question per candidate plus its 2- and 3-game questions).
"""
import argparse  # 解析命令列參數
import json  # 讀融合文件
import pickle  # 切分結果的存檔格式
import re  # 從向量檔名取出網路編號
import sys  # 設定 sys.path 與結束
import time  # 計時
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列運算

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
for d in (ROOT / "src", ROOT / "scripts"):  # 本儲存庫的套件與腳本
    if str(d) not in sys.path:  # 尚未加入時
        sys.path.insert(0, str(d))  # 加到搜尋路徑最前面
import aicup_pid  # noqa: E402,F401  匯入時設定 AI_CUP_2026-1 的路徑
from aicup.ladder import to_json  # noqa: E402  輸出 JSON 的格式化工具
from aicup.submission import check_game  # noqa: E402  判斷棋局能否使用（19 路、能合法重播、該方至少一手）
from aicup_pid import v2  # noqa: E402  新增池的成分與留局切分
from aicup_pid.components import parse_all  # noqa: E402  平行解析棋譜
from aicup_pid.embed import PlayerEmbedding  # noqa: E402  每局嵌入成分的模型
from aicup_pid.fusion import Fusion, standardise  # noqa: E402  融合權重與逐列標準化
from aicup_pid.trunkstyle import load_store  # noqa: E402  讀每局主幹列

SIZES = (1, 2, 3)  # 擬題池的題目局數（沒有 5 局題）


def per_question(F, truth):  # 每題的官方分數
    rank = (F > F[np.arange(len(truth)), truth][:, None]).sum(1)  # 分數高於正解的候選人數（0 起算的名次）
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)  # 前五名得 e^-名次，否則 0 分


def mix(sc, n, sel=None):  # 各局數分數的平均（「mix」）
    sel = np.ones(len(n), bool) if sel is None else sel  # 預設全部題目
    return float(np.mean([sc[sel & (n == m)].mean() for m in SIZES]))  # 每種局數等權平均


def boot_mix(d, n, cluster, B, rng):  # 成對差的 bootstrap 區間
    """Bootstrap over clusters of the size-balanced mean of d."""
    ids, inv = np.unique(cluster, return_inverse=True)  # 群集編號與每題所屬的群集
    num = np.stack([np.bincount(inv, weights=d * (n == m), minlength=len(ids)) for m in SIZES])  # [3, K] 各群集各局數的差之和
    den = np.stack([np.bincount(inv, weights=(n == m).astype(float), minlength=len(ids)) for m in SIZES])  # 各群集各局數的題數
    pick = rng.integers(0, len(ids), (B, len(ids)))  # B 次有放回地抽群集
    bs = (num[:, pick].sum(2) / den[:, pick].sum(2)).mean(0)  # 每次抽樣的局數平衡平均
    return {"diff": float((num.sum(1) / den.sum(1)).mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}  # 點估計與 95% 區間


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))  # 官方 v2 檔所在資料夾
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2ge.json"))  # 要評估的融合文件
    ap.add_argument("--ft-dir", default="/work/u2507115/pid_v2/seq_ft_offsim")  # 擬題池微調向量的資料夾
    ap.add_argument("--holdout", type=int, default=3)  # 每位候選人留出幾局當題目
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])  # 切分種子
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50, 100])  # 要比較的微調步數
    ap.add_argument("--alt", default=None, help="a second fusion document (another network composition): scored at --alt-step next to the first, paired difference")  # 第二份融合文件（另一種網路組成）
    ap.add_argument("--alt-step", type=int, default=50)  # 第二份文件使用的微調步數
    ap.add_argument("--mix", nargs="*", type=int, default=[], help="an extra arm 'mix': every network's vectors after each of these step counts, all averaged (e.g. 50 100)")  # 多個步數的向量一起平均的額外一臂
    ap.add_argument("--boot", type=int, default=2000)  # bootstrap 次數
    ap.add_argument("--workers", type=int, default=12)  # 解析棋譜的行程數
    ap.add_argument("--out", required=True)  # 結果 JSON
    ap.add_argument("--only", default=None, help="POOL:SEED - compute this one split and write it to --dump; no statistics, --out is not written")  # 只算一個切分（平行版）
    ap.add_argument("--dump", default=None, help="the directory of the split files")  # 切分檔的資料夾
    ap.add_argument("--from-dumps", action="store_true", help="read the splits of --seeds from --dump instead of computing them")  # 讀回已算好的切分再做統計
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    import torch  # 只為載入嵌入模型，延後匯入
    T, rng = Path(a.tests_dir), np.random.default_rng(0)  # 官方檔資料夾；統計用的固定亂數流
    doc = json.loads(Path(a.fusion).read_text())  # 讀融合文件
    comps = list(doc["components"])  # 成分清單
    assert comps[-1] == "seq" and isinstance(doc["emb_model"], str)  # 逐手成分必須在最後，且要有嵌入模型
    only = (a.only.split(":")[0], int(a.only.split(":")[1])) if a.only else None  # 解析 POOL:SEED
    assert not (only and a.from_dumps) and (a.dump or not (only or a.from_dumps))  # 兩種平行模式互斥，且都需要 --dump
    if not a.from_dumps:  # 要自己算切分時才載入模型與列
        emb = PlayerEmbedding.from_state(torch.load(str(ROOT / doc["emb_model"]), map_location="cpu", weights_only=False))  # 文件指定的每局嵌入模型
        store = load_store(str(ROOT / doc["trunk_store"]))  # 新增池的每局主幹列
    nets_of = lambda d_: [re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", Path(x[0]).name).groups() for x in d_["seq_official"]]  # noqa: E731  文件中每個網路的 (net, tag)
    nets = nets_of(doc)  # 主文件的網路清單
    alt = json.loads(Path(a.alt).read_text()) if a.alt else None  # 第二份文件（可以沒有）
    assert alt is None or list(alt["components"]) == comps  # 兩份文件的成分必須相同
    print(f"document {Path(a.fusion).name}: {len(nets)} sequence networks, weights {dict(zip(comps, np.round(doc['weights'], 3).tolist()))}", flush=True)  # 回報文件內容
    load = lambda f: (lambda z: dict(zip(z["key"].astype(str), z["emb"].astype(np.float32))))(np.load(f))  # noqa: E731  讀一個向量檔成 {鍵: 向量}
    runs, half = [], {}  # 各切分的結果；各題池候選人的兩半分組
    for tag, pool in (("public", "pub2"), ("private", "priv2")):  # 兩個新增池依序處理（pub2 先）
        if a.from_dumps:  # 讀回模式
            for s in a.seeds:  # 每個種子的切分
                with open(Path(a.dump) / f"split_{pool}_{s}.pkl", "rb") as fh:  # 開啟切分檔
                    r = pickle.load(fh)  # 讀回切分結果
                if pool not in half:  # 這個題池第一次出現
                    half[pool] = rng.permutation(r["players"]) % 2  # 與原版 pid_v2_offsim.py 相同的抽法：每個題池一次，pub2 先
                runs.append(r)  # 收進結果
                print(r["line"], flush=True)  # 印出計算該切分時存下的摘要行
            continue  # 這個題池不必再算
        if only and only[0] != pool:  # 只算一個切分時跳過其他題池
            continue  # 換下一個題池
        cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")  # 讀新增池的候選人與棋譜
        players = sorted(cands)  # 候選人依編號排序
        half[pool] = rng.permutation(len(players)) % 2  # 權重校正用的候選人兩半
        ckeys = {k: c for g in cands.values() for k, c in g}  # 候選棋局的鍵 → 顏色
        bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}  # 不可用（check_game 不通過）或沒有主幹列的局，之後一律排除
        moves = parse_all({k: sgfs[k] for k in ckeys if k not in bad}, a.workers)  # 平行解析可用的候選棋局
        for s in a.seeds:  # 每個切分種子
            if only and only[1] != s:  # 只算指定的種子
                continue  # 跳過
            hold = v2.candidate_holdout(cands, a.holdout, s)  # 每位候選人留出 h 局（與微調時相同的抽法）
            held = {k for g in hold.values() for k, _ in g}  # 留出棋局的鍵
            ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}  # 其餘的局當參考棋局
            questions, truth, primary = [], [], []  # 題目、正解、是否屬於擬合集
            for i, p in enumerate(players):  # 逐位候選人出題
                h = [(k, c) for k, c in hold[p] if k not in bad]  # 可用的留出棋局
                for j, g in enumerate(h):  # 每局各出一題 1 局題
                    questions.append([g]); truth.append(i); primary.append(j == 0)  # 只有第一題 1 局題進擬合集
                for m in (2, 3):  # 2 局題與 3 局題
                    if len(h) >= m:  # 留出棋局夠多時
                        questions.append(h[:m]); truth.append(i); primary.append(True)  # 取前 m 局成一題，都進擬合集
            truth, primary, n = np.array(truth), np.array(primary), np.array([len(q) for q in questions])  # 轉成陣列；n 是題目局數
            S = v2.components(ref, questions, moves, sgfs, doc["cfg"], store, emb, which=[c for c in comps if c != "seq"])  # 逐手以外的各成分
            Z = {c: standardise(S[c]) for c in S}  # 逐列標準化
            by_t = {}  # 各步數的向量
            for t in a.steps:  # 每個微調步數
                by_t[t] = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{t}.npz") for v, tg in nets]  # 讀各網路在這個切分的向量
                Z[("seq", t)] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=by_t[t])["seq"])  # 該步數的逐手成分
            if alt is not None:  # 有第二份文件時
                seqs = [load(Path(a.ft_dir) / f"offsim_{pool}_h{a.holdout}s{s}_net{v}{tg}_t{a.alt_step}.npz") for v, tg in nets_of(alt)]  # 讀第二份文件各網路的向量
                cfg_alt = {**doc["cfg"], **({"seq_near": alt["cfg"]["seq_near"]} if "seq_near" in alt.get("cfg", {}) else {})}  # 第二份文件自己的最近局設定（METHOD_NOTES 2.141）
                Z[("seq", "alt")] = standardise(v2.components(ref, questions, moves, sgfs, cfg_alt, which=("seq",), seq=seqs)["seq"])  # 第二份文件的逐手成分
            if a.mix:  # 多步數混合的一臂
                Z[("seq", "mix")] = standardise(v2.components(ref, questions, moves, sgfs, doc["cfg"], which=("seq",), seq=[x for t in a.mix for x in by_t[t]])["seq"])  # 各步數的向量一起平均
            runs.append({"pool": pool, "seed": s, "Z": Z, "truth": truth, "n": n, "primary": primary})  # 存這個切分
            alone = {t: mix(per_question(Z[("seq", t)], truth), n) for t in a.steps}  # 逐手成分單獨的分數
            line = (f"{pool} split {s}: {len(questions)} questions of {len(players)} candidates ({len(bad)} games left out); sequence component alone (mix) "  # 摘要的前半
                    + "  ".join(f"t{t} {x:.4f}" for t, x in alone.items()) + f" ({time.time() - t0:.0f} s)")  # 各步數的分數與耗時
            print(line, flush=True)  # 印出摘要
            if only:  # 平行版：存檔後結束
                Path(a.dump).mkdir(parents=True, exist_ok=True)  # 建立切分資料夾
                tmp = Path(a.dump) / f"split_{pool}_{s}.pkl.tmp"  # 先寫暫存檔
                with open(tmp, "wb") as fh:  # 開啟暫存檔
                    pickle.dump({**runs[-1], "players": len(players), "line": line}, fh, protocol=4)  # 另存候選人數與摘要行
                tmp.rename(Path(a.dump) / f"split_{pool}_{s}.pkl")  # 寫完才換名，讀取端不會看到半個檔
                print(f"wrote {a.dump}/split_{pool}_{s}.pkl ({time.time() - t0:.0f} s)", flush=True)  # 回報
                return 0  # 這個行程只負責一個切分
    if only:  # 指定的切分沒有算到
        sys.exit(f"split {a.only} was not computed (is the seed in --seeds?)")  # 以錯誤訊息結束
    mats = lambda r, t: [r["Z"][("seq", t)] if c == "seq" else r["Z"][c] for c in comps]  # noqa: E731  依成分順序取出某切分在某一臂的矩陣
    fus = Fusion()  # 使用文件權重的融合
    fus.w = np.asarray(doc["weights"], dtype=float)  # 直接套用文件的權重
    out = {"document": Path(a.fusion).name, "holdout": a.holdout, "seeds": a.seeds, "steps": a.steps, "scores": {}, "steps_diff": {}, "calibration": {}}  # 結果 JSON 的骨架
    cat = lambda rs, f: np.concatenate([f(r) for r in rs])  # noqa: E731  把各切分的同一項串接
    groups = {"pub2": [r for r in runs if r["pool"] == "pub2"], "priv2": [r for r in runs if r["pool"] == "priv2"], "both": runs}  # 依題池分組，另加兩池合計
    arms = [*a.steps, *(["mix"] if a.mix else []), *(["alt"] if alt is not None else [])]  # 所有要比較的臂
    fus_alt = Fusion()  # 第二份文件的融合
    fus_alt.w = np.asarray((alt or doc)["weights"], dtype=float)  # 沒有第二份文件時沿用主文件的權重
    sc = {(id(r), t): per_question((fus_alt if t == "alt" else fus).scores(mats(r, t)), r["truth"]) for r in runs for t in arms}  # 每個切分、每一臂的逐題分數
    for g, rs in groups.items():  # 每個分組
        n = cat(rs, lambda r: r["n"])  # 各題的局數
        cluster = cat(rs, lambda r: np.char.add(r["pool"] + "|", r["truth"].astype(str)))  # 群集 = 題池＋候選人
        out["scores"][g] = {}  # 這一組的分數
        for t in arms:  # 每一臂
            x = cat(rs, lambda r: sc[(id(r), t)])  # 串接各切分的逐題分數
            out["scores"][g][str(t)] = {"mix": mix(x, n), **{str(m): float(x[n == m].mean()) for m in SIZES}}  # mix 與各局數的分數
        print(f"{g}: document weights, by fine-tune steps: " + "  ".join(f"t{t} mix {v_['mix']:.4f} (n=1 {v_['1']:.4f}, n=2 {v_['2']:.4f}, n=3 {v_['3']:.4f})" for t, v_ in out["scores"][g].items()), flush=True)  # 印出這一組的分數
        for x_, y_ in ((50, 0), (100, 0), (100, 50), *[(t, 50) for t in a.steps if t not in (0, 50, 100)], *[("mix", t) for t in a.mix], ("alt", a.alt_step), *([("alt", 50)] if a.alt_step != 50 else [])):  # 要比較的成對臂
            if x_ in arms and y_ in arms:  # 兩臂都有算才比較
                d = cat(rs, lambda r: sc[(id(r), x_)] - sc[(id(r), y_)])  # 逐題的成對差
                b = boot_mix(d, n, cluster, a.boot, rng)  # 對候選人做 bootstrap
                b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}  # 各局數的平均差
                out["steps_diff"][f"{g}: t{x_} vs t{y_}"] = b  # 存進結果
                print(f"   t{x_} vs t{y_}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}", flush=True)  # 印出差與區間
    # 用題池自己的候選人校正權重：在一半候選人上擬合，在另一半計分
    for t in [x for x in a.steps if x > 0]:  # 每個有微調的步數
        cal = {}  # 各題池的校正結果
        for pool in ("pub2", "priv2"):  # 兩池各自校正，不混用
            rs, H = groups[pool], half[pool]  # 這個題池的切分與候選人分半
            d_all, n_all, cl_all, w_fit = [], [], [], []  # 收集差、局數、群集與擬合出的權重
            for side in (0, 1):  # 雙向：用一半擬合、另一半計分
                fit_rows = [np.flatnonzero((H[r["truth"]] == side) & r["primary"]) for r in rs]  # 擬合用：這一半候選人的擬合集題目
                f2 = Fusion().fit([np.vstack(col) for col in zip(*[[M[i] for M in mats(r, t)] for r, i in zip(rs, fit_rows)])], np.concatenate([r["truth"][i] for r, i in zip(rs, fit_rows)]))  # 在擬合集上擬合融合權重
                w_fit.append(f2.w)  # 記下這一半擬合出的權重
                for r in rs:  # 每個切分
                    te = np.flatnonzero(H[r["truth"]] != side)  # 計分用：另一半候選人的全部題目
                    d_all.append(per_question(f2.scores([M[te] for M in mats(r, t)]), r["truth"][te]) - sc[(id(r), t)][te])  # 校正權重減文件權重的逐題差
                    n_all.append(r["n"][te]); cl_all.append(r["truth"][te])  # 局數與候選人（群集）
            d, n, cl = np.concatenate(d_all), np.concatenate(n_all), np.concatenate(cl_all)  # 合併兩個方向
            b = boot_mix(d, n, cl, a.boot, rng)  # 對候選人做 bootstrap
            b["by_n"] = {str(m): float(d[n == m].mean()) for m in SIZES}  # 各局數的平均差
            rows_all = [np.flatnonzero(r["primary"]) for r in rs]  # 全部候選人的擬合集題目
            f_all = Fusion().fit([np.vstack(col) for col in zip(*[[M[i] for M in mats(r, t)] for r, i in zip(rs, rows_all)])], np.concatenate([r["truth"][i] for r, i in zip(rs, rows_all)]))  # 用全部候選人擬合權重；定案的 w11c＝兩池在 50 步的這組擬合值取平均（METHOD_NOTES 2.233）
            b["weights_half"] = [dict(zip(comps, w.tolist())) for w in w_fit]  # 兩半各自擬合的權重
            b["weights_all"] = dict(zip(comps, f_all.w.tolist()))  # 用全部候選人擬合的權重
            cal[pool] = b  # 存這個題池的結果
            print(f"calibrated vs document weights, t{t}, {pool}: {b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  by n {({k: round(v_, 4) for k, v_ in b['by_n'].items()})}; "  # 校正權重對文件權重的差與區間
                  f"weights fitted on all its candidates {({c: round(w, 2) for c, w in b['weights_all'].items()})}", flush=True)  # 以及用全部候選人擬合的權重
        out["calibration"][str(t)] = cal  # 依步數存校正結果
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    Path(a.out).write_text(to_json(out, indent=1) + "\n")  # 寫結果 JSON
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")  # 回報


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
