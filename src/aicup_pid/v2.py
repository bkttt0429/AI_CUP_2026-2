"""The 11-08 v2 release's added pools (METHOD_NOTES 2.29): 600 candidates with 10 games each, questions of 1 / 2 / 3 / 5 games,
the player Black or White in any of them, several questions per player possible.

A game is seen from its target player: when the player is White the colours are swapped, so every model's "Black" is the
player. Opening lines and raw-coordinate habits differ between moving first and second, so those models are fitted per view
(the player's Black games / White games) and a question's score is the sum, over its games, of the log score under the model
of that game's view; a candidate without a game in a view scores the background there. Style statistics, local shapes and the
learnt embedding use all of a player's games together. Every function returns [Q, C] with the candidates in sorted id order.
"""
import numpy as np  # 分數矩陣與陣列運算

from aicup_pid.hd import HDModel  # 讓子旗標成分
from aicup_pid.local import LocalModel  # 局部棋形成分
from aicup_pid.opening import OpeningModel  # 開局模型成分
from aicup_pid.orient import OrientModel  # 原始棋盤方向成分
from aicup_pid.style import StyleModel, game_features  # 風格統計模型與每局特徵

VIEWS = ("B", "W")  # 兩種視角：玩家執黑、玩家執白
SPLIT = ("opening", "opening_raw", "orient4", "orient1", "local_split", "hd")  # 按視角分開擬合的成分
MERGED = ("local", "style", "orient4_merged", "emb", "seq")  # 不分視角、用全部棋局擬合的成分


def swap(moves):  # 把整局的黑白互換
    return [("W" if c == "B" else "B", *rest) for c, *rest in moves]  # 只換顏色，座標不變


def view(moves, colour):  # 轉成目標玩家的視角
    """The parsed game with the target player's stones marked B."""
    return moves if colour == "B" else swap(moves)  # 玩家執白時才對調


def _split(make, cand_items, q_items, players):  # 按視角分開擬合並計分
    """Per-view models: cand_items {view: {player: [items]}}, q_items [[(item, view)]] -> [Q, C]."""
    models = {v: make().fit(cand_items[v]) for v in VIEWS}  # 每個視角各擬合一個模型
    if any(list(m.players) != players for m in models.values()):  # 檢查兩個視角的候選人順序
        raise ValueError("candidate order differs between the views")  # 順序不同會讓分數錯位
    out = np.zeros((len(q_items), len(players)))  # 題目 × 候選人的分數矩陣
    for qi, games in enumerate(q_items):  # 逐題計分
        for v in VIEWS:  # 逐一視角
            items = [it for it, vv in games if vv == v]  # 題目中屬於這個視角的棋局
            if items:  # 這個視角有棋局才計分
                out[qi] += models[v].log_scores(items)  # 各視角的對數分數相加
    return out  # 回傳 [Q, C]


def components(cands, questions, moves, sgfs, cfg, rows=None, emb=None, which=None, seq=None, rows_full=None, emb_full=None):  # 計算一個題池各成分的 [Q, C] 分數矩陣（官方新增池與各種模擬題池共用）
    """{name: [Q, C]}. cands {player: [(key, colour)]}, questions [[(key, colour)]], moves {key: parsed moves as recorded},
    sgfs {key: record}, rows {key: [256] trunk row of the player's side}, emb a fitted aicup_pid.embed.PlayerEmbedding,
    seq {key: [128] unit vector of the player's side from the sequence identity model (experiments/pid_v2_seq_embed.py)};
    rows_full {key: row of the UNPROJECTED trunk statistics (scripts/pid_trunk_extract.py --full)} and emb_full, one or several
    PlayerEmbedding fitted on such rows: the component 'embfull' (METHOD_NOTES 2.75)."""
    which = set(which or (*SPLIT, *MERGED))  # 沒指定時算 SPLIT 與 MERGED 的成分（embfull 須明確指定）
    players = sorted(cands)  # 候選人依編號排序，即分數矩陣的欄順序
    vm = {k: view(moves[k], c) for games in (*cands.values(), *questions) for k, c in games}  # 每局轉成目標玩家的視角
    cand_m = {v: {p: [vm[k] for k, c in cands[p] if c == v] for p in players} for v in VIEWS}  # 候選人棋局依視角分組
    q_m = [[(vm[k], c) for k, c in q] for q in questions]  # 題目棋局附上各自的視角
    all_m = {p: [vm[k] for k, _ in cands[p]] for p in players}  # 候選人的全部棋局（合併視角的成分用）
    a, b, depth = cfg.get("alpha", 2.0), cfg.get("beta", 5.0), cfg.get("depth", 12)  # 開局模型的平滑參數與深度
    S = {}  # 收集各成分的分數矩陣
    if "opening" in which:  # 開局模型（對稱正規化）
        S["opening"] = _split(lambda: OpeningModel(depth=depth, alpha=a, beta=b, alpha_full=cfg.get("alpha_full", 1.0)), cand_m, q_m, players)  # 按視角分開擬合
    if "opening_raw" in which:  # 原始座標的開局模型
        S["opening_raw"] = _split(lambda: OpeningModel(depth=depth, alpha=a, beta=b, alpha_full=cfg.get("alpha_full", 1.0), symmetry=False), cand_m, q_m, players)  # 不做對稱正規化，按視角分開
    for n_raw in (1, 4):  # 方向成分兩版：視角中黑方（即玩家）前 1 或前 4 手的原始座標，外加對稱索引
        if f"orient{n_raw}" in which:  # 有要求這一版時
            S[f"orient{n_raw}"] = _split(lambda: OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=n_raw),  # 按視角分開的方向模型
                                         cand_m, q_m, players)  # 候選人棋局、題目棋局與候選人順序
    if "local_split" in which:  # 按視角分開的局部棋形
        S["local_split"] = _split(lambda: LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12)), cand_m, q_m, players)  # 每個視角各擬合一個
    if "hd" in which:  # 讓子旗標成分
        S["hd"] = _split(lambda: HDModel(alpha=cfg.get("hd_alpha", 2.0), beta=cfg.get("hd_beta", 5.0)), {v: {p: [sgfs[k] for k, c in cands[p] if c == v] for p in players} for v in VIEWS},  # 讀原始棋譜的 HD，依視角分組
                         [[(sgfs[k], c) for k, c in q] for q in questions], players)  # 題目棋譜附上各自的視角
    if "local" in which:  # 合併視角的局部棋形
        m = LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12)).fit(all_m)  # 用候選人的全部棋局擬合
        S["local"] = np.stack([m.log_scores([vm[k] for k, _ in q]) for q in questions])  # 每題一列對數分數
    if "orient4_merged" in which:  # 合併視角的方向成分（前 4 手）
        m = OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=4).fit(all_m)  # 用候選人的全部棋局擬合
        S["orient4_merged"] = np.stack([m.log_scores([vm[k] for k, _ in q]) for q in questions])  # 每題一列對數分數
    if "style" in which:  # 風格統計成分
        feats = {k: game_features(g) for k, g in vm.items()}  # 每局在玩家視角下的風格特徵
        m = StyleModel(shrink=cfg.get("shrink", 0.1)).fit({p: np.stack([feats[k] for k, _ in cands[p]]) for p in players})  # 每位候選人一個平均、共用收縮後的類內共變異數（LDA 式高斯模型）
        if list(m.players) != players:  # 確認候選人順序一致
            raise ValueError("style: candidate order differs")  # 順序不同就報錯
        S["style"] = np.stack([m.log_scores(np.stack([feats[k] for k, _ in q])) for q in questions])  # 每題一列對數分數
    if "emb" in which and emb is not None:  # 學習式玩家嵌入成分（沒給 emb 時略過）
        mats = []  # 每個嵌入模型的分數矩陣
        for one in (emb if isinstance(emb, (list, tuple)) else [emb]):  # 多個種子：取它們逐列標準化矩陣的平均
            C = np.stack([one.set_vector(np.stack([rows[k] for k, _ in cands[p]])) for p in players])  # 候選人的集合向量：各局嵌入相加後單位化
            Q = np.stack([one.set_vector(np.stack([rows[k] for k, _ in q])) for q in questions])  # 題目的集合向量
            mats.append(one.scale * (Q @ C.T))  # 餘弦乘上模型學到的尺度
        if len(mats) == 1:  # 只有一個模型
            S["emb"] = mats[0]  # 直接使用，不做逐列標準化
        else:  # 多個種子的模型
            from aicup_pid.fusion import standardise  # 延遲匯入逐列標準化
            S["emb"] = np.mean([standardise(m) for m in mats], axis=0)  # 逐列標準化後平均
    if "embfull" in which and emb_full is not None:  # 未投影列的嵌入成分（METHOD_NOTES 2.75）
        from aicup_pid.fusion import standardise  # 延遲匯入逐列標準化
        mats = []  # 每個模型的分數矩陣
        for one in (emb_full if isinstance(emb_full, (list, tuple)) else [emb_full]):  # 多個種子：取它們逐列標準化矩陣的平均
            C = np.stack([one.set_vector(np.stack([rows_full[k] for k, _ in cands[p]])) for p in players])  # 候選人的集合向量
            Q = np.stack([one.set_vector(np.stack([rows_full[k] for k, _ in q])) for q in questions])  # 題目的集合向量
            mats.append(standardise(one.scale * (Q @ C.T)))  # 逐列標準化後收集（與 emb 不同，單一模型也標準化）
        S["embfull"] = np.mean(mats, axis=0)  # 各模型平均
    if "seq" in which and seq is not None:  # 原型＝各局向量相加再正規化，與模型訓練時的做法相同
        unit = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # noqa: E731  逐列單位化的小函式
        mats = []  # 每個網路的分數矩陣
        # METHOD_NOTES 2.141：cfg["seq_near"] = {"k": 3, "weight": 0.5} 會對每個網路，加上題目各局與候選人「單盤」參考棋局的比對
        # （取最相似 k 盤的餘弦平均，再對題目的各局平均），逐列標準化後加入
        near, nears, done = cfg.get("seq_near"), [], {}  # 逐局比對的設定、各網路的結果與快取
        if near:  # 啟用逐局比對時，先建立攤平的索引
            c_games = [k for p in players for k, _ in cands[p]]  # 依候選人順序攤平的參考棋局
            c_len = np.array([len(cands[p]) for p in players])  # 每位候選人的參考局數
            c_start = np.concatenate([[0], np.cumsum(c_len)[:-1]])  # 每位候選人在攤平清單中的起點
            q_games = [k for q in questions for k, _ in q]  # 攤平的題目棋局
            q_of = np.repeat(np.arange(len(questions)), [len(q) for q in questions])  # 每個題目棋局屬於哪一題
            q_cnt = np.bincount(q_of, minlength=len(questions)).astype(np.float64)  # 每題的局數
            kk = int(near.get("k", 3))  # 取最相似的幾盤
        for one in (seq if isinstance(seq, (list, tuple)) else [seq]):  # 多個網路：取它們逐列標準化矩陣的平均
            C = unit(np.stack([np.sum([one[k] for k, _ in cands[p]], axis=0) for p in players]).astype(np.float64))  # 候選人原型：各局向量相加後單位化
            Q = unit(np.stack([np.sum([one[k] for k, _ in q], axis=0) for q in questions]).astype(np.float64))  # 題目原型：同樣相加後單位化
            mats.append(Q @ C.T)  # 原型之間的餘弦矩陣
            if near:  # 啟用逐局比對時才計算
                G_ = np.stack([one[k] for k in c_games])  # 這個網路的全部參考局向量
                mark = (G_[0].tobytes(), G_[-1].tobytes())  # 融合文件依權重單位重複列出同一網路：只算一次
                if mark not in done:  # 這個網路還沒算過
                    X = unit(np.stack([one[k] for k in q_games]).astype(np.float64)) @ unit(G_.astype(np.float64)).T  # [題目棋局, 參考棋局] 的餘弦
                    if (c_len == c_len[0]).all():  # 每位候選人的參考局數都相同
                        m_ = min(kk, int(c_len[0]))  # 實際取的盤數不超過參考局數
                        per = -np.partition(-X.reshape(len(q_games), len(players), int(c_len[0])), m_ - 1, axis=2)[:, :, :m_].mean(2)  # 向量化：每個題目棋局對每位候選人取最相似 m_ 盤的餘弦平均
                    else:  # 參考局數不一時
                        per = np.stack([np.sort(X[:, s_:s_ + n_], axis=1)[:, -min(kk, int(n_)):].mean(1) for s_, n_ in zip(c_start, c_len)], axis=1)  # 逐位候選人取最相似幾盤的平均
                    Nq = np.zeros((len(questions), len(players)))  # 題目 × 候選人的累計矩陣
                    np.add.at(Nq, q_of, per)  # 每局結果累加到所屬題目
                    done[mark] = Nq / q_cnt[:, None]  # 除以題目局數得到平均，存入快取
                nears.append(done[mark])  # 收集這個網路的逐局比對矩陣
        if len(mats) == 1:  # 只有一個網路（這個分支不加 seq_near 的逐局比對項）
            S["seq"] = mats[0]  # 直接使用其餘弦矩陣
        else:  # 多個網路
            from aicup_pid.fusion import standardise  # 延遲匯入逐列標準化
            S["seq"] = np.mean([standardise(m) for m in mats], axis=0)  # 逐列標準化後平均
            if near:  # 有逐局比對時
                S["seq"] = S["seq"] + float(near.get("weight", 0.5)) * np.mean([standardise(m) for m in nears], axis=0)  # 依 weight 加上逐局比對矩陣標準化後的平均
    return S  # 回傳 {成分: [Q, C]}


def official_pool(cand_path, q_path):  # 讀入一個官方新增池
    """(cands {player: [(key, colour)]}, questions [[(key, colour)]], question ids, sgfs {key: record}) of one added pool; keys as
    scripts/pid_trunk_extract.py --official2 writes them (c:<row>:<game_id>, q:<question_id>:<i>)."""
    import pandas as pd  # 延遲匯入，讀取候選人 CSV
    from aicup_pid import pools  # 讀取官方題目檔
    c = pd.read_csv(cand_path, dtype=str, keep_default_na=False)  # 全部以字串讀入、空值不轉 NaN，保留編號原樣
    cands, sgfs = {}, {}  # 候選人棋局與棋譜對照
    for i, r in enumerate(c.itertuples()):  # 逐列處理候選人棋局
        key = f"c:{i}:{r.game_id}"  # 候選局鍵：列號加棋局編號，與抽特徵時一致
        cands.setdefault(r.player_id, []).append((key, r.color))  # 依玩家收集 (鍵, 顏色)
        sgfs[key] = r.sgf_content  # 記下棋譜內容
    questions, qids = [], []  # 題目棋局與題目編號
    for q in pools.read_questions(q_path):  # 逐題讀入官方題目
        games = []  # 這一題的棋局
        for i, (sgf, colour) in enumerate(q["games"]):  # 題目中的每一局
            key = f"q:{q['question_id']}:{i}"  # 題目局鍵：題號加局序
            games.append((key, colour or "B"))  # 沒標顏色時當作執黑
            sgfs[key] = sgf  # 記下棋譜內容
        questions.append(games)  # 收進題目清單
        qids.append(q["question_id"])  # 收進題號清單
    return cands, questions, qids, sgfs  # 回傳候選人、題目、題號與棋譜


def candidate_holdout(cands, h, seed):  # 從候選人棋局切出留出局
    """{player: [h held-out (key, colour)]}: the labelled pseudo-pool cut from an official pool's own candidates (METHOD_NOTES
    2.41). Per candidate (sorted ids, games in file order) h games drawn without replacement, in the drawn order; the same
    draw in scripts/pid_v2_seq_finetune.py --holdout (those games stay out of the fine-tune) and experiments/pid_v2_offsim.py."""
    rng = np.random.default_rng(4100 + seed)  # 固定種子，讓微調與擬題池評估抽到同一批局
    return {p: [cands[p][i] for i in rng.permutation(len(cands[p]))[:h]] for p in sorted(cands)}  # 依編號順序逐人不放回抽 h 局，保留抽出順序


def original_sim(cand_csv, keys, seed, n_ref=10, sizes=(1, 1, 1, 2, 2, 3, 5)):  # 用原題池候選人切出模擬題池
    """A simulated pool cut from an ORIGINAL pool's candidates (METHOD_NOTES 2.55, item 7): official-population players no
    network has seen, 10-game candidate sets, questions of 1 / 2 / 3 / 5 games. Per candidate (sorted ids), among its games
    in `keys` (those with per-ply vectors), a random order by `seed`: the first n_ref are the candidate set, the next ones the
    questions of the given sizes. Returns (cands {player: [(key, 'B')]}, questions [[(key, 'B')]], truth [player ids],
    sgfs {key: record}). Keys c:<row in the file>:<game_id>. The original pools are Black only."""
    import pandas as pd  # 延遲匯入，讀取候選人 CSV
    c = pd.read_csv(cand_csv, dtype=str, keep_default_na=False)  # 原題池候選人檔全部以字串讀入
    rng = np.random.default_rng(5200 + seed)  # 依種子固定的亂數
    by, rec = {}, {}  # 每位玩家可用的局鍵與棋譜
    for i, r in enumerate(c.itertuples()):  # 逐列處理候選人棋局
        k = f"c:{i}:{r.game_id}"  # 與抽特徵時相同的局鍵
        if k in keys:  # 只用有逐手向量的棋局
            by.setdefault(r.player_id, []).append(k)  # 依玩家收集
            rec[k] = r.sgf_content  # 記下棋譜
    cands, questions, truth = {}, [], []  # 候選集合、題目與正解
    for p in sorted(by):  # 依編號逐位候選人
        g = [by[p][i] for i in rng.permutation(len(by[p]))]  # 依種子打亂棋局順序
        if len(g) < n_ref + sum(sizes):  # 棋局不夠切出候選集合與全部題目
            continue  # 略過這位玩家
        cands[p] = [(k, "B") for k in g[:n_ref]]  # 前 n_ref 局當候選集合（只有執黑）
        at = n_ref  # 題目從這裡開始切
        for m in sizes:  # 依序切出每種局數的題目
            questions.append([(k, "B") for k in g[at:at + m]])  # 接下來的 m 局成為一題
            truth.append(p)  # 這題的正解是這位玩家
            at += m  # 移到下一題的起點
    return cands, questions, truth, {k: rec[k] for games in (*cands.values(), *questions) for k, _ in games}  # 只回傳用到的棋譜


RANK_ROWS = ("results/m82/rows_train.npz", "results/m84_rowsp")  # 位於 AI_CUP_2026-1 底下，由棋力 session 寫出（這裡唯讀）
_PERM = np.r_[0:64, 128:192, 64:128, 192:256]  # 棋力 session 存成 [兩個平均, 兩個標準差]；重排成本儲存庫（競賽一）的 [池化平均, 池化標準差, 落點平均, 落點標準差]


def load_side_rows(aicup1_root, keep=None):  # 讀入棋力 session 算好的每局主人一方主幹列
    """{training game key: [256] row of the game owner's side, either colour}, in competition 1's block order (mean and sd of
    the pooled vector, then of the played-point vector). The rank session stores the same features as [means, sds]; on the
    Black games both sessions have, the rows agree to 0.002 after the reorder (data/day_scripts/rows_compat.py, 2026-11-12)."""
    import glob  # 列出分段檔
    from pathlib import Path  # 組合路徑
    root, out = Path(aicup1_root), {}  # AI_CUP_2026-1 根目錄與輸出字典
    z = np.load(root / RANK_ROWS[0])  # 讀入 m82 的訓練列
    files = [(z["key"], z["T"], z["nT"])]  # 局鍵、主人一方的 256 維列與其手數
    for f in sorted(glob.glob(str(root / RANK_ROWS[1] / "*.npz"))):  # m84_rowsp 的每個分段檔
        y = np.load(f)  # 讀入分段檔
        files.append((y["key"], y["T"], y["nT"]))  # 加入同樣格式的三個陣列
    for key, T, n in files:  # 逐一處理每個來源
        for k, row, m in zip(key.astype(str), T, n):  # 每局的鍵、列與手數
            if m > 0 and (keep is None or k in keep):  # 該方有著手且在保留清單內
                out[k] = np.ascontiguousarray(row[_PERM], dtype=np.float32)  # 重排成競賽一的順序後存入（同一局鍵以後讀的檔為準）
    return out  # 回傳 {局鍵: 列}


def training_games(train_dir):  # 列出全部訓練棋局
    """(player_id, rank, game_key, color) of every training game, game_key 'RANK:game_id' as aicup.scoring."""
    import pandas as pd  # 延遲匯入，讀取訓練 CSV
    from pathlib import Path  # 組合路徑
    from aicup.metric import LEVELS  # 棋力競賽的十個級別，每級一個訓練檔
    frames = []  # 各級別的資料表
    for r in LEVELS:  # 逐一級別讀檔
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"], dtype=str)  # 只讀需要的四欄
        df["game_key"] = df["rank"] + ":" + df["game_id"]  # 組成與 aicup.scoring 相同的局鍵
        frames.append(df[["player_id", "rank", "game_key", "color"]])  # 保留輸出的四個欄位
    return pd.concat(frames, ignore_index=True)  # 合併成一張表


def pool_normalised(vecs, cands, questions, shrink=0.3):  # 以題池自身候選人的類內共變異數白化後比對
    """[Q, C] cosine of prototypes and question vectors after within-class covariance normalisation fitted on the pool's own
    candidates: W = covariance of the candidates' game vectors around their player's mean, shrunk towards (trace / d) I by
    `shrink`; every vector is multiplied by W^(-1/2) and renormalised. vecs {key: unit vector}; no question enters W."""
    players = sorted(cands)  # 候選人依編號排序
    X = [np.stack([vecs[k] for k, _ in cands[p]]).astype(np.float64) for p in players]  # 每位候選人的局向量矩陣
    resid = np.vstack([x - x.mean(axis=0) for x in X])  # 減去各自玩家平均後的殘差
    d = resid.shape[1]  # 向量維度
    W = resid.T @ resid / max(len(resid) - len(X), 1)  # 類內共變異數（每位玩家扣一個自由度）
    W = (1 - shrink) * W + shrink * (np.trace(W) / d) * np.eye(d)  # 往等向的對角矩陣收縮
    val, vec = np.linalg.eigh(W)  # 對稱矩陣的特徵分解，用來算 W^(-1/2)
    B = vec @ np.diag(1.0 / np.sqrt(np.maximum(val, 1e-12))) @ vec.T  # W^(-1/2)，特徵值設下限避免除以 0
    unit = lambda M: M / (np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12)  # noqa: E731  單位化的小函式
    C = unit(np.stack([unit(x @ B).sum(axis=0) for x in X]))  # 候選人原型：白化後各局單位化、相加再單位化
    Q = unit(np.stack([unit(np.stack([vecs[k] for k, _ in q]).astype(np.float64) @ B).sum(axis=0) for q in questions]))  # 題目原型：同樣做法
    return Q @ C.T  # 原型之間的餘弦 [Q, C]


def load_phase_rows(aicup1_root, stored_dir, keep=None):  # 讀入帶階段平均的訓練棋局列
    """{training game key: [640]} = the 256-dim row of the owner's side (competition 1's block order) + the three phase means
    (plies t < 60, 60-150, >= 150; 128 dims each). Sources: the rank session's results/m84_rowsp (games without stored per-ply
    vectors) and experiments/pid_v2_phase_rows.py's stored_f<fold>.npz (the others); the same function computed both."""
    import glob  # 列出分段檔
    from pathlib import Path  # 組合路徑
    out = {}  # 局鍵 → 640 維列
    files = sorted(glob.glob(str(Path(aicup1_root) / RANK_ROWS[1] / "*.npz"))) + sorted(glob.glob(str(Path(stored_dir) / "stored_f*.npz")))  # 棋力 session 的 m84_rowsp 分段檔，加上 pid_v2_phase_rows.py 依折寫出的 stored_f 檔
    for f in files:  # 逐檔讀取
        z = np.load(f)  # 讀入 npz
        n = z["nT"] if "nT" in z.files else np.ones(len(z["key"]))  # m84_rowsp 有手數欄；stored 檔沒有（寫檔時已濾掉該方沒有著手的局），全部視為可用
        for k, row, ph, m in zip(z["key"].astype(str), z["T"], z["PT"], n):  # 每局的鍵、256 維列、階段平均與手數
            if m > 0 and (keep is None or k in keep):  # 該方有著手且在保留清單內
                out[k] = np.concatenate([row[_PERM].astype(np.float32), ph.astype(np.float32)])  # 重排後接上階段平均
    return out  # 回傳 {局鍵: [640]}


def load_store_phase(*dirs):  # 讀入帶階段平均的官方棋局列
    """{official game key: [640]} from pid_trunk_extract.py parts that carry the phase means."""
    import glob  # 列出分段檔
    store = {}  # 局鍵 → 640 維列
    for d in dirs:  # 逐一處理每個資料夾
        for f in sorted(glob.glob(f"{d}/*.npz")):  # 依檔名順序讀每個分段檔
            if f.endswith(".tmp.npz"):  # 抽取程式寫入中的暫存檔
                continue  # 略過，避免讀到不完整的資料
            z = np.load(f, allow_pickle=False)  # 讀入分段檔（不允許 pickle）
            for k, row, ph in zip(z["keys"], z["feats"], z["phase"]):  # 每局的鍵、256 維列與階段平均
                store[str(k)] = np.concatenate([row.astype(np.float32), ph.astype(np.float32)])  # 已是本儲存庫的區塊順序，不必重排；接上階段平均成 640 維
    return store  # 回傳完整的對照表
