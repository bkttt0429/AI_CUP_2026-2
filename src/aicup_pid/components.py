"""Build the per-candidate score matrices of a pool for the ladder's components (shared by pid_cv.py and predict_pid.py)."""
import multiprocessing as mp  # 平行解析棋譜

import numpy as np  # 陣列運算

from aicup_pid.hd import HDModel  # 讓子習慣模型
from aicup_pid.local import LocalModel  # 局部棋形模型
from aicup_pid.orient import OrientModel  # 棋盤方向模型
from aicup_pid.ref import RefModel, cosine_scores  # 參考玩家親和度
from aicup_pid.opening import OpeningModel, parse  # 開局模型與棋譜解析
from aicup_pid.style import StyleModel, game_features  # 風格統計模型與特徵

COMPONENTS = ("opening", "style")  # 正式流程的融合成分
OPTIONAL = ("local", "hd", "orient", "opening_raw", "ref", "trunk", "emb")  # 支線 B2（fusion3）、B5（hd）、B6（orient、opening_raw）、B7（ref）、B4（trunk）、B8（emb）：只有融合文件列出時才擬合
ALL_COMPONENTS = (*COMPONENTS, *OPTIONAL)  # 所有成分名稱


def _features(sgf):  # 給行程池用：解析一局並算風格特徵
    return game_features(parse(sgf))  # 解析後計算特徵列


def parse_all(sgfs, workers=1):  # 解析所有棋譜
    """{game_key: parsed moves}."""
    keys = list(sgfs)  # 固定鍵的順序
    if workers > 1:  # 使用多個行程
        with mp.get_context("fork").Pool(workers) as pool:  # 以 fork 建立行程池
            moves = pool.map(parse, [sgfs[k] for k in keys], chunksize=256)  # 平行解析
    else:  # 單一行程
        moves = [parse(sgfs[k]) for k in keys]  # 逐局解析
    return dict(zip(keys, moves))  # 鍵對應解析結果


def features_all(sgfs, workers=1):  # 計算所有棋局的風格特徵
    """{game_key: style feature row}."""
    keys = list(sgfs)  # 固定鍵的順序
    if workers > 1:  # 使用多個行程
        with mp.get_context("fork").Pool(workers) as pool:  # 以 fork 建立行程池
            rows = pool.map(_features, [sgfs[k] for k in keys], chunksize=64)  # 平行計算特徵
    else:  # 單一行程
        rows = [_features(sgfs[k]) for k in keys]  # 逐局計算
    return dict(zip(keys, rows))  # 鍵對應特徵列


def fit_from_parsed(cand_keys_by_player, moves, feats, cfg):  # 擬合開局與風格兩個主線模型
    """(OpeningModel, StyleModel) fitted on already parsed games: moves / feats cover every key of cand_keys_by_player."""
    opening = OpeningModel(depth=cfg.get("depth", 12), alpha=cfg.get("alpha", 2.0), beta=cfg.get("beta", 5.0), alpha_full=cfg.get("alpha_full", 1.0))  # 依設定建立開局模型
    opening.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})  # 以候選人的棋局擬合
    style = StyleModel(shrink=cfg.get("shrink", 0.1))  # 依設定建立風格模型
    style.fit({p: np.stack([feats[k] for k in keys]) for p, keys in cand_keys_by_player.items()})  # 以候選人的特徵列擬合
    if opening.players != style.players:  # 兩個模型的候選人順序必須一致
        raise ValueError("the two models disagree on the candidate order")  # 順序不符就中止
    return opening, style  # 回傳兩個模型


def fit_models(cand_keys_by_player, sgfs, cfg, workers=1):  # 從棋譜開始擬合兩個主線模型
    """(OpeningModel, StyleModel, parsed moves, feature rows) fitted on the candidates' games; sgfs covers every key used."""
    moves = parse_all(sgfs, workers)  # 解析所有棋譜
    feats = features_all(sgfs, workers)  # 計算所有特徵列
    opening, style = fit_from_parsed(cand_keys_by_player, moves, feats, cfg)  # 擬合兩個模型
    return opening, style, moves, feats  # 一併回傳解析結果供計分重用


def score_questions(opening, style, question_keys, moves, feats):  # 主線兩個成分的分數矩陣
    """{component: [Q, C]} for questions given as lists of game keys."""
    S_o = np.stack([opening.log_scores([moves[k] for k in keys]) for keys in question_keys]) if question_keys else np.zeros((0, len(opening.players)))  # 開局成分的分數矩陣
    S_s = np.stack([style.log_scores(np.stack([feats[k] for k in keys])) for keys in question_keys]) if question_keys else np.zeros((0, len(style.players)))  # 風格成分的分數矩陣
    return {"opening": S_o, "style": S_s}  # 回傳兩個主線成分


def fit_local(cand_keys_by_player, moves, cfg):  # 擬合局部棋形模型
    """LocalModel on the candidates' parsed games (cfg keys local_alpha, local_beta, local_start)."""
    model = LocalModel(alpha=cfg.get("local_alpha", 2.0), beta=cfg.get("local_beta", 5.0), start=cfg.get("local_start", 12))  # 依設定建立模型
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})  # 以候選人的棋局擬合


def score_local(model, question_keys, moves):  # 以符記模型計算 [Q, C] 分數（局部、方向、原始開局共用）
    """[Q, C] local-shape log scores for questions given as lists of game keys."""
    if not question_keys:  # 沒有題目
        return np.zeros((0, len(model.players)))  # 回傳空矩陣
    return np.stack([model.log_scores([moves[k] for k in keys]) for keys in question_keys])  # 每題一列


def fit_hd(cand_keys_by_player, sgfs, cfg):  # 擬合讓子習慣模型
    """HDModel on the candidates' records (cfg keys hd_alpha, hd_beta); sgfs covers every candidate key."""
    model = HDModel(alpha=cfg.get("hd_alpha", 2.0), beta=cfg.get("hd_beta", 5.0))  # 依設定建立模型
    return model.fit({p: [sgfs[k] for k in keys] for p, keys in cand_keys_by_player.items()})  # 以候選人的棋譜擬合


def score_hd(model, question_keys, sgfs):  # 讓子習慣成分的分數矩陣
    """[Q, C] handicap-habit log scores for questions given as lists of game keys."""
    if not question_keys:  # 沒有題目
        return np.zeros((0, len(model.players)))  # 回傳空矩陣
    return np.stack([model.log_scores([sgfs[k] for k in keys]) for keys in question_keys])  # 每題一列


def fit_orient(cand_keys_by_player, moves, cfg):  # 擬合棋盤方向模型
    """OrientModel on the candidates' parsed games (cfg keys orient_alpha, orient_beta, orient_depth, orient_raw)."""
    model = OrientModel(alpha=cfg.get("orient_alpha", 2.0), beta=cfg.get("orient_beta", 5.0), depth=cfg.get("orient_depth", 12), n_raw=cfg.get("orient_raw", 1))  # 依設定建立模型
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})  # 以候選人的棋局擬合


def fit_opening_raw(cand_keys_by_player, moves, cfg):  # 擬合原始座標的開局模型
    """The opening back-off model on raw coordinates (no symmetry normalisation; same depth / alpha / beta / alpha_full)."""
    model = OpeningModel(depth=cfg.get("depth", 12), alpha=cfg.get("alpha", 2.0), beta=cfg.get("beta", 5.0), alpha_full=cfg.get("alpha_full", 1.0), symmetry=False)  # 關閉對稱正規化
    return model.fit({p: [moves[k] for k in keys] for p, keys in cand_keys_by_player.items()})  # 以候選人的棋局擬合


def fit_ref(ref_keys_by_player, moves, cfg):  # 擬合參考玩家親和度模型
    """RefModel on the reference players' parsed games (cfg keys ref_alpha, ref_beta, local_start, orient_depth, ref_orient_raw)."""
    model = RefModel(alpha=cfg.get("ref_alpha", 2.0), beta=cfg.get("ref_beta", 5.0), local_start=cfg.get("local_start", 12),  # 平滑參數與局部棋形起始手數
                     orient_depth=cfg.get("orient_depth", 12), orient_raw=cfg.get("ref_orient_raw", 2))  # 方向符記的設定
    return model.fit({p: [moves[k] for k in keys] for p, keys in ref_keys_by_player.items()})  # 以參考玩家的棋局擬合


def score_ref(model, cand_keys_by_player, question_keys, moves):  # 參考親和度成分的分數矩陣
    """[Q, C] cosine between the questions' and the candidates' reference-affinity profiles (candidates in sorted id order,
    as the other models)."""
    players = sorted(cand_keys_by_player)  # 候選人依 id 排序
    C = np.stack([model.profile([moves[k] for k in cand_keys_by_player[p]]) for p in players])  # 每位候選人的輪廓
    if not question_keys:  # 沒有題目
        return np.zeros((0, len(players)))  # 回傳空矩陣
    Q = np.stack([model.profile([moves[k] for k in keys]) for keys in question_keys])  # 每題的輪廓
    return cosine_scores(Q, C)  # 輪廓之間的餘弦


def score_components(comps, opening, style, question_keys, moves, feats, local=None, hd=None, sgfs=None, orient=None, opening_raw=None, ref=None, trunk=None, emb=None):  # 依融合文件組出所需成分
    """{component: [Q, C]} for the named components: the two production ones, plus 'local' / 'orient' / 'opening_raw'
    (model given), 'hd' (model and the sgf strings given), 'ref' (the [Q, C] matrix of score_ref) and 'trunk' (the
    [Q, C] matrix of aicup_pid.trunkstyle.score_trunk) and 'emb' (aicup_pid.embed.score_embed); the last three are
    computed by the caller."""
    S = score_questions(opening, style, question_keys, moves, feats)  # 先算兩個主線成分
    if "ref" in comps:  # 需要參考親和度
        if ref is None:  # 呼叫端沒有提供
            raise ValueError("the fusion names 'ref' but no reference-affinity matrix was given (score_ref)")  # 中止並說明
        S["ref"] = ref  # 直接使用呼叫端算好的矩陣
    if "trunk" in comps:  # 需要主幹風格
        if trunk is None:  # 呼叫端沒有提供
            raise ValueError("the fusion names 'trunk' but no trunk-style matrix was given (aicup_pid.trunkstyle.score_trunk)")  # 中止並說明
        S["trunk"] = trunk  # 直接使用呼叫端算好的矩陣
    if "emb" in comps:  # 需要每局嵌入
        if emb is None:  # 呼叫端沒有提供
            raise ValueError("the fusion names 'emb' but no embedding matrix was given (aicup_pid.embed.score_embed)")  # 中止並說明
        S["emb"] = emb  # 直接使用呼叫端算好的矩陣
    if "opening_raw" in comps:  # 需要原始座標開局
        if opening_raw is None:  # 沒有擬合模型
            raise ValueError("the fusion names 'opening_raw' but no raw opening model was fitted")  # 中止並說明
        if opening_raw.players != opening.players:  # 候選人順序必須一致
            raise ValueError("the raw opening model disagrees on the candidate order")  # 中止並說明
        S["opening_raw"] = score_local(opening_raw, question_keys, moves)  # 計算分數矩陣
    if "orient" in comps:  # 需要棋盤方向
        if orient is None:  # 沒有擬合模型
            raise ValueError("the fusion names 'orient' but no orientation model was fitted")  # 中止並說明
        if orient.players != opening.players:  # 候選人順序必須一致
            raise ValueError("the orientation model disagrees on the candidate order")  # 中止並說明
        S["orient"] = score_local(orient, question_keys, moves)  # 計算分數矩陣
    if "local" in comps:  # 需要局部棋形
        if local is None:  # 沒有擬合模型
            raise ValueError("the fusion names 'local' but no local model was fitted")  # 中止並說明
        if local.players != opening.players:  # 候選人順序必須一致
            raise ValueError("the local model disagrees on the candidate order")  # 中止並說明
        S["local"] = score_local(local, question_keys, moves)  # 計算分數矩陣
    if "hd" in comps:  # 需要讓子習慣
        if hd is None or sgfs is None:  # 缺模型或棋譜字串
            raise ValueError("the fusion names 'hd' but no hd model (or no sgf strings) was given")  # 中止並說明
        if hd.players != opening.players:  # 候選人順序必須一致
            raise ValueError("the hd model disagrees on the candidate order")  # 中止並說明
        S["hd"] = score_hd(hd, question_keys, sgfs)  # 計算分數矩陣
    return {c: S[c] for c in comps}  # 只回傳融合文件指定的成分

