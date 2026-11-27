"""Raw style statistics of a game's Black moves and a Gaussian candidate model (docs/records/ARCHITECTURE.md ladder (2a)).

game_features replays the game on a small board (captures handled) and summarises Black's moves: lines, distances to the
previous Black move and to White's last move, contact moves, captures, the first two Black moves' point types, length.
StyleModel fits one mean per candidate and a pooled within-candidate covariance (shrunk toward its diagonal); the log
score of a query (mean of its games' features) under candidate c is -1/2 d' S^-1 d with d = q - mu_c and S = Sigma x (1/m +
1/n_c): an LDA-style score, one value per candidate.
"""
import numpy as np  # 陣列運算

BOARD = 19  # 棋盤大小
LINES = (1, 2, 3, 4)  # 第 5 線以上歸入最後一格
POINT_TYPES = ("44", "34", "33", "35", "45", "other")  # 開局點位類型：星位、小目、三三、目外、高目、其他
FEATURE_NAMES = (["log_moves", "black_moves", *[f"line_{k}" for k in (*LINES, "5plus")],  # 特徵名稱：長度與各線的比例
                  *[f"first_{t}" for t in POINT_TYPES], *[f"second_{t}" for t in POINT_TYPES],  # 黑棋第一、二手的點位類型
                  "prev_le2", "prev_3to5", "prev_ge6", "prev_mean", "opp_le1", "opp_2", "opp_3to5", "opp_ge6", "contact",  # 與己方上一手、白棋上一手的距離及接觸手
                  "captures_made", "stones_lost", "early_line3", "early_line4"])  # 提子、被提子與早期的三、四線比例


class Board:  # 計算提子用的簡易棋盤
    """Stones with captures (no suicide handling beyond removing a libertyless own group)."""

    def __init__(self):  # 建立空棋盤
        self.s = [[0] * BOARD for _ in range(BOARD)]  # 0 空、1 黑、2 白

    def _group(self, x, y):  # 找出相連的棋塊與其氣
        colour = self.s[y][x]  # 棋塊的顏色
        seen, stack, libs = {(x, y)}, [(x, y)], set()  # 已走訪的點、待處理的點、氣
        while stack:  # 深度優先搜尋
            cx, cy = stack.pop()  # 取出一個點
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):  # 四個相鄰點
                if 0 <= nx < BOARD and 0 <= ny < BOARD:  # 在棋盤內
                    v = self.s[ny][nx]  # 相鄰點的狀態
                    if v == 0:  # 空點
                        libs.add((nx, ny))  # 是一口氣
                    elif v == colour and (nx, ny) not in seen:  # 同色且尚未走訪
                        seen.add((nx, ny))  # 加入棋塊
                        stack.append((nx, ny))  # 繼續往外擴展
        return seen, libs  # 回傳棋塊與氣

    def play(self, colour, x, y):  # 落子並處理提子
        """Place a stone (1 Black, 2 White); returns the number of opponent stones captured."""
        self.s[y][x] = colour  # 放下棋子
        other, captured = 3 - colour, 0  # 對方的顏色與提子數
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):  # 檢查四個相鄰點
            if 0 <= nx < BOARD and 0 <= ny < BOARD and self.s[ny][nx] == other:  # 相鄰的對方棋子
                group, libs = self._group(nx, ny)  # 對方棋塊與其氣
                if not libs:  # 沒有氣了
                    for gx, gy in group:  # 提掉整塊
                        self.s[gy][gx] = 0  # 清空該點
                    captured += len(group)  # 累計提子數
        if captured == 0:  # 沒有提子時檢查自殺手
            group, libs = self._group(x, y)  # 己方棋塊與其氣
            if not libs:  # 自殺手：資料中不合法，但仍讓棋盤保持一致
                for gx, gy in group:  # 移除己方棋塊
                    self.s[gy][gx] = 0  # 清空該點
        return captured  # 回傳提子數

    def adjacent_to(self, x, y, colour):  # 該點是否緊鄰某色棋子
        return any(0 <= nx < BOARD and 0 <= ny < BOARD and self.s[ny][nx] == colour  # 任一相鄰點為該顏色
                   for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))  # 四個相鄰點


def line_of(x, y):  # 第幾線
    return min(x, BOARD - 1 - x, y, BOARD - 1 - y) + 1  # 到最近邊的距離加 1


def point_type(x, y):  # 開局點位類型
    a, b = sorted((min(x, BOARD - 1 - x), min(y, BOARD - 1 - y)))  # 兩軸到邊的距離，排序後與方向無關
    return {(3, 3): "44", (2, 3): "34", (2, 2): "33", (2, 4): "35", (3, 4): "45"}.get((a, b), "other")  # 對應到點位名稱


def _cheb(a, b):  # 切比雪夫距離
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # 兩軸距離取最大值


def game_features(moves, early=30):  # 一局棋的風格特徵列
    """[len(FEATURE_NAMES)] floats of a parsed game's Black moves (zeros where Black has no moves)."""
    f = dict.fromkeys(FEATURE_NAMES, 0.0)  # 所有特徵先設為 0
    board = Board()  # 用來計算提子的棋盤
    n_black, prev_black, last_white = 0, None, None  # 黑棋手數、黑棋上一手、白棋上一手
    lines, prev_d, opp_d, early_lines = [], [], [], []  # 各手的線、與己方上一手距離、與白棋上一手距離、早期各手的線
    contact = captures = lost = 0  # 接觸手數、提子數、被提子數
    firsts = []  # 黑棋前兩手的點位類型
    for i, (c, x, y) in enumerate(moves):  # 逐手重播
        if x is None:  # 這一手是 pass
            continue  # 略過，不影響棋盤與統計
        colour = 1 if c == "B" else 2  # 棋盤上的顏色代碼
        if colour == 1:  # 黑棋的落子
            n_black += 1  # 累計黑棋手數
            lines.append(line_of(x, y))  # 記下第幾線
            if i < early:  # 在前 early 手內
                early_lines.append(line_of(x, y))  # 記下早期的線
            if len(firsts) < 2:  # 黑棋的前兩手
                firsts.append(point_type(x, y))  # 記下點位類型
            if prev_black is not None:  # 有黑棋上一手
                prev_d.append(_cheb((x, y), prev_black))  # 與己方上一手的距離
            if last_white is not None:  # 有白棋上一手
                opp_d.append(_cheb((x, y), last_white))  # 與白棋上一手的距離
            if board.adjacent_to(x, y, 2):  # 緊貼白棋
                contact += 1  # 累計接觸手
            captures += board.play(1, x, y)  # 落子並累計提子數
            prev_black = (x, y)  # 更新黑棋上一手
        else:  # 白棋的落子
            lost += board.play(2, x, y)  # 落子並累計黑棋被提的子數
            last_white = (x, y)  # 更新白棋上一手
    f["log_moves"] = float(np.log1p(len(moves)))  # 總手數（含 pass）加 1 後取對數
    f["black_moves"] = float(n_black)  # 黑棋手數
    if n_black == 0:  # 黑棋沒有落子
        return np.array([f[k] for k in FEATURE_NAMES])  # 其餘特徵維持 0
    lines = np.asarray(lines)  # 轉成陣列以便計算各線的比例
    for k in LINES:  # 第 1..4 線
        f[f"line_{k}"] = float((lines == k).mean())  # 各線的比例
    f["line_5plus"] = float((lines >= 5).mean())  # 第 5 線以上的比例
    for j, name in enumerate(("first", "second")):  # 第一手與第二手
        if j < len(firsts):  # 有這一手
            f[f"{name}_{firsts[j]}"] = 1.0  # 點位類型的 one-hot
    if prev_d:  # 有與己方上一手的距離
        d = np.asarray(prev_d)  # 轉成陣列以便計算各距離段的比例
        f["prev_le2"], f["prev_3to5"], f["prev_ge6"], f["prev_mean"] = float((d <= 2).mean()), float(((d >= 3) & (d <= 5)).mean()), float((d >= 6).mean()), float(d.mean())  # 分段比例與平均距離
    if opp_d:  # 有與白棋上一手的距離
        d = np.asarray(opp_d)  # 轉成陣列以便計算各距離段的比例
        f["opp_le1"], f["opp_2"], f["opp_3to5"], f["opp_ge6"] = float((d <= 1).mean()), float((d == 2).mean()), float(((d >= 3) & (d <= 5)).mean()), float((d >= 6).mean())  # 分段比例
    f["contact"] = contact / n_black  # 接觸手的比例
    f["captures_made"], f["stones_lost"] = float(captures), float(lost)  # 提子數與被提子數
    if early_lines:  # 有早期的落子
        e = np.asarray(early_lines)  # 轉成陣列以便計算早期三、四線的比例
        f["early_line3"], f["early_line4"] = float((e == 3).mean()), float((e == 4).mean())  # 早期三線與四線的比例
    return np.array([f[k] for k in FEATURE_NAMES])  # 依固定順序輸出特徵列


class StyleModel:  # 高斯（類 LDA）的風格模型
    """fit({player_id: [n_games, d] feature rows}); log_scores([m, d] query rows) -> [n_players] in self.players' order."""

    def __init__(self, shrink=0.1):  # 設定共變異數的收縮比例
        if not 0.0 <= shrink <= 1.0:  # 收縮比例須在 [0, 1] 之間
            raise ValueError("shrink in [0, 1]")  # 拒絕不合法的參數
        self.shrink = float(shrink)  # 保存參數

    def fit(self, feats_by_player):  # 以候選人的特徵列擬合
        self.players = sorted(feats_by_player)  # 候選人依 id 排序
        X = [np.asarray(feats_by_player[p], dtype=float) for p in self.players]  # 每位候選人的特徵矩陣
        self.n_games = np.array([len(x) for x in X], dtype=float)  # 每位候選人的局數
        self.means = np.stack([x.mean(axis=0) for x in X])  # 每位候選人的平均
        resid = np.vstack([x - x.mean(axis=0) for x in X])  # 減去各自平均後的殘差
        dof = max(len(resid) - len(X), 1)  # 合併共變異數的自由度
        cov = resid.T @ resid / dof  # 合併的候選人內共變異數
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov))  # 向對角線收縮
        cov += np.eye(cov.shape[0]) * 1e-9 * max(np.trace(cov) / cov.shape[0], 1e-12)  # 加上極小的脊項，確保可逆
        self.cov = cov  # 保存共變異數
        self.prec = np.linalg.inv(cov)  # 精度矩陣
        return self  # 回傳自身以便串接

    def log_scores(self, query_rows):  # 查詢對每位候選人的對數分數
        q = np.asarray(query_rows, dtype=float)  # 查詢的特徵列
        m = max(len(q), 1)  # 查詢的局數
        d = self.means - q.mean(axis=0)  # 每位候選人的平均與查詢平均之差
        scale = 1.0 / m + 1.0 / self.n_games  # [C]：兩邊平均的變異數比例 1/m + 1/n_c
        quad = np.einsum("cd,de,ce->c", d, self.prec, d) / scale  # 馬氏距離的平方，依兩邊局數縮放
        return -0.5 * quad  # 高斯對數概似（省略常數項）

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"shrink": self.shrink, "players": list(self.players), "features": list(FEATURE_NAMES)}  # 參數、候選人與特徵名稱
