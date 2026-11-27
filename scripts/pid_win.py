"""METHOD_NOTES 2.170 (probe W): the stones around the played point as an explicit input of the sequence network.

The network of m88_seqid reads, per ply, two projected trunk vectors and the played point; what the board looks like around
the move reaches it only through the 64 projected dimensions of the trunk at that point and through attention over the
earlier plies' point embeddings.  Here every ply also gets the size x size neighbourhood of its point, read by a small
convolutional encoder that is trained with the network:

  cell states   0 empty, 1 a stone of the mover, 2 a stone of the other side, 3 off the board,
                4 the stone of the previous ply, 5 the stone of the ply before that (the mover's own last move)
  the board     the stones PLAYED so far, the latest one on a point (captures are not replayed: a captured stone stays
                until something is played there again) - built on the GPU from the plies' points and colours, so no feature
                extraction and no file is needed, and the trainer's board symmetries / hidden plies apply to it as to the points.

wrap(net, cfg) turns a built m88_seqid network into this variant in place (its class gets a forward with one more term; the
new term's last layer starts at zero, so the wrapped network starts as the plain one).  A checkpoint of such a network
carries cfg under "win"; the loaders call wrap() when they find it.
"""


def windows(pos, mov, size):  # 組出每一手落點周圍的視窗
    """pos [B, T] points (y * 19 + x; 361 pass, 362 padding / hidden), mov [B, T] colours -> [B, T, size * size] cell states
    of the board before each ply around its point, and [B, T] whether the ply has a point."""
    import torch  # 延後匯入，模組本身不依賴 torch
    import torch.nn.functional as F  # 補邊用的 pad
    B, T = pos.shape  # 批次大小與手數
    dev = pos.device  # 在輸入所在的裝置上計算
    ok = pos < 361  # 有落點的手（不是 pass／補齊）
    t = torch.arange(1, T + 1, device=dev, dtype=torch.int16)[None].expand(B, T)  # 每手的手數（1 起算）
    stamp = torch.zeros(B, T, 363, dtype=torch.int16, device=dev)  # 棋子落下處記為（手數）* 2 + 顏色
    stamp.scatter_(2, pos[..., None], torch.where(ok, t * 2 + mov.to(torch.int16), torch.zeros_like(t))[..., None])  # 每手只在自己的落點蓋章
    last = torch.cummax(stamp[..., :361], dim=1).values  # 每一手之後各點上最新的棋子
    before = torch.zeros_like(last)  # 落子「之前」的盤面
    before[:, 1:] = last[:, :-1]  # 往後錯開一手
    del stamp, last  # 釋放大張量
    ply = (before >> 1).long()  # 0 = 空，否則是該子的手數（1 起算）
    col = (before & 1).long()  # 該子的顏色
    now = torch.arange(1, T + 1, device=dev)[None, :, None]  # 目前這一手的手數（1 起算）
    state = torch.where(ply == 0, 0, torch.where(col == mov[..., None], 1, 2))  # 空、己方或對方的子
    state = torch.where((ply > 0) & (ply == now - 1), 4, state)  # 上一手的子
    state = torch.where((ply > 0) & (ply == now - 2), 5, state)  # 再上一手（自己的上一手）的子
    del ply, col, before  # 釋放記憶體
    p = size // 2  # 視窗半徑
    W = 19 + 2 * p  # 補邊後的盤面寬度
    board = F.pad(state.reshape(B, T, 19, 19), (p, p, p, p), value=3).reshape(B, T, W * W)  # 四周補上「盤外」= 3
    cen = torch.where(ok, pos, torch.zeros_like(pos))  # 視窗中心；沒有落點的手先用 0 代替
    d = torch.arange(size, device=dev)  # 視窗內的位移
    off = (d[:, None] * W + d[None, :]).reshape(-1)  # 視窗各格相對其角落 (y - p, x - p) 的位移（補邊座標）
    idx = ((cen // 19) * W + cen % 19)[..., None] + off[None, None]  # 每格在補邊盤面中的索引
    return board.gather(2, idx), ok  # 視窗格子與有落點的遮罩


def wrap(net, cfg):  # 為網路加上視窗編碼器（也是各變體的分派入口）
    """Adds the window encoder to a built m88_seqid.SeqID (in place) and returns it. cfg: {"size": odd, "ch": channels}."""
    if cfg.get("exact"):  # METHOD_NOTES 2.180：精確盤面（移除被提的子）在 pid_winx.py
        import sys  # 調整搜尋路徑
        from pathlib import Path  # 取本檔所在的資料夾
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的模組可以匯入
        import pid_winx  # 精確視窗的實作
        return pid_winx.wrap(net, cfg)  # 交給它包裝
    if cfg.get("age") or cfg.get("prev"):  # METHOD_NOTES 2.192：視窗變體（棋子年齡、第二個視窗）在 pid_win2.py
        import sys  # 調整搜尋路徑
        from pathlib import Path  # 取本檔所在的資料夾
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # 讓同資料夾的模組可以匯入
        import pid_win2  # 視窗變體的實作
        return pid_win2.wrap(net, cfg)  # 交給它包裝
    import torch  # 一般的視窗在這裡建立
    from torch import nn  # 神經網路模組
    size, ch = int(cfg.get("size", 9)), int(cfg.get("ch", 64))  # 視窗大小與通道數
    assert size % 2 == 1 and 3 <= size <= 19  # 視窗必須是奇數且不超過棋盤
    D = net.gl.out_features  # 序列網路的寬度
    base = net.__class__  # 原本的網路類別

    class Win(nn.Module):  # 視窗編碼器
        def __init__(self):  # 建構子
            super().__init__()  # 初始化 nn.Module
            self.emb = nn.Embedding(6, 16)  # 六種格子狀態各一個 16 維嵌入
            self.conv = nn.Sequential(nn.Conv2d(16, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, ch, 3, padding=1, stride=2), nn.GELU(),  # 三層卷積，第二層解析度減半
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.GELU())  # 第三層卷積
            self.out = nn.Linear(2 * ch, D)  # 中心格與全視窗平均接起來，投影到網路寬度
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # 輸出層從零開始，起點等於原網路
            self.size = size  # 記下視窗大小

        def forward(self, pos, mov):  # 每手的視窗向量
            B, T = pos.shape  # 批次大小與手數
            with torch.no_grad():  # 組視窗不需要梯度
                cells, ok = windows(pos, mov, self.size)  # 格子狀態與有落點的遮罩
            flat = ok.reshape(-1)  # 攤平成每手一個旗標
            x = self.emb(cells.reshape(B * T, -1)[flat]).reshape(-1, self.size, self.size, 16).permute(0, 3, 1, 2)  # 只編碼有落點的手，轉成 [N, 16, size, size]
            h = self.conv(x)  # 卷積特徵
            c = h.shape[-1] // 2  # 中心格的位置
            f = self.out(torch.cat([h[:, :, c, c], h.mean((2, 3))], 1))  # 中心格＋全視窗平均 → 網路寬度
            res = torch.zeros(B * T, f.shape[1], dtype=f.dtype, device=f.device)  # pass／補齊的手為零
            res[flat] = f  # 填回有落點的手
            return res.reshape(B, T, -1)  # 還原成每局逐手的輸出形狀 [B, T, D]

    class SeqIDW(base):  # 帶視窗輸入的序列網路
        def forward(self, gl, pos, mov, length, subj):  # 與 SeqID.forward 相同，只多了視窗一項
            B, T = pos.shape  # 批次大小與手數
            x = (self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov)  # 主幹向量、落點、是否主體方、顏色的嵌入相加
                + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2)))  # 再加手序位置與兩種位移（對上一手、對自己上一手）的嵌入
            x = x + self.win(pos, mov).to(x.dtype)  # 加上視窗向量
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1)  # 前面接上 [CLS]
            pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # 補齊遮罩：位置 0 是 [CLS]，超過局長的位置不參與注意力
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]  # Transformer 編碼後取 [CLS]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)  # 主體方的單位向量

    net.win = Win().to(net.gl.weight.device)  # 編碼器放到網路所在的裝置
    net.__class__ = SeqIDW  # 就地換掉類別，forward 多了視窗
    net.win_cfg = {"size": size, "ch": ch}  # 在網路物件上記下實際採用的設定，可據此重建同樣的包裝
    return net  # 回傳包裝後的網路


def selftest():  # 自我檢查
    """The window builder against a plain replay (no captures) on random games; the wrapped network against the plain one at start."""
    import numpy as np  # 單純重播用 numpy
    import torch  # 呼叫 windows
    rng = np.random.default_rng(0)  # 固定種子
    B, T, size = 6, 40, 9  # 6 局、40 手、9 x 9 視窗
    pos = rng.integers(0, 361, (B, T)); pos[0, 5] = 361; pos[1, 30:] = 362; pos[2, 7] = pos[2, 3]  # 一手 pass、補齊、同一點下兩次
    mov = np.tile(np.arange(T) % 2, (B, 1))  # 黑白交替
    cells, ok = windows(torch.as_tensor(pos), torch.as_tensor(mov), size)  # 待測的組法
    cells, ok = cells.numpy(), ok.numpy()  # 轉回 numpy 比對
    p = size // 2  # 視窗半徑
    bad = 0  # 不一致的格數
    for b in range(B):  # 每一局
        board = np.zeros((19, 19), int); when = np.zeros((19, 19), int)  # 各點的顏色＋1 與落子手數
        for t in range(T):  # 每一手
            if pos[b, t] < 361:  # 有落點的手
                y, x = divmod(pos[b, t], 19)  # 落點座標
                for i in range(size):  # 視窗的列
                    for j in range(size):  # 視窗的行
                        yy, xx = y - p + i, x - p + j  # 這格在盤上的座標
                        if not (0 <= yy < 19 and 0 <= xx < 19):  # 盤外
                            want = 3  # 狀態 3
                        elif board[yy, xx] == 0:  # 空點
                            want = 0  # 狀態 0
                        elif when[yy, xx] == t:  # 上一手的手數（1 起算）
                            want = 4  # 上一手的子
                        elif when[yy, xx] == t - 1:  # 再上一手
                            want = 5  # 自己上一手的子
                        else:  # 更早的子
                            want = 1 if board[yy, xx] - 1 == mov[b, t] else 2  # 己方 1、對方 2
                        bad += int(cells[b, t, i * size + j] != want)  # 累計不一致
                board[y, x] = mov[b, t] + 1; when[y, x] = t + 1  # 落子（不提子）
            assert ok[b, t] == (pos[b, t] < 361)  # 遮罩也要一致
    print(f"window builder against a plain replay: {bad} differing cells of {B * T * size * size}")  # 回報結果
    return bad == 0  # 全部一致才算通過


if __name__ == "__main__":  # 直接執行時跑自我檢查
    raise SystemExit(0 if selftest() else 1)  # 通過時結束碼為 0
