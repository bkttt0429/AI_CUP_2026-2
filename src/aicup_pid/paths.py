"""Where the shared code and data live.

The SGF parser, the training-set readers, the KataGo features and the nano4 scripts are AI_CUP_2026-1's `aicup`
package. Its checkout is found from AICUP1_ROOT, else a sibling directory named AI_CUP_2026-1 / AICUP_2026-1 /
ai_cup_2026-1 of this repository; its `src` and `scripts` go on sys.path. The training CSVs are AICUP_TRAIN_DIR,
else AI_CUP_2026-1's data/training.
"""
import os  # 讀取環境變數
import sys  # 修改模組搜尋路徑
from pathlib import Path  # 路徑處理

ROOT = Path(__file__).resolve().parents[2]  # 本儲存庫根目錄（src/aicup_pid 往上兩層）
SIBLINGS = ("AI_CUP_2026-1", "AICUP_2026-1", "ai_cup_2026-1", "BK_AICUP_2026-1")  # 可接受的同層資料夾名稱


def aicup1_root():  # 尋找 AI_CUP_2026-1 的位置
    """The AI_CUP_2026-1 checkout (FileNotFoundError when none is found)."""
    env = os.environ.get("AICUP1_ROOT")  # 讀環境變數 AICUP1_ROOT；有設定時只試這個位置
    cands = [Path(env)] if env else [ROOT.parent / s for s in SIBLINGS]  # 要嘗試的位置：環境變數指定的一處，否則本儲存庫旁的各個同層資料夾
    for c in cands:  # 逐一嘗試
        if (c / "src" / "aicup" / "__init__.py").exists():  # 有 aicup 套件才算找到
            return c.resolve()  # 回傳絕對路徑
    raise FileNotFoundError("AI_CUP_2026-1 not found: set AICUP1_ROOT or check it out next to this repository "  # 都找不到就中止
                            f"(tried {[str(c) for c in cands]})")  # 錯誤訊息附上嘗試過的位置


AICUP1 = aicup1_root()  # 匯入時即決定位置
for d in (AICUP1 / "src", AICUP1 / "scripts"):  # 讓 aicup 套件與其腳本可以匯入
    if str(d) not in sys.path:  # 避免重複加入
        sys.path.insert(0, str(d))  # 放在搜尋路徑最前面

TRAIN_DIR = Path(os.environ.get("AICUP_TRAIN_DIR", AICUP1 / "data" / "training"))  # 訓練 CSV 的資料夾：環境變數 AICUP_TRAIN_DIR，否則 AI_CUP_2026-1 的 data/training
FOLDS_CSV = AICUP1 / "splits" / "player_folds.csv"  # 玩家分折檔
