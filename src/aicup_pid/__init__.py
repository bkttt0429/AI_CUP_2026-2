"""Player identification (AI CUP 2026 competition 1): pools, opening language model, raw style statistics, fusion.

Importing the package makes the rank competition's `aicup` package (AI_CUP_2026-1) importable, see aicup_pid.paths.
"""
from aicup_pid import paths  # 匯入時即尋找 AI_CUP_2026-1 並設定 sys.path

AICUP1, TRAIN_DIR = paths.AICUP1, paths.TRAIN_DIR  # 對外提供 AI_CUP_2026-1 的位置與訓練資料夾
