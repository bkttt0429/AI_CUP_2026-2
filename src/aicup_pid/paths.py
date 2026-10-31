"""Where the shared code and data live.

The SGF parser, the training-set readers, the KataGo features and the nano4 scripts are AI_CUP_2026-1's `aicup`
package. Its checkout is found from AICUP1_ROOT, else a sibling directory named AI_CUP_2026-1 / AICUP_2026-1 /
ai_cup_2026-1 of this repository; its `src` and `scripts` go on sys.path. The training CSVs are AICUP_TRAIN_DIR,
else AI_CUP_2026-1's data/training.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIBLINGS = ("AI_CUP_2026-1", "AICUP_2026-1", "ai_cup_2026-1", "BK_AICUP_2026-1")


def aicup1_root():
    """The AI_CUP_2026-1 checkout (FileNotFoundError when none is found)."""
    env = os.environ.get("AICUP1_ROOT")
    cands = [Path(env)] if env else [ROOT.parent / s for s in SIBLINGS]
    for c in cands:
        if (c / "src" / "aicup" / "__init__.py").exists():
            return c.resolve()
    raise FileNotFoundError("AI_CUP_2026-1 not found: set AICUP1_ROOT or check it out next to this repository "
                            f"(tried {[str(c) for c in cands]})")


AICUP1 = aicup1_root()
for d in (AICUP1 / "src", AICUP1 / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

TRAIN_DIR = Path(os.environ.get("AICUP_TRAIN_DIR", AICUP1 / "data" / "training"))
FOLDS_CSV = AICUP1 / "splits" / "player_folds.csv"
