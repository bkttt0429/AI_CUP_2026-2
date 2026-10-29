import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import aicup_pid  # puts AI_CUP_2026-1 on sys.path

AICUP1 = aicup_pid.AICUP1
