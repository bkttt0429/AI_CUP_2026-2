"""Score submissions on the void 10-30 Public pool against its known answers (the Top-1 of results/probe/upload_fix.csv,
Public 1.0000000): official score, Top-1 errors, and the errors' question ids. Usage: pid_score_public_truth.py sub.csv ..."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
truth = pd.read_csv(ROOT / "results/probe/upload_fix.csv", dtype=str)
truth = truth[truth.question_id.str.startswith("pub_")]  # only the Public Top-1 are known answers; its Private rows are predictions
truth = dict(zip(truth.question_id, truth.top_1))
for path in sys.argv[1:]:
    s = pd.read_csv(path, dtype=str)
    s = s[s.question_id.isin(truth)]
    tops = s[[f"top_{i}" for i in range(1, 6)]].values
    sc = [np.exp(-list(t).index(truth[q])) if truth[q] in t else 0.0 for q, t in zip(s.question_id, tops)]
    wrong = [q for q, t in zip(s.question_id, tops) if t[0] != truth[q]]
    print(f"{path}: Public {np.mean(sc):.7f} over {len(s)}, Top-1 wrong {len(wrong)}: {' '.join(wrong[:12])}")
