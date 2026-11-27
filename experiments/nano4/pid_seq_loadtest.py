#!/usr/bin/env python
"""Memory check of the training loader (scripts/pid_seq_train.py load_games): load the games exactly as a training run would and
print the size of the arrays and the memory the process holds. No network, no fold 0. Run from the AI_CUP_2026-1 checkout."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pid_seq_train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ext-ply", nargs="*", default=None)
ap.add_argument("--ext-select", nargs="*", default=None)
a = ap.parse_args()
t0 = time.time()
G = T.load_games(False, lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True), False, a.ext_ply, a.ext_select)
print(f"arrays {sum(x.nbytes for x in G.values()) / 1e9:.1f} GB; process memory now {T._rss_gb():.1f} GB", flush=True)
