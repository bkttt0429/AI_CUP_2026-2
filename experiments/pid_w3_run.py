#!/usr/bin/env python
"""METHOD_NOTES 2.202: runs another script with pid_win3's dispatch installed, so that a loader (pid_v2_seq_finetune.py,
pid_v2_seq_embed.py) can rebuild a network whose stored window cfg has "own2" or "depth" WITHOUT a change to the shared
modules pid_win.py / pid_win2.py (other jobs' steps import them all night). Every other cfg is handled as before.

  python -I scripts/pid_w3_run.py <script> <the script's arguments>

Only for the short runs of scripts/nano4/pid_exp_early3.sh; a variant that is expanded to a full family gets the permanent
dispatch in pid_win.py after a regression check on existing checkpoints.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pid_win3  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: pid_w3_run.py <script> [arguments]")
    pid_win3.install()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
