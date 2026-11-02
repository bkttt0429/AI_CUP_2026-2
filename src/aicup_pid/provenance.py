"""What every result file of this repository records: aicup.ladder.environment (AI_CUP_2026-1's commit, code hashes, torch,
device) plus this repository's own commit, since the two checkouts move independently."""
import subprocess

from aicup.ladder import environment

from aicup_pid.paths import ROOT


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def pid_environment(device="cpu"):
    """environment(device) with pid_git_head (short commit of AI_CUP_2026-2, None outside a checkout) and pid_git_dirty."""
    head = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {**environment(device), "pid_git_head": head, "pid_git_dirty": None if status is None else bool(status)}
