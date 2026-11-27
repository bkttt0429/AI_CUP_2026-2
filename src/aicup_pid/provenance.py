"""What every result file of this repository records: aicup.ladder.environment (AI_CUP_2026-1's commit, code hashes, torch,
device) plus this repository's own commit, since the two checkouts move independently."""
import subprocess  # 呼叫 git

from aicup.ladder import environment  # AI_CUP_2026-1 的環境紀錄（commit、程式雜湊、torch 版本、裝置）

from aicup_pid.paths import ROOT  # 本儲存庫的根目錄


def _git(*args):  # 在本儲存庫執行一個 git 指令
    try:  # git 不存在或不在 checkout 中時不中止
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=20).stdout.strip()  # 回傳去掉前後空白的輸出
    except (OSError, subprocess.SubprocessError):  # 執行失敗
        return None  # 以 None 表示無法取得


def pid_environment(device="cpu"):  # 結果檔要記錄的環境資訊
    """environment(device) with pid_git_head (short commit of AI_CUP_2026-2, None outside a checkout) and pid_git_dirty."""
    head = _git("rev-parse", "--short", "HEAD")  # 本儲存庫的短 commit
    status = _git("status", "--porcelain", "--untracked-files=no")  # 已追蹤的檔案是否有未提交的修改
    return {**environment(device), "pid_git_head": head, "pid_git_dirty": None if status is None else bool(status)}  # 合併兩個儲存庫的紀錄
