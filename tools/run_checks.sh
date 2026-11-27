#!/usr/bin/env bash
# 執行本儲存庫的全部規則檢查；回報完成或 commit 前都要執行（AGENTS.md 第 7 節）。
# 用法：bash tools/run_checks.sh --all
# 結束碼：0 全部通過；1 有檢查失敗；2 環境不完整，有檢查沒有執行（不算完成）。
set -uo pipefail  # 未定義的變數視為錯誤，管線任一段失敗都算失敗
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # 儲存庫根目錄（本檔位於 tools/）
cd "$ROOT" || exit 1  # 所有檢查都在根目錄執行
MODE="${1:---all}"  # 預設檢查全專案
fail=0  # 有檢查失敗時設為 1
skipped=0  # 有檢查因環境不完整而略過時設為 1

step() {  # 印出步驟標題
  printf '\n== %s\n' "$*"  # 空一行再印標題，方便閱讀
}  # 結束函式定義

find_python() {  # 找出要使用的 Python
  local cand  # 迴圈中的候選路徑
  for cand in "${PYTHON:-}" "$ROOT/.venv/bin/python" "$ROOT/../AI_CUP_2026-1/.venv/bin/python"; do  # 依序：環境變數、本儲存庫、並排的競賽二儲存庫
    if [ -n "$cand" ] && [ -x "$cand" ]; then  # 候選存在且可執行
      echo "$cand"  # 回傳這個 Python
      return 0  # 找到就結束
    fi  # 結束判斷
  done  # 結束迴圈
  command -v python3 || true  # 最後退回系統的 python3
}  # 結束函式定義

PY="$(find_python)"  # 決定使用的 Python
if [ -z "$PY" ]; then  # 完全找不到 Python
  echo "錯誤：找不到 Python，無法檢查"  # 說明原因
  exit 2  # 環境不完整
fi  # 結束判斷
echo "使用 $PY（$("$PY" -V 2>&1)）"  # 印出實際使用的直譯器

step "規則檢查：中文註解、重複函式、封存區、文件規則"  # 第一項檢查
"$PY" -I tools/check_code_rules.py "$MODE" || fail=1  # 違規時記為失敗

step "ruff"  # 第二項檢查
if "$PY" -m ruff --version >/dev/null 2>&1; then  # 環境內有 ruff
  "$PY" -m ruff check --isolated --select E9,F63,F7,F82 src scripts tests tools || fail=1  # 全範圍只查語法錯誤與未定義名稱（--isolated：不受任何設定檔影響）
  "$PY" -m ruff check --isolated --select E4,E7,E9,F tools || fail=1  # 新寫的 tools/ 套用 ruff 的基本規則（含未使用的 import 與變數）
else  # 環境內沒有 ruff
  echo "警告：找不到 ruff，略過；請安裝 requirements-dev.txt"  # 提示安裝方式
  skipped=1  # 記為略過
fi  # 結束判斷

step "pytest"  # 第三項檢查
if [[ "$(hostname)" == *-lgn* ]]; then  # 國網的登入節點
  echo "警告：國網規定登入節點不可執行測試；請以 sbatch 在計算節點執行本腳本"  # 說明為何略過
  skipped=1  # 記為略過
elif [ ! -d "$ROOT/../AI_CUP_2026-1/src/aicup" ]; then  # 找不到並排的競賽二儲存庫
  echo "警告：本儲存庫的測試需要並排的 AI_CUP_2026-1，略過"  # 說明為何略過
  skipped=1  # 記為略過
elif "$PY" -c "import pytest" >/dev/null 2>&1; then  # 環境內有 pytest
  (cd "$ROOT/../AI_CUP_2026-1" && "$PY" -I -m pytest -q -p no:cacheprovider "$ROOT/tests") || fail=1  # 依 README 從競賽二的根目錄執行測試，不留快取目錄
else  # 環境內沒有 pytest
  echo "警告：找不到 pytest，略過；請先建立環境"  # 提示建立環境
  skipped=1  # 記為略過
fi  # 結束判斷

step "結果"  # 總結
if [ "$fail" -ne 0 ]; then  # 有失敗
  echo "有檢查未通過"  # 回報失敗
  exit 1  # 失敗
fi  # 結束判斷
if [ "$skipped" -ne 0 ]; then  # 有略過
  echo "有檢查沒有執行，不算完成"  # 回報未完成
  exit 2  # 未完成
fi  # 結束判斷
echo "全部通過"  # 全部通過
