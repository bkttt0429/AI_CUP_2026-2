#!/usr/bin/env bash
# Rewrite the checkpoints that hold their state twice with one copy (scripts/pid_ckpt_compact.py; METHOD_NOTES 2.101).
# From the AI_CUP_2026-1 checkout, any node.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet
  du -sh $S
  python -I $R2/scripts/pid_ckpt_compact.py --dir $S "$@"
  local rc=$?
  ls $S/*.compact_tmp 2>/dev/null
  du -sh $S
  return $rc
}
main "$@"; exit $?
