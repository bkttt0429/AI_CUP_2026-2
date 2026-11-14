#!/usr/bin/env bash
# METHOD_NOTES 2.101: official networks on SUPERVISED per-ply input at the scale of the composition's members - competition
# games of folds 1-4 (never fold 0) plus the 116,420 external accounts, whose supervised features come from
# scripts/nano4/pid_exp_supext.sh. The full-scale form of o11U (which has no external accounts). Symmetry + ply masking 0.3,
# 30,000 steps with a snapshot at 20,000 (the same schedule, so the snapshot is a 20,000-step network), EMA only.
# Families by what the network reads:
#   o12A  the board-pooled half of the input replaced by the supervised features (probe B's best single network)
#   o12B  the played-point half replaced
#   o12C  the supervised features alone (nothing of the random projection: the member most unlike the existing ones)
#   pid_exp_o12.sh a   o12As0-3 + o12Bs0-3
#   pid_exp_o12.sh b   o12As4-7 + o12Cs0-3
# The checkpoints record their input, so the pool fine-tune applies it by itself (scripts/pid_v2_seq_finetune.py).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probec E=/work/u2507115/pid_v2/sup_ext
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $B/ply $D/ply $E/ply --sup-stats $B/stats.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 20000" specs=() k
  case $1 in
    a) for k in 0 1 2 3; do specs+=("-1 o12As$k $C --sup-dims 0 64 --seed $k"); done
       for k in 0 1 2 3; do specs+=("-1 o12Bs$k $C --sup-dims 64 128 --seed $((k + 10))"); done ;;
    b) for k in 4 5 6 7; do specs+=("-1 o12As$k $C --sup-dims 0 64 --seed $k"); done
       for k in 0 1 2 3; do specs+=("-1 o12Cs$k $C --sup-dims 64 128 --sup-only --seed $((k + 20))"); done ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
  echo "supervised parts: competition $(ls $B/ply $D/ply | grep -c npz), external $(ls $E/ply | grep -c npz)"
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
