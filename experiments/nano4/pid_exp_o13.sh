#!/usr/bin/env bash
# METHOD_NOTES 2.110: the two things that work, together - SUPERVISED per-ply input (o12: the strongest single networks on the
# official candidates) and LONGER training (o7F). Official networks (competition folds 1-4, never fold 0; 116,420 external
# accounts), symmetry + ply masking 0.3, 40,000 steps on the 60,000-step cosine with a snapshot at 30,000, EMA only.
#   o13A  the pooled half of the input replaced by the supervised features (o12's best kind: 0.544 / 0.557 at n=1)
#   o13C  the supervised features alone (0.534 / 0.548; complementary to A)
#   pid_exp_o13.sh a   o13As0-5 (half of an episode external), o13As6-7 (70 % external)
#   pid_exp_o13.sh b   o13As8-9 (half external), o13As10-11 (70 % external), o13Cs0-3
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probec E=/work/u2507115/pid_v2/sup_ext
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $B/ply $D/ply $E/ply --sup-stats $B/stats.npz"
  local C="--steps 40000 --horizon 60000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 30000" specs=() k
  case $1 in
    a) for k in 0 1 2 3 4 5; do specs+=("-1 o13As$k $C --ext-share 0.5 --sup-dims 0 64 --seed $((k + 30))"); done
       for k in 6 7; do specs+=("-1 o13As$k $C --ext-share 0.7 --sup-dims 0 64 --seed $((k + 30))"); done ;;
    b) for k in 8 9; do specs+=("-1 o13As$k $C --ext-share 0.5 --sup-dims 0 64 --seed $((k + 30))"); done
       for k in 10 11; do specs+=("-1 o13As$k $C --ext-share 0.7 --sup-dims 0 64 --seed $((k + 30))"); done
       for k in 0 1 2 3; do specs+=("-1 o13Cs$k $C --ext-share 0.5 --sup-dims 64 128 --sup-only --seed $((k + 50))"); done ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
