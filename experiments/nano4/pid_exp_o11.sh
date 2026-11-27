#!/usr/bin/env bash
# METHOD_NOTES 2.100: official networks on SUPERVISED per-ply input (probe B: +0.048 alone and +0.021 beyond the row embedding at
# small scale). Whatever probe C says about the size of the gain at scale, these networks read a different representation
# than every other member of the composition, which is what the composition lacks (2.96, 2.99).
#   pid_exp_o11.sh pools   supervised features of the official pools' games (8 shards)
#   pid_exp_o11.sh train   eight official networks, competition games only (net -1 = folds 1-4, never fold 0; the external
#                          games have no supervised features yet), symmetry + ply masking 0.3, 30,000 steps, EMA only:
#                          o11Us0-3 the pooled half of the input replaced, o11Us4-7 the played-point half replaced
# The checkpoints record their input, so the pool fine-tune applies it by itself (scripts/pid_v2_seq_finetune.py).
# From the AI_CUP_2026-1 checkout, 8 GPUs. The body is a function called on the last line with exit.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probec P=/work/u2507115/pid_v2/sup_pools
  case $1 in
    pools)
      mkdir -p $P
      bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --pools $R2/data/tests_1111 --proj $B/proj.npz --stats $B/stats.npz --out $P/ply --workers 10
      echo "pool parts: $(ls $P/ply | grep -c npz), $(du -sh $P/ply | cut -f1)" ;;
    train)
      local SUP="--sup-ply $B/ply $D/ply --sup-stats $B/stats.npz" C="--steps 30000 --horizon 60000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only" specs=() k
      for k in 0 1 2 3; do specs+=("-1 o11Us$k $C --sup-dims 0 64 --seed $k"); done
      for k in 4 5 6 7; do specs+=("-1 o11Us$k $C --sup-dims 64 128 --seed $k"); done
      bash $R2/scripts/nano4/pid_seq_train_fork.sh "$SUP" "${specs[@]}" ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
}
main "$@"; exit $?
