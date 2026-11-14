#!/usr/bin/env bash
# METHOD_NOTES 2.107, probe P6, steps 2-4 (after scripts/nano4/pid_exp_probep6_x.sh): the recipe of probe B (folds 1 and 2 left
# out, symmetry + ply masking, 10,000 steps), the pooled half of the input replaced by the JOINT projection's features:
#   p6 / p6b     64 dims (two seeds)                 p6s / p6sb   64 dims + 4 policy scalars (input dimensions 0-68)
# then fold vectors, the simulated-pool comparison against probe B's q2 (results/final/v2_sim14p6.json; p2 = the first probe's
# residual features, for reference), the comparison with the row-embedding fusion (v2_probe_p6_fuse.json), and LAST the
# row-level probe that was stopped in the first job (v2_probe_polrows.json).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2/probep S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local C="--steps 10000 --horizon 30000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only" G p pids=() rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -ge 8 ] || { echo "need 8 GPUs, have ${#G[@]}"; return 1; }
  CUDA_VISIBLE_DEVICES=${G[0]},${G[1]},${G[2]},${G[3]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_j" \
    "1 p6 $C --sup-dims 0 64" "2 p6 $C --sup-dims 0 64" "1 p6b $C --sup-dims 0 64 --seed 1" "2 p6b $C --sup-dims 0 64 --seed 1" > logs/nano4/p6train_j-$J.log 2>&1 &
  pids+=($!)
  CUDA_VISIBLE_DEVICES=${G[4]},${G[5]},${G[6]},${G[7]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_js" \
    "1 p6s $C --sup-dims 0 68" "2 p6s $C --sup-dims 0 68" "1 p6sb $C --sup-dims 0 68 --seed 1" "2 p6sb $C --sup-dims 0 68 --seed 1" > logs/nano4/p6train_js-$J.log 2>&1 &
  pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a trainer group FAILED"; rc=1; }; done
  tail -n 2 logs/nano4/p6train_j-$J.log logs/nano4/p6train_js-$J.log | cut -c1-200
  emb() {  # <GPU> <tag> [supervised options]
    CUDA_VISIBLE_DEVICES=$1 python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix "_$2" --name "_n$2" --out $R2/data/seq "${@:3}" > logs/nano4/p6emb_$2-$J.log 2>&1
  }
  pids=()
  emb ${G[0]} p6e --sup-ply $D/ply_j --sup-dims 0 64 & pids+=($!)
  emb ${G[1]} p6be --sup-ply $D/ply_j --sup-dims 0 64 & pids+=($!)
  emb ${G[2]} p6se --sup-ply $D/ply_js --sup-dims 0 68 & pids+=($!)
  emb ${G[3]} p6sbe --sup-ply $D/ply_js --sup-dims 0 68 & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a fold-vector run FAILED"; rc=1; }; done
  python -I $R2/scripts/pid_v2_sim14.py --names q2e=_nq2e q0e=_nq0e p2e=_np2e p2be=_np2be p6e=_np6e p6be=_np6be p6se=_np6se p6sbe=_np6sbe --ref q2e --folds 1 2 --out $R2/results/final/v2_sim14p6.json || rc=1
  python -I $R2/scripts/pid_probe_b_fuse.py --tags q2e q0e p2e p6e p6be p6se p6sbe --ref q2e --out $R2/results/final/v2_probe_p6_fuse.json || rc=1
  echo "== main results done at $(date +%H:%M:%S); the row-level probe follows"
  python -I $R2/scripts/pid_probe_polrows.py --emb-seeds 0 --out $R2/results/final/v2_probe_polrows.json || echo "row-level probe failed"
  return $rc
}
main "$@"; exit $?
