#!/usr/bin/env bash
# METHOD_NOTES 2.102, probe P, steps 2-4 (after scripts/nano4/pid_exp_probep_x.sh):
#   2. row level: scripts/pid_probe_polrows.py (shares the first GPU with a trainer)   -> results/final/v2_probe_polrows.json
#   3. per-ply, the recipe of probe B (folds 1 and 2 left out, symmetry + ply masking, 10,000 steps), two seeds each:
#        p2 / p2b   the pooled half of the input replaced by the existing supervised projection of [pooled, played - EXPECTED]
#                   (probe B's q2 with the expected vector subtracted)
#        p4 / p4b   all 128 input dimensions = [the existing supervised features, the expected vector through the same
#                   projection] (probe B's q3 plus the expected vector)
#   4. fold vectors (also of probe B's q0 / q2 / q3, whose vectors were removed), the simulated-pool comparison against q2
#      (results/final/v2_sim14p.json) and the comparison with the row-embedding fusion (v2_probe_p_fuse.json)
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local C="--steps 10000 --horizon 30000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only" G p pids=() rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -ge 8 ] || { echo "need 8 GPUs, have ${#G[@]}"; return 1; }
  echo "rows $(ls $D/rows | grep -c npz) parts, per-ply $(ls $D/ply_r | grep -c npz) / $(ls $D/ply_fc | grep -c npz); $(du -sh $D | cut -f1)"
  CUDA_VISIBLE_DEVICES=${G[0]} python -I $R2/scripts/pid_probe_polrows.py --out $R2/results/final/v2_probe_polrows.json > logs/nano4/polrows-$J.log 2>&1 &
  local rowpid=$!
  CUDA_VISIBLE_DEVICES=${G[0]},${G[1]},${G[2]},${G[3]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_r" \
    "1 p2 $C --sup-dims 0 64" "2 p2 $C --sup-dims 0 64" "1 p2b $C --sup-dims 0 64 --seed 1" "2 p2b $C --sup-dims 0 64 --seed 1" > logs/nano4/poltrain_r-$J.log 2>&1 &
  pids+=($!)
  CUDA_VISIBLE_DEVICES=${G[4]},${G[5]},${G[6]},${G[7]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_fc" \
    "1 p4 $C --sup-dims 0 128" "2 p4 $C --sup-dims 0 128" "1 p4b $C --sup-dims 0 128 --seed 1" "2 p4b $C --sup-dims 0 128 --seed 1" > logs/nano4/poltrain_fc-$J.log 2>&1 &
  pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a trainer group FAILED"; rc=1; }; done
  wait $rowpid || { echo "the row-level probe FAILED"; rc=1; }
  grep -E "^(emb|lda) |new path|rows of|->" logs/nano4/polrows-$J.log
  tail -n 3 logs/nano4/poltrain_r-$J.log logs/nano4/poltrain_fc-$J.log
  emb() {  # <GPU> <tag> [supervised options]
    CUDA_VISIBLE_DEVICES=$1 python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix "_$2" --name "_n$2" --out $R2/data/seq "${@:3}" > logs/nano4/polemb_$2-$J.log 2>&1
  }
  pids=()
  emb ${G[0]} p2e --sup-ply $D/ply_r --sup-dims 0 64 & pids+=($!)
  emb ${G[1]} p2be --sup-ply $D/ply_r --sup-dims 0 64 & pids+=($!)
  emb ${G[2]} p4e --sup-ply $D/ply_fc --sup-dims 0 128 & pids+=($!)
  emb ${G[3]} p4be --sup-ply $D/ply_fc --sup-dims 0 128 & pids+=($!)
  emb ${G[4]} q0e & pids+=($!)
  emb ${G[5]} q2e --sup-ply $B/ply --sup-dims 0 64 & pids+=($!)
  emb ${G[6]} q3e --sup-ply $B/ply --sup-dims 64 128 --sup-only & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a fold-vector run FAILED"; rc=1; }; done
  tail -q -n 1 logs/nano4/polemb_*-$J.log | cut -c1-200
  python -I $R2/scripts/pid_v2_sim14.py --names q2e=_nq2e q0e=_nq0e q3e=_nq3e p2e=_np2e p2be=_np2be p4e=_np4e p4be=_np4be --ref q2e --folds 1 2 --out $R2/results/final/v2_sim14p.json || rc=1
  python -I $R2/scripts/pid_probe_b_fuse.py --tags q2e q0e q3e p2e p2be p4e p4be --ref q2e --out $R2/results/final/v2_probe_p_fuse.json || rc=1
  return $rc
}
main "$@"; exit $?
