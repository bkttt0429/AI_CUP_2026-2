#!/usr/bin/env bash
# METHOD_NOTES 2.128: the by-games held-out test (notes 2.124, job 779827) lost two of its four networks when /work ran
# full at 13:38 (fold 1 gb and fold 2 ga failed writing their final checkpoints; fold 1 ga and fold 2 gb were written).
#   now:   a partial reading on the two networks that exist - ga on the two pools of fold 1, gb on the two pools of fold 2,
#          each against zme on the same pools (results/final/v2_sim14g_f1.json, v2_sim14g_f2.json);
#   then:  the two lost networks trained again with the same options and seed (the same job would have written them),
#          and the registered four-pool test: results/final/v2_sim14g.json.
# From the AI_CUP_2026-1 checkout, 3 GPUs (two trainers + one for the vectors). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ext-share 0.5 --ema-only"
  local G; IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
  (
    export CUDA_VISIBLE_DEVICES=${G[2]}
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 --ckpt-dir $S --ckpt-suffix _gae --name _ngae --out $R2/data/seq || echo "embedding of gae (fold 1) failed"
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 2 --ckpt-dir $S --ckpt-suffix _gbe --name _ngbe --out $R2/data/seq || echo "embedding of gbe (fold 2) failed"
    python -I $R2/scripts/pid_v2_sim14.py --names zme=_nzme gae=_ngae --ref zme --folds 1 --out $R2/results/final/v2_sim14g_f1.json
    python -I $R2/scripts/pid_v2_sim14.py --names zme=_nzme gbe=_ngbe --ref zme --folds 2 --out $R2/results/final/v2_sim14g_f2.json
  ) > logs/nano4/bygames2-$J-partial.log 2>&1 &
  local p_part=$!
  CUDA_VISIBLE_DEVICES=${G[0]},${G[1]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "1 gb $C --own-by-games sqrt" "2 ga $C --own-by-games linear" || { echo "training failed"; wait $p_part; cat logs/nano4/bygames2-$J-partial.log; return 1; }
  wait $p_part
  echo "== partial reading (written at the start of the job)"; cat logs/nano4/bygames2-$J-partial.log
  export CUDA_VISIBLE_DEVICES=${G[2]}
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 2 --ckpt-dir $S --ckpt-suffix _gae --name _ngae --out $R2/data/seq || echo "embedding of gae (fold 2) failed"
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 --ckpt-dir $S --ckpt-suffix _gbe --name _ngbe --out $R2/data/seq || echo "embedding of gbe (fold 1) failed"
  echo "== the registered test (four pools)"
  python -I $R2/scripts/pid_v2_sim14.py --names zme=_nzme yse=_nyse gae=_ngae gbe=_ngbe --ref zme --folds 1 2 --out $R2/results/final/v2_sim14g.json
}
main "$@"; exit $?
