#!/usr/bin/env bash
# METHOD_NOTES 2.164: acceptance of the lossless re-compression (scripts/pid_npz_recompress.py), independent of the
# converter's own per-file comparison:
#   1  the final file rebuilt from the converted real-pool vectors must be byte-identical to upload/pid_v2bgm_d2_a2k_p3.csv
#   2  the trainer's own loader reads every training archive (competition + external per-ply parts, second feature set):
#      the counts must be the ones of the o25 training of 11-15 (2,179,740 games = 323,364 stored + 1,856,376 extracted;
#      supervised features for 2,055,047 games, 68 dimensions)
# From the AI_CUP_2026-1 checkout, 2 GPUs' worth of memory (about 215 GB is loaded). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj J=${SLURM_JOB_ID:-local} a b rc=0
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  df -h /work/u2507115 /home/u2507115 | tail -2
  bash $R2/scripts/nano4/pid_v2_make_file4.sh v2bgm_d2 zz_rccheck_v2bgm_d2.csv 2 0.5 0.05 > logs/nano4/rccheck-$J-file.log 2>&1 || { echo "the file was not built:"; tail -5 logs/nano4/rccheck-$J-file.log; rc=1; }
  a=$(md5sum $R2/upload/zz_rccheck_v2bgm_d2.csv 2> /dev/null | cut -c1-32); b=$(md5sum $R2/upload/pid_v2bgm_d2_a2k_p3.csv | cut -c1-32)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "1  final file rebuilt from the converted vectors: IDENTICAL ($a)"; else echo "1  final file rebuilt: DIFFERENT (rebuilt '$a', final $b)"; rc=1; fi
  rm -f /home/u2507115/BK_AICUP_2026-2/upload/zz_rccheck_v2bgm_d2.csv
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train.py --net -1 --tag rccheck --prepare-only $OFF $SUP --sup-require --sup-name js --sup-dims 0 68 > logs/nano4/rccheck-$J-load.log 2>&1 || { echo "the loader failed:"; tail -8 logs/nano4/rccheck-$J-load.log; rc=1; }
  grep -E "games loaded|supervised per-ply features" logs/nano4/rccheck-$J-load.log | cut -c1-260 | sed 's/^/2  /'
  grep -q "games loaded: 2179740 (323364 stored of folds 1-4 + 1856376 extracted" logs/nano4/rccheck-$J-load.log && grep -q "supervised per-ply features: 2055047 of 2179740 games, 68 dimensions" logs/nano4/rccheck-$J-load.log \
    && echo "2  loader counts: AS BEFORE" || { echo "2  loader counts: DIFFERENT from the reference"; rc=1; }
  return $rc
}
main "$@"; exit $?
