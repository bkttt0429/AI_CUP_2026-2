#!/usr/bin/env bash
# METHOD_NOTES 2.163: lossless re-compression of the project's .npz feature archives (scripts/pid_npz_recompress.py).
#   pid_npz_recompress.sh sample                 measure the gain on 12 files of every directory; nothing is written
#   pid_npz_recompress.sh run work|home|all      convert the directories of this project on that file system
# Only directories of the player-identification project; the rank session's trees are not touched (a few of them are
# sampled read-only for its information). From the AI_CUP_2026-1 checkout, CPU only. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox W=/work/u2507115/pid_v2 mode=${1:-sample} which=${2:-all} d
  local WORK=($X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $X/ext_ply_home/scale1_c $X/ext_ply_home/scale2_top $X/ext_ply_home/scale2_5d
              $W/supj/comp $W/supj/ext $W/supj/pools $W/probep/ply_js $W/probep $W/ply_official $W/sup_pools $W/seq $W/seq_ft $W/seq_opp $W/pair_eval)
  local HOME_=($R2/data/pid_v2_ply_original $R2/data/pid_v2_seq_ft_offsim $R2/data/pid_v2_seq_ft_oldsim $R2/data/seq_ft $R2/data/seq $R2/data/pid_v2_phase_rows $R2/data/mock_official)
  local RANK=(/work/u2507115/aicup1_moved/m90_ply0 /work/u2507115/m197_choice64_v2)
  python -I -m py_compile $R2/scripts/pid_npz_recompress.py || { echo "the script does not compile"; return 1; }
  for d in "${WORK[@]}" "${HOME_[@]}"; do [ -d "$d" ] && [ ! -L "$d" ] || { echo "$d is missing or is a symbolic link"; return 1; }; done
  df -h /work/u2507115 /home/u2507115 | tail -2
  mkdir -p $R2/logs/storage
  case "$mode" in
    sample)
      python -I $R2/scripts/pid_npz_recompress.py --sample 12 --dirs "${WORK[@]}" "${HOME_[@]}"
      echo "== the rank session's trees (read-only sample, for its information)"
      python -I $R2/scripts/pid_npz_recompress.py --sample 8 --dirs "${RANK[@]}" || true ;;
    run)
      local dirs=()
      [ "$which" = test ] && dirs+=($W/pair_eval $W/seq_opp)
      [ "$which" = work ] || [ "$which" = all ] && dirs+=("${WORK[@]}")
      [ "$which" = home ] || [ "$which" = all ] && dirs+=("${HOME_[@]}")
      python -I $R2/scripts/pid_npz_recompress.py --workers "${WORKERS:-16}" --min-free-mb "${FLOOR_MB:-6000}" --method "${METHOD:-deflate}" --level "${LEVEL:-6}" --manifest $R2/logs/storage/recompress_$(date +%Y%m%d_%H%M)_${SLURM_JOB_ID:-local}.jsonl --dirs "${dirs[@]}"
      local rc=$?
      df -h /work/u2507115 /home/u2507115 | tail -2
      echo "md5 of the final file: $(md5sum $R2/upload/pid_v2bgm_d2_a2k_p3.csv | cut -c1-32)"
      return $rc ;;
    *) echo "usage: pid_npz_recompress.sh sample | run work|home|all"; return 2 ;;
  esac
}
main "$@"; exit $?
