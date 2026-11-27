#!/usr/bin/env bash
# METHOD_NOTES 2.84: does training longer help? Snapshots of the 40,000-step official run (o3SL, seeds 12-19) as families of
# their own (scripts/nano4/pid_snap_links.sh): o3SK = 20,000 steps, o3SM = 30,000, o3SL = 40,000 (the finished run).
# For every family given: the real-pool vectors of its networks (0 and 50 fine-tune steps, skipped when present), then the
# leave-games-out check on the official candidates: single networks by family, next to o3S and o4S.
#   pid_exp_snapcmp.sh <family> [<family> ...]      (2-4 GPUs, from the AI_CUP_2026-1 checkout)
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local S=$R2/data/seq_ft fam k i j nets=() pids=() groups comps=() l
  IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
  for fam in "$@"; do
    for k in $(seq 0 23); do [ -e $SN/ckpt_-1_${fam}s${k}e.pt ] && nets+=("-1 _${fam}s${k}e"); done
  done
  echo "${#nets[@]} networks of $*; ${#G[@]} GPU(s)"
  for i in "${!G[@]}"; do
    (
      for j in "${!nets[@]}"; do
        [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
        CUDA_VISIBLE_DEVICES=${G[$i]} bash $R2/scripts/nano4/pid_v2_ft_many2.sh real "${nets[$j]}"
      done
    ) > logs/nano4/snapcmp-$J-$i.log 2>&1 &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || echo "a worker failed (logs/nano4/snapcmp-$J-*.log)"; done
  groups=("m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz" "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz"
          "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pb=$S/official_{pool}_net{1|2}_ybe_t0.npz")
  for fam in o1 o3S o4S "$@"; do
    l=""
    for k in $(seq 0 23); do [ -s $S/official_pub2_net-1_${fam}s${k}e_t0.npz ] && l="$l|$k"; done
    l=${l#|}
    [ -n "$l" ] || continue
    if [[ "$l" == *"|"* ]]; then groups+=("$fam=$S/official_{pool}_net-1_${fam}s{$l}e_t0.npz"); else groups+=("$fam=$S/official_{pool}_net-1_${fam}s${l}e_t0.npz"); fi
  done
  for fam in "$@"; do comps+=("$fam=$fam"); done
  python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=o3S" "${comps[@]}" --ref ref --out $R2/results/final/v2_offcand2_snap_$(echo "$*" | tr ' ' '_').json
}
main "$@"; exit $?
