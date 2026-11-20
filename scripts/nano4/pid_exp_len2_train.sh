#!/usr/bin/env bash
# METHOD_NOTES 2.211 (second part): the training-length question on the two other 384x8 recipes, in the idle nodes of job
# 796544 (which ends 00:29, hence the step counts). Everything but the number of steps as the twins (recipe, seeds, sampling,
# fold 0, 9x9 window + second window, scale3b accounts, trainer pid_seq_train_w2.py, the 60,000-step cosine):
#   o57 = o47's recipe, 45,000 steps (o47: 30,000); seeds 330-337
#   o58 = o44's recipe (plain inputs), 55,000 steps (o44: 40,000); seeds 300-307
# (o45's recipe to 60,000 = o53, o46's = o56: scripts/nano4/pid_exp_len_train.sh.)
#   pid_exp_len2_train.sh <o57|o58> "<member> [<member> ...]"        members 0-7
# EMA only, no state files, no snapshots. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj
  local fam=$1 members=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b
  local d i k n per slot set free cards lines=() tags=() C tag extra seed mem rc spec data
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/comp0 $O/ext $SUP_H --sup-stats $P/stats_js.npz"
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check"; return 1; }
  for d in results/trunk_vec results/trunk_vec_x40 results/trunk_vec_x80; do
    [ "$(ls "$d/fold0" 2> /dev/null | grep -c 'games_s.*\.csv$')" -ge 200 ] || { echo "the stored parts of fold 0 in $d are missing"; return 1; }
  done
  case "$fam" in o57|o58) ;; *) echo "unknown family $fam"; return 2 ;; esac
  if [ "$fam" != o58 ] && [ ! -e $O/comp0.CHECK_PASSED ]; then echo "the second feature set of fold 0 has not passed its check: not started"; return 1; fi
  mem=$(free -g | awk 'NR==2{print $4}')
  [ "$mem" -ge "${MIN_FREE_GB:-200}" ] || { echo "node $(hostname) has only $mem GB of free memory (page cache full): not started"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 4500 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_win2.py $R2/scripts/pid_seq_train_w2.py || { echo "a script does not compile"; return 1; }
  # shellcheck disable=SC2086
  set -- $members
  n=$#
  case "$n" in 1|2|4|8) ;; *) echo "1, 2, 4 or 8 members per node"; return 2 ;; esac
  per=$((8 / n))
  slot=0
  for i in "$@"; do
    case "$i" in 0|1|2|3|4|5|6|7) ;; *) echo "member $i: 0-7"; return 2 ;; esac
    case "$fam" in
      o58) C="--steps 55000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
           if [ "$i" -lt 4 ]; then tag=o58Ss$i; extra=""; else tag=o58SMs$((i - 4)); extra="--drop-open 0.3"; fi
           seed=$((300 + i)) ;;
      o57) C="--steps 45000 --horizon 60000 $JS"
           if [ "$i" -lt 3 ]; then tag=o57As$i; extra="--own-by-games linear"
           elif [ "$i" -lt 6 ]; then tag=o57As$i; extra="--own-by-games sqrt"
           elif [ "$i" -eq 6 ]; then tag=o57Cs0; extra="--sup-only --own-by-games linear"
           else tag=o57Cs1; extra="--sup-only --own-by-games sqrt"; fi
           seed=$((330 + i)) ;;
    esac
    extra="$extra --shape-prev 1"
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }
    set=""
    for k in $(seq 0 $((per - 1))); do set="$set${set:+,}${cards[$((per * slot + k))]}"; done
    if [ "$per" -gt 1 ]; then
      CUDA_VISIBLE_DEVICES="$set" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("cards ok:", torch.cuda.device_count())' \
        || { echo "GPU set $set on $(hostname) failed the multi-card check"; return 1; }
    else
      CUDA_VISIBLE_DEVICES="$set" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=m(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("card ok:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory >> 30, "GB")' \
        || { echo "GPU $set on $(hostname) failed the single-card check"; return 1; }
    fi
    lines+=("$set logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --shape-win $WIN --seed $seed")
    tags+=("$tag")
    slot=$((slot + 1))
  done
  spec=$(mktemp "${TMPDIR:-/tmp}/len2_spec_XXXXXX")
  printf '%s\n' "${lines[@]}" > "$spec"
  cat "$spec"
  echo "node $(hostname): family $fam members $members, $n model(s) x $per GPU(s), FOLD 0 TRAINED ON; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}'); /work free $free MB; $(date '+%F %T')"
  data="$OFF"
  [ "$fam" = o58 ] || data="$OFF $SUP"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w2.py --fork "$spec" --all-folds $data
  rc=$?
  rm -f "$spec"
  grep -hE "training on|supervised per-ply|training forward" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260
  for tag in "${tags[@]}"; do  # fold 0 must be among the folds trained on (a missing feature store would drop it silently)
    grep -qE "training on .*folds \[-9, 0, 1, 2, 3, 4\]" logs/nano4/pidseq_-1_${tag}-$J.log || { echo "ALERT: the log of $tag does not show fold 0 among the training folds"; rc=1; }
  done
  echo "checkpoints of $fam so far: $(ls /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"); $(date '+%F %T')"
  return $rc
}
main "$@"; exit $?
