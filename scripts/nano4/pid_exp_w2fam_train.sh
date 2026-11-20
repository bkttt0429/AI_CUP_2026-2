#!/usr/bin/env bash
# METHOD_NOTES 2.193: a full family with the SECOND WINDOW (scripts/pid_win2.py, --shape-prev 1: the 9x9 neighbourhood of
# the previous ply's point, read by its own encoder) - the arm PRV of the short runs of 2.192, which met the rule registered
# there (four-network single mean +0.0038 against the plain window, both jobs' two-network differences positive, one
# interval above zero).
#   o40 = o38 (second feature set, 30,000 steps, horizon 60,000, the accounts of scale3b AND scale4, 9x9 window; the same
#         seed and sampling for every member) + --shape-prev 1
# The node's eight GPUs are split evenly over the members given (1 member = eight GPUs, 2 = four each, 4 = two each): the
# big job ends at 00:21 and one network on two GPUs would need about 4.5 h.
#   pid_exp_w2fam_train.sh <o40> "<member> [<member> ...]"        members 0-7
# EMA only, no state files, no snapshots. Fold 0 is never read. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj
  local fam=$1 members=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b PLY4=$X/scale4/ply SUP4=$X/scale4/supj
  local d i k n per slot set free cards lines=() C tag extra seed mem rc spec
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H $PLY4 --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv $X/scale4/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $SUP_H $SUP4 --sup-stats $P/stats_js.npz"
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H $PLY4; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  [ -e "$X/scale3b/CHECK_PASSED" ] && [ -e "$X/scale4/CHECK_PASSED" ] || { echo "the scale3b / scale4 stores have not passed their check"; return 1; }
  mem=$(free -g | awk 'NR==2{print $4}')
  [ "$mem" -ge "${MIN_FREE_GB:-200}" ] || { echo "node $(hostname) has only $mem GB of free memory (page cache full): not started"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_win2.py $R2/scripts/pid_seq_train_w2.py || { echo "a script does not compile"; return 1; }
  # shellcheck disable=SC2086
  set -- $members
  n=$#
  case "$n" in 1|2|4) ;; *) echo "1, 2 or 4 members per node"; return 2 ;; esac
  per=$((8 / n))
  slot=0
  for i in "$@"; do
    case "$fam" in
      o40) C="--steps 30000 --horizon 60000 $JS"
           if [ "$i" -lt 3 ]; then tag=o40As$i; extra="--own-by-games linear"; seed=$((140 + i))
           elif [ "$i" -lt 6 ]; then tag=o40As$i; extra="--own-by-games sqrt"; seed=$((140 + i))
           elif [ "$i" -eq 6 ]; then tag=o40Cs0; extra="--sup-only --own-by-games linear"; seed=150
           elif [ "$i" -eq 7 ]; then tag=o40Cs1; extra="--sup-only --own-by-games sqrt"; seed=151
           else echo "member $i: 0-7"; return 2; fi
           extra="$extra --shape-prev 1" ;;
      *) echo "unknown family $fam"; return 2 ;;
    esac
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }
    set=""
    for k in $(seq 0 $((per - 1))); do set="$set${set:+,}${cards[$((per * slot + k))]}"; done
    CUDA_VISIBLE_DEVICES="$set" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("cards ok:", torch.cuda.device_count())' \
      || { echo "GPU set $set on $(hostname) failed the multi-card check"; return 1; }
    lines+=("$set logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --shape-win $WIN --seed $seed")
    slot=$((slot + 1))
  done
  spec=$(mktemp "${TMPDIR:-/tmp}/w2fam_spec_XXXXXX")
  printf '%s\n' "${lines[@]}" > "$spec"
  cat "$spec"
  echo "node $(hostname): family $fam members $members, $n model(s) x $per GPUs; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}'); /work free $free MB; $(date '+%F %T')"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w2.py --fork "$spec" $OFF $SUP
  rc=$?
  rm -f "$spec"
  grep -hE "training on|supervised per-ply|training forward" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260
  echo "checkpoints of $fam so far: $(ls /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"); $(date '+%F %T')"
  return $rc
}
main "$@"; exit $?
