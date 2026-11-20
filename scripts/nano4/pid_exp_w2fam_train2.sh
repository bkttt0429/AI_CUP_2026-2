#!/usr/bin/env bash
# METHOD_NOTES 2.197: the other three recipes with the SECOND WINDOW (scripts/pid_win2.py, --shape-prev 1), trained only when
# o40 has met the rule of notes 2.193. Recipes, seeds, steps and sampling exactly as the merged retrain o32 / o33 / o35
# (scripts/nano4/pid_exp_s3bw_train.sh: the accounts of scale3b, 9x9 window; NO scale4 accounts - notes 2.194: they add
# nothing at the family level), so every network has a twin without the second window:
#   o41 = o32 + second window (plain inputs, 40,000 steps)
#   o42 = o33 + second window (second feature set, 40,000 steps)
#   o43 = o35 + second window (second feature set, 512x10, 30,000 steps)
# (o40 = o38 + second window is pid_exp_w2fam_train.sh.) The node's eight GPUs are split evenly over the members given
# (1 member = eight GPUs, 2 = four each, 4 = two each).
#   pid_exp_w2fam_train2.sh <o41|o42|o43> "<member> [<member> ...]"        members 0-7
# EMA only, no state files, no snapshots. Fold 0 is never read. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj
  local fam=$1 members=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b
  local d i k n per slot set free cards lines=() C tag extra seed mem rc spec data
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $SUP_H --sup-stats $P/stats_js.npz"
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check"; return 1; }
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
    case "$i" in 0|1|2|3|4|5|6|7) ;; *) echo "member $i: 0-7"; return 2 ;; esac
    case "$fam" in
      o41) C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
           if [ "$i" -lt 4 ]; then tag=o41Ss$i; extra=""; seed=$((180 + i)); else tag=o41SMs$((i - 4)); extra="--drop-open 0.3"; seed=$((180 + i)); fi ;;
      o42) C="--steps 40000 --horizon 60000 $JS --own-by-games linear"
           if [ "$i" -lt 6 ]; then tag=o42As$i; extra=""; seed=$((200 + i)); else tag=o42Cs$((i - 6)); extra="--sup-only"; seed=$((210 + i - 6)); fi ;;
      o43) C="--steps 30000 --horizon 60000 $JS --arch 512,10,8,2048 --own-by-games linear"
           if [ "$i" -lt 6 ]; then tag=o43As$i; extra=""; seed=$((160 + i)); else tag=o43Cs$((i - 6)); extra="--sup-only"; seed=$((170 + i - 6)); fi ;;
      *) echo "unknown family $fam"; return 2 ;;
    esac
    extra="$extra --shape-prev 1"
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
  data="$OFF"
  [ "$fam" = o41 ] || data="$OFF $SUP"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w2.py --fork "$spec" $data
  rc=$?
  rm -f "$spec"
  grep -hE "training on|supervised per-ply|training forward" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260
  echo "checkpoints of $fam so far: $(ls /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"); $(date '+%F %T')"
  return $rc
}
main "$@"; exit $?
