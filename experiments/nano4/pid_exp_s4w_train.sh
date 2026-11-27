#!/usr/bin/env bash
# METHOD_NOTES 2.186 / 2.187: the scale4 round - the four by-games families trained once more with the accounts of
# scale3b AND scale4 and the window input. Recipes, seeds, steps and sampling as o32-o35 (= o27-o30 = the originals):
#   o36 = o32 (plain inputs, 40,000 steps)            o37 = o33 (second feature set, 40,000 steps)
#   o38 = o34 (second feature set, 30,000 steps)      o39 = o35 (second feature set, 512x10) at 20,000 steps instead of
#   30,000 (METHOD_NOTES 2.189, registered before any reading of o36-o39; same schedule horizon, so it is o35 stopped earlier)
# One job = four models of a family, two GPUs each. A node that fails the two-card check within DP_TIMEOUT seconds or has
# less than MIN_FREE_GB of free memory ends the job at once (the orchestrator sends it to another node).
#   [WIN=9] pid_exp_s4w_train.sh <o36|o37|o38|o39> <group 0|1>       group 0 = members 0-3, group 1 = members 4-7
# EMA only, no state files, no snapshots. Fold 0 is never read. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj fam=$1 grp=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b PLY4=$X/scale4/ply SUP4=$X/scale4/supj d i k slot pair free cards lines=() C tag extra seed mem
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H $PLY4 --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv $X/scale4/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $SUP_H $SUP4 --sup-stats $P/stats_js.npz"
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H $PLY4; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check (ext_fox_scale3b.sh check)"; return 1; }
  [ -e "$X/scale4/CHECK_PASSED" ] || { echo "the scale4 stores have not passed their check (ext_fox_scale4.sh check)"; return 1; }
  mem=$(free -g | awk 'NR==2{print $4}')  # a node whose memory is mostly page cache loads for an hour (node 082, 11-17): fail fast, the job is sent elsewhere
  [ "$mem" -ge "${MIN_FREE_GB:-200}" ] || { echo "node $(hostname) has only $mem GB of free memory (page cache full): not started"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_seq_train_w.py || { echo "a script does not compile"; return 1; }
  case "$WIN" in 5|7|9|11|13) ;; *) echo "WIN=$WIN is not an odd size between 5 and 13"; return 2 ;; esac
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  for slot in 0 1 2 3; do
    pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}
    CUDA_VISIBLE_DEVICES="$pair" timeout ${DP_TIMEOUT:-150} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize()' \
      || { echo "GPU pair $pair on $(hostname) failed the two-card check"; return 1; }
  done
  slot=0
  for i in $((4 * grp)) $((4 * grp + 1)) $((4 * grp + 2)) $((4 * grp + 3)); do
    case "$fam" in
      o36) C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
           if [ "$i" -lt 4 ]; then tag=o36Ss$i; extra=""; seed=$((180 + i)); else tag=o36SMs$((i - 4)); extra="--drop-open 0.3"; seed=$((180 + i)); fi ;;
      o37) C="--steps 40000 --horizon 60000 $JS --own-by-games linear"
           if [ "$i" -lt 6 ]; then tag=o37As$i; extra=""; seed=$((200 + i)); else tag=o37Cs$((i - 6)); extra="--sup-only"; seed=$((210 + i - 6)); fi ;;
      o38) C="--steps 30000 --horizon 60000 $JS"
           if [ "$i" -lt 3 ]; then tag=o38As$i; extra="--own-by-games linear"; seed=$((140 + i))
           elif [ "$i" -lt 6 ]; then tag=o38As$i; extra="--own-by-games sqrt"; seed=$((140 + i))
           elif [ "$i" -eq 6 ]; then tag=o38Cs0; extra="--sup-only --own-by-games linear"; seed=150
           else tag=o38Cs1; extra="--sup-only --own-by-games sqrt"; seed=151; fi ;;
      o39) C="--steps 20000 --horizon 60000 $JS --arch 512,10,8,2048 --own-by-games linear"  # 20,000 not 30,000: METHOD_NOTES 2.189 (rule of 2.177: o35 read 0.5824 at 20,000 steps and 0.5693 at 30,000)
           if [ "$i" -lt 6 ]; then tag=o39As$i; extra=""; seed=$((160 + i)); else tag=o39Cs$((i - 6)); extra="--sup-only"; seed=$((170 + i - 6)); fi ;;
      *) echo "unknown family $fam"; return 2 ;;
    esac
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }
    pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}
    lines+=("$pair logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --shape-win $WIN --seed $seed")
    slot=$((slot + 1))
  done
  local spec
  spec=$(mktemp "${TMPDIR:-/tmp}/s3b_spec_XXXXXX")
  printf '%s\n' "${lines[@]}" > "$spec"
  cat "$spec"
  echo "node $(hostname): family $fam group $grp, four models x two GPUs; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}'); /work free $free MB"
  local data="$OFF"
  [ "$fam" = o36 ] || data="$OFF $SUP"
  # shellcheck disable=SC2086
  # the dual-GPU trainer with the window input (scripts/pid_seq_train_w.py = pid_seq_train_dp.py + --shape-win)
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w.py --fork "$spec" $data
  local rc=$?
  rm -f "$spec"
  grep -hE "training on|supervised per-ply" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260
  ls -la /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"
  return $rc
}
main "$@"; exit $?
