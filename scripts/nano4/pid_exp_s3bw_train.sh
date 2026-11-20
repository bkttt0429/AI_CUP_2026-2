#!/usr/bin/env bash
# METHOD_NOTES 2.177 (merged retrain): the four by-games families of the final's composition trained with BOTH small
# changes that helped a family on its own - more external accounts (scale3b, as o27-o30) AND the window input of o31
# (the stones in a WIN x WIN window around every ply's point; scripts/pid_win.py, trainer scripts/pid_seq_train_w.py).
# Recipes, seeds, steps and sampling as o27-o30 (= the originals), so every network has a twin without the window.
#   o32 = o27 + window (plain inputs, 40,000 steps)            o33 = o28 + window (second feature set, 40,000 steps)
#   o34 = o29 + window (second feature set, 30,000 steps)      o35 = o30 + window (second feature set, 512x10, 30,000 steps)
# One job = four models of a family, two GPUs each.
# The first two members of every family also write snapshots (ckpt_-1_<tag>k<step/1000>e.pt) for the trajectory over the
# training steps; they are read with pid_exp_snapread.sh and removed.
#   [WIN=9] pid_exp_s3bw_train.sh <o32|o33|o34|o35> <group 0|1>       group 0 = members 0-3, group 1 = members 4-7
# EMA only, no state files. Fold 0 is never read. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj fam=$1 grp=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b d i k slot pair free cards lines=() C tag extra seed
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $SUP_H --sup-stats $P/stats_js.npz"
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check (ext_fox_scale3b.sh check)"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_seq_train_w.py || { echo "a script does not compile"; return 1; }
  case "$WIN" in 5|7|9|11|13) ;; *) echo "WIN=$WIN is not an odd size between 5 and 13"; return 2 ;; esac
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  for slot in 0 1 2 3; do
    pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}
    CUDA_VISIBLE_DEVICES="$pair" timeout 600 python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize()' \
      || { echo "GPU pair $pair on $(hostname) failed the two-card check"; return 1; }
  done
  slot=0
  for i in $((4 * grp)) $((4 * grp + 1)) $((4 * grp + 2)) $((4 * grp + 3)); do
    case "$fam" in
      o32) C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
           if [ "$i" -lt 4 ]; then tag=o32Ss$i; extra=""; seed=$((180 + i)); else tag=o32SMs$((i - 4)); extra="--drop-open 0.3"; seed=$((180 + i)); fi ;;
      o33) C="--steps 40000 --horizon 60000 $JS --own-by-games linear"
           if [ "$i" -lt 6 ]; then tag=o33As$i; extra=""; seed=$((200 + i)); else tag=o33Cs$((i - 6)); extra="--sup-only"; seed=$((210 + i - 6)); fi ;;
      o34) C="--steps 30000 --horizon 60000 $JS"
           if [ "$i" -lt 3 ]; then tag=o34As$i; extra="--own-by-games linear"; seed=$((140 + i))
           elif [ "$i" -lt 6 ]; then tag=o34As$i; extra="--own-by-games sqrt"; seed=$((140 + i))
           elif [ "$i" -eq 6 ]; then tag=o34Cs0; extra="--sup-only --own-by-games linear"; seed=150
           else tag=o34Cs1; extra="--sup-only --own-by-games sqrt"; seed=151; fi ;;
      o35) C="--steps 30000 --horizon 60000 $JS --arch 512,10,8,2048 --own-by-games linear"
           if [ "$i" -lt 6 ]; then tag=o35As$i; extra=""; seed=$((160 + i)); else tag=o35Cs$((i - 6)); extra="--sup-only"; seed=$((170 + i - 6)); fi ;;
      *) echo "unknown family $fam"; return 2 ;;
    esac
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }
    if [ "$i" -lt 2 ]; then
      case "$fam" in o32) extra="$extra --snap 10000 20000 30000" ;; o33) extra="$extra --snap 30000 35000" ;;
                     o34) extra="$extra --snap 10000 15000 20000 25000" ;; o35) extra="$extra --snap 5000 10000 15000 20000" ;; esac
    fi
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
  [ "$fam" = o32 ] || data="$OFF $SUP"
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
