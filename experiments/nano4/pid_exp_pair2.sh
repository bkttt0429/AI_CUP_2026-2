#!/usr/bin/env bash
# METHOD_NOTES 2.146: the pair network, second batch - look-alike wrong candidates for the EXTERNAL accounts too.
# Reading at step 2000 of the first batch (job 783537): on players kept out of the pair training the network ranks the right
# candidate first in 0.42-0.51 of 8, on the players it trains on in 0.65: it does not carry over well, and its look-alike
# episodes come from 5,893 competition players only. Here the 116,420 external accounts get look-alikes as well (ext_nn=64:
# drawn from an account's 64 nearest accounts by the teacher's prototypes, the nearest 2 left out - two accounts may be one
# person): twenty times as many identities in the hard episodes.
#   qa qb qc  the first batch's design + external look-alikes, three starts / seeds
#   qd        three quarters of the questions external        qe  nothing left out (ext_skip=0)
#   qf        16 candidates x 2 reference games                qg  ply masking 0.5
#   qh        start = an opening-hidden network (o22SMs0)
# Same registered test as the first batch (notes 2.144; the tasks file exists), then one analysis over both batches.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox W=/work/u2507115/pid_v2 J=${SLURM_JOB_ID:-local} g i s GPUS=() ALL runs=() nets=() both=()
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local TASKS=$W/pair_eval/tasks_v2bgm_d2_k10.pkl
  python -I -m py_compile $R2/scripts/pid_pair_train.py $R2/scripts/pid_pair_eval.py || { echo "a script does not compile"; return 1; }
  [ -s "$TASKS" ] || { echo "no tasks file $TASKS"; return 1; }
  df -h /work/u2507115 /home/u2507115 | tail -2
  IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
  for g in "${ALL[@]}"; do
    if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
  done
  [ "${#GPUS[@]}" -ge 4 ] || { echo "only ${#GPUS[@]} usable GPUs"; return 1; }
  echo "node $(hostname): GPUs ${GPUS[*]}; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}')"
  # start-up checks of the two changed scripts: a 30-step network on a few parts of the games, then the scorer on 40 questions
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_pair_train.py --smoke $OFF --run "gpu=${GPUS[0]} tag=smoke2 steps=30 warm=5 anchors=8 k=4 refs=2 val=10 nn_top=8 min_games=3 ext_nn=16" > logs/nano4/pair-$J-smoke.log 2>&1
  if [ $? -ne 0 ] || [ ! -s $W/seqnet/pair_smoke2.pt ]; then
    echo "start-up check of the trainer FAILED:"; tail -5 logs/nano4/pair-$J-smoke.log; tail -30 logs/nano4/pair_smoke2-$J.log 2>/dev/null
    return 1
  fi
  echo "start-up check of the trainer passed: $(grep -E 'kept-out players at step' logs/nano4/pair_smoke2-$J.log | tail -1)"
  python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets $W/seqnet/pair_smoke2.pt --gpus "${GPUS[0]}" --ft 0 3 --limit 40 --redo --eval-dir $W/pair_eval/smoke2 --out $W/pair_eval/smoke2/eval.json > logs/nano4/pair-$J-evalsmoke.log 2>&1 \
    && python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets $W/seqnet/pair_smoke2.pt --ft 0 3 --analyse-only --eval-dir $W/pair_eval/smoke2 --out $W/pair_eval/smoke2/eval2.json >> logs/nano4/pair-$J-evalsmoke.log 2>&1
  if [ $? -ne 0 ]; then echo "start-up check of the scorer FAILED:"; tail -30 logs/nano4/pair-$J-evalsmoke.log; return 1; fi
  echo "start-up check of the scorer passed: $(grep -c 'wrote ' logs/nano4/pair-$J-evalsmoke.log) result files"
  local specs=("tag=qa seed=11 ext_nn=64 init=ckpt_-1_o22Ss0e.pt" "tag=qb seed=12 ext_nn=64 init=ckpt_-1_o22Ss1e.pt" "tag=qc seed=13 ext_nn=64 init=ckpt_-1_o22Ss2e.pt" "tag=qd seed=14 ext_nn=64 own_share=0.25"
               "tag=qe seed=15 ext_nn=64 ext_skip=0" "tag=qf seed=16 ext_nn=64 k=16 refs=2" "tag=qg seed=17 ext_nn=64 mask=0.5" "tag=qh seed=18 ext_nn=64 init=ckpt_-1_o22SMs0e.pt")
  i=0
  for s in "${specs[@]}"; do runs+=(--run "gpu=${GPUS[$((i % ${#GPUS[@]}))]} $s"); i=$((i + 1)); done
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_pair_train.py $OFF "${runs[@]}" || echo "a trainer failed (logs/nano4/pair_q?-$J.log)"
  for g in a b c d e f g h; do
    echo "== pair_q$g: $(grep -E 'the teacher ranks' logs/nano4/pair_q$g-$J.log | sed 's/.*the teacher ranks the right candidate first in \([0-9.]*\).*/teacher \1/') | kept-out: $(grep -E 'kept-out players at step' logs/nano4/pair_q$g-$J.log | sed 's/.*at step \([0-9]*\): right candidate first \([0-9.]*\).*/\1:\2/' | tr '\n' ' ') | $(grep -E '  step ' logs/nano4/pair_q$g-$J.log | tail -1 | cut -c1-150)"
    [ -s $W/seqnet/pair_q$g.pt ] && nets+=("$W/seqnet/pair_q$g.pt")
  done
  grep -h "external look-alikes" logs/nano4/pair_qa-$J.log logs/nano4/pair_qe-$J.log 2>/dev/null | cut -c1-300
  [ "${#nets[@]}" -ge 1 ] || { echo "no pair network was written"; return 1; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets "${nets[@]}" --gpus "${GPUS[@]}" --ft 0 50 150 \
    --mean q3=pair_qa,pair_qb,pair_qc q8=pair_qa,pair_qb,pair_qc,pair_qd,pair_qe,pair_qf,pair_qg,pair_qh --out $R2/results/final/v2_pair_eval2_v2bgm_d2.json
  # both batches together (the first batch's scores are read as they are; a network whose scores are not complete is left out)
  for g in a b c d e f g h; do [ -s $W/seqnet/pair_p$g.pt ] && both+=("$W/seqnet/pair_p$g.pt"); done
  echo; echo "===== both batches"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets "${both[@]}" "${nets[@]}" --ft 0 50 150 --analyse-only \
    --mean p3=pair_pa,pair_pb,pair_pc q3=pair_qa,pair_qb,pair_qc pq6=pair_pa,pair_pb,pair_pc,pair_qa,pair_qb,pair_qc \
           pq14=pair_pa,pair_pb,pair_pc,pair_pd,pair_pe,pair_pg,pair_ph,pair_qa,pair_qb,pair_qc,pair_qd,pair_qe,pair_qg,pair_qh --out $R2/results/final/v2_pair_eval12_v2bgm_d2.json
}
main "$@"; exit $?
