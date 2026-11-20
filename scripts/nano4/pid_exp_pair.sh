#!/usr/bin/env bash
# METHOD_NOTES 2.144: the PAIR network - a network built for the near misses (1-game questions whose owner sits at rank
# 2-5 behind look-alikes; every group of sequence networks makes the same errors). It reads the question game and one
# reference game of a candidate together and is trained against look-alike players (scripts/pid_pair_train.py); at test
# time it re-orders the first 10 candidates of the fused score (scripts/pid_pair_eval.py). The rule is in the notes.
#   0  start-up check: a 30-step network on a few parts of the games (the job stops here when it fails)
#   1  in the background (CPU): the tasks of the final's composition v2bgm_d2 on the official pseudo-pools, seeds 0-3
#   2  eight networks, one per GPU, 16,000 steps, start = the by-games plain-input networks o22Ss0-2:
#        pa pb pc  the design (48 questions x 8 candidates x 4 reference games, 128 plies each, look-alikes from the 64
#                  nearest players, half of the questions external accounts) from three starts / seeds
#        pd        competition players only            pe  whole games (256 plies), 24 questions
#        pf        random wrong candidates (control)   pg  look-alikes from the 16 nearest
#        ph        96 questions x 2 reference games
#   3  the registered test: pair scores without and with a fine-tune on the pool's reference sets (50 / 150 steps),
#      setting chosen on seeds 0, 1, verdict on seeds 2, 3
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox W=/work/u2507115/pid_v2 J=${SLURM_JOB_ID:-local} g i s tpid="" GPUS=() ALL runs=() nets=()
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local TASKS=$W/pair_eval/tasks_v2bgm_d2_k10.pkl
  python -I -m py_compile $R2/scripts/pid_pair_train.py $R2/scripts/pid_pair_eval.py || { echo "a script does not compile"; return 1; }
  df -h /work/u2507115 /home/u2507115 | tail -2
  IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
  for g in "${ALL[@]}"; do
    if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
  done
  [ "${#GPUS[@]}" -ge 4 ] || { echo "only ${#GPUS[@]} usable GPUs"; return 1; }
  echo "node $(hostname): GPUs ${GPUS[*]}; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}')"
  mkdir -p $W/pair_eval
  if [ ! -s "$TASKS" ]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_eval.py --build --doc v2bgm_d2 --tasks "$TASKS" > logs/nano4/pair-$J-tasks.log 2>&1 &
    tpid=$!
  fi
  # 0. start-up check of the trainer (a few parts of the games, 30 steps)
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_pair_train.py --smoke $OFF --run "gpu=${GPUS[0]} tag=smoke steps=30 warm=5 anchors=8 k=4 refs=2 val=10 nn_top=8 min_games=3" > logs/nano4/pair-$J-smoke.log 2>&1
  if [ $? -ne 0 ] || [ ! -s $W/seqnet/pair_smoke.pt ]; then
    echo "start-up check FAILED:"; tail -5 logs/nano4/pair-$J-smoke.log; tail -30 logs/nano4/pair_smoke-$J.log 2>/dev/null
    [ -n "$tpid" ] && wait "$tpid"
    return 1
  fi
  echo "start-up check of the trainer passed: $(grep -E 'step 30/30|kept-out players at step' logs/nano4/pair_smoke-$J.log | tail -2 | tr '\n' ' ')"
  # the scorer's start-up check runs beside the training as soon as the tasks exist (its log: pair-<job>-evalsmoke.log)
  (
    while [ ! -s "$TASKS" ] && { [ -z "$tpid" ] || kill -0 "$tpid" 2> /dev/null; }; do sleep 20; done
    [ -s "$TASKS" ] && python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets $W/seqnet/pair_smoke.pt --gpus "${GPUS[0]}" --ft 0 3 --limit 40 --redo --eval-dir $W/pair_eval/smoke --out $W/pair_eval/smoke/eval.json
    echo "scorer start-up check: exit $?"
  ) > logs/nano4/pair-$J-evalsmoke.log 2>&1 &
  local spid=$!
  # 2. the networks
  local specs=("tag=pa seed=1 init=ckpt_-1_o22Ss0e.pt" "tag=pb seed=2 init=ckpt_-1_o22Ss1e.pt" "tag=pc seed=3 init=ckpt_-1_o22Ss2e.pt" "tag=pd seed=4 own_share=1.0"
               "tag=pe seed=5 tq=256 anchors=24" "tag=pf seed=6 nn_top=0" "tag=pg seed=7 nn_top=16" "tag=ph seed=8 anchors=96 refs=2")
  i=0
  for s in "${specs[@]}"; do runs+=(--run "gpu=${GPUS[$((i % ${#GPUS[@]}))]} $s"); i=$((i + 1)); done
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_pair_train.py $OFF "${runs[@]}" || echo "a trainer failed (logs/nano4/pair_p?-$J.log)"
  for g in a b c d e f g h; do
    echo "== pair_p$g: $(grep -E 'the teacher ranks' logs/nano4/pair_p$g-$J.log | sed 's/.*the teacher ranks the right candidate first in \([0-9.]*\).*/teacher \1/') | kept-out: $(grep -E 'kept-out players at step' logs/nano4/pair_p$g-$J.log | sed 's/.*at step \([0-9]*\): right candidate first \([0-9.]*\).*/\1:\2/' | tr '\n' ' ') | $(grep -E '  step ' logs/nano4/pair_p$g-$J.log | tail -1 | cut -c1-150)"
    [ -s $W/seqnet/pair_p$g.pt ] && nets+=("$W/seqnet/pair_p$g.pt")
  done
  wait "$spid"; echo "scorer start-up check: $(tail -1 logs/nano4/pair-$J-evalsmoke.log)"
  [ -n "$tpid" ] && { wait "$tpid" || { echo "the tasks were not built:"; tail -20 logs/nano4/pair-$J-tasks.log; return 1; }; }
  grep -E '^  (pub2|priv2) seed' logs/nano4/pair-$J-tasks.log 2>/dev/null
  [ "${#nets[@]}" -ge 1 ] || { echo "no pair network was written"; return 1; }
  # 3. the registered test
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_eval.py --tasks "$TASKS" --nets "${nets[@]}" --gpus "${GPUS[@]}" --ft 0 50 150 \
    --mean abc=pair_pa,pair_pb,pair_pc all=pair_pa,pair_pb,pair_pc,pair_pd,pair_pe,pair_pg,pair_ph --out $R2/results/final/v2_pair_eval_v2bgm_d2.json
}
main "$@"; exit $?
