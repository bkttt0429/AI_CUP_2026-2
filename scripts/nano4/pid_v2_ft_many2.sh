#!/usr/bin/env bash
# Pool fine-tunes of several of my own sequence networks in one job (METHOD_NOTES 2.32, 2.42), skipping what is already there.
#   pid_v2_ft_many.sh real   "<net> <suffix>" ...   the real added pools: vectors after 0 / 50 / 100 steps -> data/seq_ft
#   pid_v2_ft_many.sh offsim "<net> <suffix>" ...   official pseudo-pools (3 games per candidate held out, seeds 0 and 1):
#                                                   vectors after 0 / 50 steps -> /work/u2507115/pid_v2/seq_ft_offsim
# Checkpoints are /work/u2507115/pid_v2/seqnet/ckpt_<net><suffix>.pt. Run through scripts/nano4/run.sbatch with one GPU.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
mode=$1
shift
for spec in "$@"; do
  read -r v sfx <<< "$spec"
  for pool in pub2 priv2; do
    common=(--mode official --pool "$pool" --net "$v" --horizon 1000 --ckpt-dir /work/u2507115/pid_v2/seqnet --ckpt-suffix "$sfx" --tag "$sfx")
    if [ "$mode" = real ]; then
      out=$R2/data/seq_ft
      [ -s "$out/official_${pool}_net${v}${sfx}_t50.npz" ] || python -I $R2/scripts/pid_v2_seq_finetune.py "${common[@]}" --steps 0 50 --out "$out"
    else
      out=/work/u2507115/pid_v2/seq_ft_offsim
      for s in 0 1; do
        [ -s "$out/offsim_${pool}_h3s${s}_net${v}${sfx}_t50.npz" ] || python -I $R2/scripts/pid_v2_seq_finetune.py "${common[@]}" --steps 0 50 --holdout 3 --holdout-seed "$s" --out "$out"
      done
    fi
  done
done
