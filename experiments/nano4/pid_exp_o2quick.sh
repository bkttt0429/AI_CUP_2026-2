#!/usr/bin/env bash
# Quick reading of "39,683 -> 107,458 external accounts" on the official candidates with the two o2 networks that are
# finished (o2s0, o2s2): vectors before the pool fine-tune, leave-games-out check against o1 (METHOD_NOTES 2.65).
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=$R2/data/seq_ft
for k in 0 2; do for pool in pub2 priv2; do
  python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool $pool --net -1 --steps 0 --horizon 1000 --ckpt-dir /work/u2507115/pid_v2/seqnet --ckpt-suffix _o2s${k}e --tag _o2s${k}e --out $S
done; done
python -I $R2/scripts/pid_v2_offcand2.py --groups "o1=$S/official_{pool}_net-1_o1s{0|1|2}e_t0.npz" "o2=$S/official_{pool}_net-1_o2s{0|2}e_t0.npz" "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz" \
  --comps "o1x2=o1" "o2x2=o2" "both=o1+o2" --ref o1x2 --out $R2/results/final/v2_offcand2_o2quick.json
