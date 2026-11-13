#!/usr/bin/env bash
# Eight official networks on one node (net -1: competition folds 1-4, never fold 0; all external accounts incl. the high-dan
# batch; half of every episode external; EMA): four as before (o3s0-3) and four with rank-matched episodes (METHOD_NOTES 2.55):
# o3hms0/1 = half of the episodes from one rank class, o3hbs0/1 = every episode from a class and its two neighbours.
# Every network reads its own copy of the games (8 x 143 GB): the shared-memory loader failed on 2026-11-14 because the
# node's /dev/shm was full of another user's files. The node must have the commit room for eight copies; checked first.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
free_kb=$(awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{print l-c}' /proc/meminfo)
echo "node $(hostname): commit room $((free_kb / 1048576)) GB; /dev/shm $(df -h /dev/shm | awk 'NR==2{print $3" used of "$2}')"
if [ "$free_kb" -lt $((1250 * 1048576)) ]; then echo "not enough commit room for eight copies of the games on this node (need 1250 GB): resubmit excluding $(hostname)" >&2; exit 3; fi
C="--steps 20000 --horizon 60000 --ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv --ext-share 0.5 --no-state"
bash $R2/scripts/nano4/pid_seq_train_multi.sh \
  "-1 o3s0 $C --seed 0" "-1 o3s1 $C --seed 1" "-1 o3s2 $C --seed 2" "-1 o3s3 $C --seed 3" \
  "-1 o3hms0 $C --seed 0 --rank-ep class --rank-ep-share 0.5" "-1 o3hms1 $C --seed 1 --rank-ep class --rank-ep-share 0.5" \
  "-1 o3hbs0 $C --seed 0 --rank-ep band" "-1 o3hbs1 $C --seed 1 --rank-ep band"
