#!/usr/bin/env bash
# One stage of the per-ply extraction of an external selection, on this job's GPUs (METHOD_NOTES 2.70):
#   ext_ply_stage.sh <name> <out dir name> <games FROM> <games TO> <max plies> <stop row or 0> <first shard> <shards in total> [copies, default: every GPU]
# reads /work/u2507115/ext_fox/<name>/select.csv, takes the games FROM .. TO-1 of every account (the file is written account by
# account), the first <stop> of those rows (0: all), and writes /work/u2507115/ext_fox/<name>/<out dir name>/ply_x_*.npz.
# Several jobs can share one stage: give each its own first shard. Resumable per part. From the AI_CUP_2026-1 checkout.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
name=$1; out=$2; g0=$3; g1=$4; plies=$5; stop=$6; first=$7; total=$8
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
copies=${9:-${#G[@]}}
df -h /work/u2507115 | awk 'NR==2{print "work free before:", $4}'
bash $R2/scripts/nano4/shards_offset.sh "$copies" "$first" "$total" python -I $R2/scripts/pid_ext_ply_extract.py --select $X/$name/select.csv --out $X/$name/$out \
  --account-games "$g0" "$g1" --max-plies "$plies" --stop "$stop" --workers 10
echo "parts: $(ls $X/$name/$out | grep -c 'ply_x_.*npz'); size $(du -sh $X/$name/$out | cut -f1)"
df -h /work/u2507115 | awk 'NR==2{print "work free after:", $4}'
