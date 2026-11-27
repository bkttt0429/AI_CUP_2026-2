#!/usr/bin/env bash
# METHOD_NOTES 2.244: diagnostic - is 50 still the right number of pool fine-tune steps for the CURRENT networks? (50 was set
# on 11-13 / 11-14 with the networks of that week.) One node, everything in /dev/shm:
#   1. every network of the document, both pools, seen hold-out seeds 0-3: ONE fine-tune run saved after 25, 50 and 75 steps
#      (same schedule as production: --horizon 1000, 50 warm-up steps)
#   2. scripts/pid_v2_offsim_par.py --steps 25 50 75: the document's composition at each step count, t25 vs t50 and t75 vs t50
#   pid_exp_ftsteps.sh [<document, default w11c>]
# Result: results/final/v2_offsim_ftsteps_<doc>_s0_1_2_3.json. Nothing here chooses or writes a file.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local} d=${1:-w11c} G W w j k n e=0 x=0 pids=() q=() tasks=() t0=$SECONDS D L a b c f pool s
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  W=${#G[@]}
  [ -s $R2/results/pid_fusion_$d.json ] || { echo "no document $d"; return 1; }
  D=/dev/shm/ftsteps_$J
  mkdir -p $D/ft $D/dump || return 1
  L=$(mktemp "${TMPDIR:-/tmp}/ftsteps_XXXXXX")
  python3 -I - "$R2" "$SN" "$d" > "$L" <<'PY' || { rm -f "$L"; return 1; }
import json, os, re, sys
R, SN, d = sys.argv[1:4]
nets = sorted({re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", os.path.basename(x[0])).groups() for x in json.load(open(f"{R}/results/pid_fusion_{d}.json"))["seq_official"]}, key=lambda x: x[1])
for s in (0, 1, 2, 3):
    for v, tg in nets:
        assert os.path.getsize(f"{SN}/ckpt_{v}{tg}.pt") > 0
        for pool in ("pub2", "priv2"):
            print(pool, s, v, tg)
PY
  mapfile -t tasks < "$L"; rm -f "$L"
  echo "folder $D on $(hostname): ${#tasks[@]} fine-tunes (document $d, seeds 0-3, saved after 25 / 50 / 75 steps) on $W GPUs, start $(date +%T)"
  for w in $(seq 0 $((W - 1))); do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % W )) -eq "$w" ] || continue
        read -r a b c f <<< "${tasks[$j]}"
        [ -s "$D/ft/offsim_${a}_h3s${b}_net${c}${f}_t75.npz" ] && continue
        CUDA_VISIBLE_DEVICES=${G[$w]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$a" --net "$c" --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$f" --tag "$f" \
          --steps 25 50 75 --holdout 3 --holdout-seed "$b" --out $D/ft < /dev/null || echo "FAILED: ${tasks[$j]}"
      done
    ) > logs/nano4/ftsteps-$J-ft-$w.log 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k" || e=1; done
  n=$(cat logs/nano4/ftsteps-$J-ft-*.log | grep -c FAILED)
  echo "fine-tunes done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n; files $(ls $D/ft | wc -l) ($(du -sh $D/ft | cut -f1))"
  [ "$n" = 0 ] || { grep -h -B2 FAILED logs/nano4/ftsteps-$J-ft-*.log | head -12 | cut -c1-240; find $D -delete 2> /dev/null; return 1; }
  # the 50-step vectors of this run against the stored ones (same run settings): a check of repeatability, reported only
  python -I - "$D/ft" /work/u2507115/pid_v2/vtmp8/ft <<'PY' 2>&1 | tail -n 2
import glob, os, sys
import numpy as np
new, old = sys.argv[1:3]
fs = sorted(glob.glob(f"{new}/offsim_pub2_h3s0_net-1_*_t50.npz"))[:6]
dm = []
for f in fs:
    g = os.path.join(old, os.path.basename(f))
    if os.path.exists(g):
        a, b = np.load(f), np.load(g)
        ka, kb = a["key"].astype(str), b["key"].astype(str)
        ib = {k: i for i, k in enumerate(kb)}
        idx = np.array([ib[k] for k in ka])
        dm.append(float(np.abs(a["emb"].astype(np.float32) - b["emb"].astype(np.float32)[idx]).max()))
print("50-step vectors of this run vs the stored ones, largest absolute difference per file:", [round(x, 5) for x in dm])
PY
  for pool in pub2 priv2; do for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${G[$(( x % W ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$d.json \
      --ft-dir $D/ft --steps 25 50 75 --seeds 0 1 2 3 --workers 6 --only $pool:$s --dump $D/dump --out /dev/null > logs/nano4/ftsteps-$J-split-$pool-$s.log 2>&1 &
    q+=($!); x=$((x + 1))
  done; done
  for k in "${q[@]}"; do wait "$k" || e=1; done
  python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$d.json --ft-dir $D/ft --steps 25 50 75 --seeds 0 1 2 3 --from-dumps --dump $D/dump \
    --out $R2/results/final/v2_offsim_ftsteps_${d}_s0_1_2_3.json > logs/nano4/ftsteps-$J-stat.log 2>&1 || e=1
  grep -E '^(pub2|priv2|both): document weights|^   t(25|75) vs t50' logs/nano4/ftsteps-$J-stat.log | cut -c1-260
  find $D -delete 2> /dev/null
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min), rc $e"
  return $e
}
main "$@"; exit $?
