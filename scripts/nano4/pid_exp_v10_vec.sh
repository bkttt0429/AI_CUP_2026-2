#!/usr/bin/env bash
# METHOD_NOTES 2.229 / 2.230: the pseudo-pool vectors (50-step pool fine-tune, 3 games per candidate held out) of the
# networks of documents and / or families for given hold-out seeds, all into ONE folder. With shm:<name> the folder is
# node-local (/dev/shm/f0c_<name>/ft: nothing is written to /work or /home) and is KEPT for the comparison step that runs
# later on the SAME node (pid_exp_v10_cmp.sh). Missing files only; one fine-tune per GPU (a fine-tune holds up to ~100 GB).
#   pid_exp_v10_vec.sh <out folder | shm:<name>> "<seeds>" <document | family> ...
#     <document>   a name with results/pid_fusion_<name>.json: every network of its seq_official
#     <family>     e.g. o64F: its linked checkpoints (scripts/nano4/pid_fam_links.sh makes the links)
# The items are done in the order given (the first document's networks first). From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local} out=$1 seeds=$2 G W w j n f k a b c d rc=0 pids=() tasks=() t0=$SECONDS L
  shift 2
  case "$out" in shm:*) out=/dev/shm/f0c_${out#shm:}/ft ;; esac
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  mkdir -p "$out" || return 1
  for f in "$@"; do  # families: member links
    [ -s $R2/results/pid_fusion_$f.json ] && continue
    n=$(bash $R2/scripts/nano4/pid_fam_links.sh "$f" | awk '{print $2}')
    echo "family $f: ${n:-0} linked checkpoints"
    [ "${n:-0}" -ge 7 ] || { echo "family $f has only ${n:-0} networks: not started"; return 1; }
  done
  L=$(mktemp "${TMPDIR:-/tmp}/v10vec_XXXXXX")
  python3 -I - "$R2" "$SN" "$out" "$seeds" "$@" > "$L" <<'PY' || { rm -f "$L"; return 1; }
import json, os, re, sys
R, SN, out, seeds, items = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split()], sys.argv[5:]
seen, order = set(), []
for it in items:
    d = f"{R}/results/pid_fusion_{it}.json"
    if os.path.exists(d):
        nets = sorted({re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", os.path.basename(x[0])).groups() for x in json.load(open(d))["seq_official"]}, key=lambda x: x[1])
    else:
        nets = [("-1", f"_{it}s{k}e") for k in range(24) if os.path.exists(f"{SN}/ckpt_-1_{it}s{k}e.pt")]
        assert nets, f"{it}: neither a document nor a family with linked checkpoints"
    for v, tg in nets:
        if (v, tg) not in seen:
            seen.add((v, tg)); order.append((v, tg))
n = 0
for s in seeds:  # seed by seed, so that an interrupted run leaves complete seeds first
    for v, tg in order:
        assert os.path.getsize(f"{SN}/ckpt_{v}{tg}.pt") > 0, f"no checkpoint ckpt_{v}{tg}.pt"
        for pool in ("pub2", "priv2"):
            if not os.path.exists(f"{out}/offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz"):
                print(pool, s, v, tg); n += 1
print(f"{len(order)} networks x {len(seeds)} seeds x 2 pools: {n} fine-tunes to do", file=sys.stderr)
PY
  mapfile -t tasks < "$L"; rm -f "$L"
  echo "folder $out on $(hostname): ${#tasks[@]} fine-tunes to do (seeds $seeds; $*) on ${#G[@]} GPUs, start $(date +%T)"
  W=${#G[@]}
  for w in $(seq 0 $((W - 1))); do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % W )) -eq "$w" ] || continue
        read -r a b c d <<< "${tasks[$j]}"
        CUDA_VISIBLE_DEVICES=${G[$w]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$a" --net "$c" --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$d" --tag "$d" \
          --steps 50 --holdout 3 --holdout-seed "$b" --out "$out" < /dev/null || echo "FAILED: ${tasks[$j]}"
      done
    ) > logs/nano4/v10vec-$J-$w.log 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k" || rc=1; done
  n=$(cat logs/nano4/v10vec-$J-*.log | grep -c FAILED)
  echo "fine-tunes done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n; files in $out: $(ls "$out" | grep -c '_t50\.npz$') ($(du -sh "$out" | cut -f1))"
  [ "$n" = 0 ] || { grep -h FAILED logs/nano4/v10vec-$J-*.log | head -5; rc=1; }
  return $rc
}
main "$@"; exit $?
