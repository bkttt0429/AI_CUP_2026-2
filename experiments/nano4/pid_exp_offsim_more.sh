#!/usr/bin/env bash
# METHOD_NOTES 2.120: replication of a pseudo-pool comparison on FRESH splits. Every chain's official pseudo-pools use the
# hold-out seeds 0 and 1 (3 games of every official candidate held out). This job fine-tunes every network of two fusion
# documents on the splits of further seeds (default 2 3; 50 steps, the invocation of scripts/nano4/pid_v2_ft_many2.sh) and
# scores the candidate against the reference on the new seeds alone and on all seeds together.
#   pid_exp_offsim_more.sh <reference document> <candidate document> [seed ...]
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 A1=/home/u2507115/BK_AICUP_2026-1 SN=/work/u2507115/pid_v2/seqnet O=/work/u2507115/pid_v2/seq_ft_offsim J=${SLURM_JOB_ID:-local}
  local ref=$1 cand=$2 i j s pool v tag opts n pids=() tasks=() G
  shift 2
  local seeds=("$@")
  [ ${#seeds[@]} -gt 0 ] || seeds=(2 3)
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  mapfile -t nets < <(python -I - "$R2/results/pid_fusion_$ref.json" "$R2/results/pid_fusion_$cand.json" <<'EOF'
import json, re, sys
seen = []
for d in sys.argv[1:]:
    for x in json.load(open(d))["seq_official"]:
        m = re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", (x[0] if isinstance(x, list) else x).rsplit("/", 1)[-1])
        k = (m.group(1), m.group(2))
        if k not in seen:
            seen.append(k)
print("\n".join(f"{v} {t}" for v, t in seen))
EOF
  )
  echo "documents $ref / $cand: ${#nets[@]} distinct networks; new seeds ${seeds[*]}; ${#G[@]} GPU(s)"
  [ ${#nets[@]} -gt 0 ] || { echo "no networks found"; return 1; }
  for n in "${nets[@]}"; do for pool in pub2 priv2; do for s in "${seeds[@]}"; do tasks+=("$n $pool $s"); done; done; done
  for i in "${!G[@]}"; do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
        read -r v tag pool s <<< "${tasks[$j]}"
        [ -s "$O/offsim_${pool}_h3s${s}_net${v}${tag}_t50.npz" ] && continue
        if [ "$tag" = _m101x ]; then opts=(--ckpt-dir $A1/results/m101 --ckpt-suffix _s3 --state last); else opts=(--ckpt-dir $SN --ckpt-suffix "$tag"); fi
        CUDA_VISIBLE_DEVICES=${G[$i]} python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$pool" --net "$v" --horizon 1000 "${opts[@]}" --tag "$tag" \
          --steps 50 --holdout 3 --holdout-seed "$s" --out "$O" || echo "FAILED: net $v$tag $pool seed $s"
      done
    ) > logs/nano4/offmore-$J-ft-$i.log 2>&1 &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || echo "a fine-tune worker failed (logs/nano4/offmore-$J-ft-*.log)"; done
  echo "fine-tunes done at $(date +%H:%M:%S): $(grep -c FAILED logs/nano4/offmore-$J-ft-*.log | awk -F: '{n+=$2} END{print n}') failed; files of the new seeds: $(for s in "${seeds[@]}"; do ls $O | grep -c "_h3s${s}_.*_t50.npz"; done | tr '\n' ' ')"
  local name
  name=$(IFS=_; echo "${seeds[*]}")
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$cand.json --steps 50 --seeds "${seeds[@]}" --workers 8 \
    --out $R2/results/final/v2_offsim_${cand}_vs_${ref}_s${name}.json > logs/nano4/offmore-$J-new.log 2>&1 || echo "scoring on the new seeds failed"
  echo "== new seeds ${seeds[*]}: $cand vs $ref"
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/offmore-$J-new.log | cut -c1-240
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$cand.json --steps 50 --seeds 0 1 "${seeds[@]}" --workers 8 \
    --out $R2/results/final/v2_offsim_${cand}_vs_${ref}_s0_1_${name}.json > logs/nano4/offmore-$J-all.log 2>&1 || echo "scoring on all seeds failed"
  echo "== all seeds 0 1 ${seeds[*]}: $cand vs $ref"
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/offmore-$J-all.log | cut -c1-240
}
main "$@"; exit $?
