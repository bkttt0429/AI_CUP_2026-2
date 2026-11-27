#!/usr/bin/env bash
# 11/04 driver: find the four official files in ~/BK_AICUP_2026-2/data/tests, submit the check job and the four
# predict variants (fusion / fusion3 x off / --assign auto) as separate 1-GPU-node CPU jobs, wait per job id,
# then print the flags and the report summaries. Reconcile is run afterwards by hand (NANO4 4.2 step 3).
set -euo pipefail
R1=~/BK_AICUP_2026-1; R2=~/BK_AICUP_2026-2; T=$R2/data/tests
EXC="--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]"
cd "$T"
pick() { # $1 = public|private ; $2 = cand|q
  local files; files=$(ls | grep -i -E "\.csv$" | grep -i "$1" | grep -v -i -E "sample|readme|truth|submission" || true)
  if [ "$2" = cand ]; then echo "$files" | grep -i candidate | head -1; else echo "$files" | grep -v -i candidate | head -1; fi
}
CP=$(pick public cand); CV=$(pick private cand); QP=$(pick public q); QV=$(pick private q)
for v in CP CV QP QV; do [ -n "${!v}" ] || { echo "cannot find $v in $T:"; ls -la; exit 2; }; done
echo "candidates: $CP $CV"; echo "questions:  $QP $QV"
wc -l "$CP" "$CV" "$QP" "$QV"
CANDS="$T/$CP $T/$CV"; QS="$T/$QP $T/$QV"
cd "$R1"
sub() { local id; until id=$(sbatch --parsable $EXC -c 12 "$@"); do sleep 60; done; echo "$id"; }
ids=()
ids+=("$(sub -t 02:00:00 -J pidchk scripts/nano4/run.sbatch python -I $R2/scripts/pid_check_files.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion.json --out $R2/results/checks/official_checks.json --workers 12)")
ids+=("$(sub -t 02:00:00 -J pidsub scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion.json --out $R2/results/submission/submission_player.csv --expect-rows 800 --workers 12)")
ids+=("$(sub -t 02:00:00 -J pidsubf3 scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion3.json --out $R2/results/submission_f3/submission_player.csv --expect-rows 800 --workers 12)")
ids+=("$(sub -t 02:00:00 -J pidsubas scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion.json --out $R2/results/submission_assign/submission_player.csv --expect-rows 800 --workers 12 --assign auto)")
ids+=("$(sub -t 02:00:00 -J pidsubf3as scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion3.json --out $R2/results/submission_f3_assign/submission_player.csv --expect-rows 800 --workers 12 --assign auto)")
echo "jobs: ${ids[*]}"
for id in "${ids[@]}"; do until [ -z "$(squeue -h -j "$id" 2>/dev/null)" ]; do sleep 60; done; echo "done $id"; done
for f in logs/nano4/pidchk-${ids[0]}.out logs/nano4/pidsub-${ids[1]}.out logs/nano4/pidsubf3-${ids[2]}.out logs/nano4/pidsubas-${ids[3]}.out logs/nano4/pidsubf3as-${ids[4]}.out; do echo "== $f"; tail -n 25 "$f"; done
