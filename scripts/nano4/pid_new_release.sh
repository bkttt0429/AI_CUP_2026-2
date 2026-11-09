#!/usr/bin/env bash
# Re-released official files -> submissions, end to end (login node only submits and waits; all work runs in jobs).
# usage: bash scripts/nano4/pid_new_release.sh data/tests_<date>   (folder holding the four official CSVs + sample)
#   1. compare with the 10-30 release (data/tests) and run the official checks on the new files
#   2. trunk rows of every new official game: 2 jobs x 8 GPUs into data/trunk/<folder name>
#   3. predict fusion3o_ref_trunk_emb (main since 11-08, METHOD_NOTES 2.23: + learnt player embedding results/embed/embx_final.pt),
#      fusion3o_ref_trunk (previous main) and fusion3o_ref (no-trunk fallback), each with --assign off and auto.
#      Rule: emb with --assign auto (assignment applies only when the pool's Top-1 picks are >= 87.5% distinct); emb was
#      never worse than f3oreft in the plain or the rotated simulation, with or without assignment (2.22 symfit retired)
#   4. reconcile against the checks + format check against the new sample submission
#   5. if the assignment did not apply to a pool, it prints the B3 command (scripts/nano4/pid_b3_official.sh; METHOD_NOTES 2.26 option A)
# Nothing here uploads anything; which file to upload stays the user's call.
set -euo pipefail
R1=~/BK_AICUP_2026-1; R2=~/BK_AICUP_2026-2
T=$(cd "$R2" && realpath "${1:?usage: pid_new_release.sh <folder of the new official files>}")
TAG=$(basename "$T"); STORE=$R2/data/trunk/$TAG
EXC="--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]"
CANDS="$T/player_identification_candidates_public.csv $T/player_identification_candidates_private.csv"
QS="$T/player_identification_test_public.csv $T/player_identification_test_private.csv"
for f in $CANDS $QS $T/sample_submission_player.csv; do [ -s "$f" ] || { echo "missing $f (rename the new files to the 10-30 names first)"; ls -la "$T"; exit 2; }; done
cd "$R1"
sub() { local id; until id=$(sbatch --parsable $EXC "$@"); do sleep 60; done; echo "$id"; }
wait_ids() { for id in "$@"; do until [ -z "$(squeue -h -j "$id" 2>/dev/null)" ]; do sleep 30; done; done; }
cmp=$(sub -c 8 -t 01:00:00 -J p_relcmp scripts/nano4/run.sbatch python -I $R2/scripts/pid_compare_releases.py --old $R2/data/tests --new $T --out $R2/results/checks/release_compare_$TAG.json)
chk=$(sub -c 12 -t 02:00:00 -J p_chk_$TAG scripts/nano4/run.sbatch python -I $R2/scripts/pid_check_files.py --candidates $CANDS --questions $QS --fusion $R2/results/pid_fusion.json --out $R2/results/checks/official_checks_$TAG.json --workers 12)
tx=()
for node in 0 1; do
  tx+=("$(sub -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J p_trunk_$TAG scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_trunk_extract.py --official --tests-dir $T --out $STORE --node $node --nodes 2 --workers 10)")
done
echo "compare $cmp, checks $chk, trunk ${tx[*]}"
wait_ids "$cmp"; tail -n 60 logs/nano4/p_relcmp-$cmp.out
wait_ids "${tx[@]}"
n_parts=$(ls "$STORE"/*.npz 2>/dev/null | grep -vc tmp.npz || true); echo "trunk parts: $n_parts"
ids=(); outs=()
for spec in "emb:$R2/results/pid_fusion3o_ref_trunk_emb.json:--trunk-store $STORE" "f3oreft:$R2/results/pid_fusion3o_ref_trunk.json:--trunk-store $STORE" "f3oref:$R2/results/pid_fusion3o_ref.json:"; do
  IFS=: read -r name fus extra <<<"$spec"
  for mode in off auto; do
    out=submission_${TAG}_$name; [ $mode = auto ] && out=${out}_assign
    outs+=("$out")
    ids+=("$(sub -c 12 -t 01:30:00 -J p_$out scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $fus --out $R2/results/$out/submission_player.csv --workers 12 --assign $mode $extra)")
  done
done
wait_ids "$chk" "${ids[@]}"
for i in "${!ids[@]}"; do echo "== ${outs[$i]} (job ${ids[$i]})"; grep -E "^wrote|assign|Error|Traceback" logs/nano4/p_${outs[$i]}-${ids[$i]}.out | tail -n 4; done
for i in "${!ids[@]}"; do
  if [ "${outs[$i]}" = "submission_${TAG}_emb_assign" ] && grep -q "not applied" "logs/nano4/p_${outs[$i]}-${ids[$i]}.out"; then
    echo "B3 (METHOD_NOTES 2.26, option A): the assignment did not apply to a pool -> bash scripts/nano4/pid_b3_official.sh $T (8 GPUs x 2 pools, ~2.5 h)"
  fi
done
cat > $R2/data/day_scripts/pid_release_check_$TAG.sh <<EOS
set -uo pipefail
for v in ${outs[*]}; do
  echo "== reconcile \$v"
  python -I $R2/scripts/pid_reconcile.py --checks $R2/results/checks/official_checks_$TAG.json --submission $R2/results/\$v/submission_player.csv --out $R2/results/checks/official_\${v}_reconcile.json; echo "exit \$?"
done
python -I - <<'PY'
import pandas as pd
T, R2 = "$T", "$R2"
sample = pd.read_csv(f"{T}/sample_submission_player.csv", dtype=str)
pools = {"pub_": set(pd.read_csv(f"{T}/player_identification_candidates_public.csv", dtype=str, usecols=["player_id"])["player_id"]),
         "priv_": set(pd.read_csv(f"{T}/player_identification_candidates_private.csv", dtype=str, usecols=["player_id"])["player_id"])}
for v in "${outs[*]}".split():
    s = pd.read_csv(f"{R2}/results/{v}/submission_player.csv", dtype=str)
    tops = s[[f"top_{i}" for i in range(1, 6)]].values
    print(v, "rows", len(s), "= sample", len(s) == len(sample), "columns", list(s.columns) == list(sample.columns),
          "order", list(s.question_id) == list(sample.question_id), "five distinct", all(len(set(r)) == 5 for r in tops),
          "own pool", all(all(p in pools[q[:q.index('_') + 1]] for p in r) for q, r in zip(s.question_id, tops)),
          "distinct top1", {k: int(s[s.question_id.str.startswith(k)].top_1.nunique()) for k in pools}, "na", int(s.isna().sum().sum()))
PY
EOS
fin=$(sub -c 8 -t 00:30:00 -J p_relfin scripts/nano4/run.sbatch bash $R2/data/day_scripts/pid_release_check_$TAG.sh)
wait_ids "$fin"; cat logs/nano4/p_relfin-$fin.out
