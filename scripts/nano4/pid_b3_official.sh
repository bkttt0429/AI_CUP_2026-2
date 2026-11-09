#!/usr/bin/env bash
# B3 on re-released official files (METHOD_NOTES 2.26, user's option A, 2026-11-09): run ONLY when pid_new_release.sh reports that the
# one-to-one assignment did not apply to a pool (distinct Top-1 < 87.5%); with the assignment on, main+b3 = main on every pool test.
# usage: bash scripts/nano4/pid_b3_official.sh data/tests_<date>   (after pid_new_release.sh: needs the trunk store data/trunk/<folder name>)
#   1. build the rank-test-format inputs per pool (candidates in chunks of <= 20 games + the test file)
#   2. per pool, the rank session's test-side path (read-only): predict_submission.py shards (8 GPUs) + trunk_predict.py extract (8 GPUs)
#      -> merge -> capture ct9's zq (24 members x 2 L1b heads, 1 GPU). Old Public (45k games) took 2 h 07 + 28 min in parallel.
#   3. predict main+b3 (results/pid_fusion3o_ref_trunk_emb_b3.json) with --assign off and auto + format check
# Nothing here uploads anything; which file to upload stays the user's call.
set -euo pipefail
R1=~/BK_AICUP_2026-1; R2=~/BK_AICUP_2026-2
T=$(cd "$R2" && realpath "${1:?usage: pid_b3_official.sh <folder of the new official files>}")
TAG=$(basename "$T"); STORE=${B3_STORE:-$R2/data/trunk/$TAG}; W=/work/u2507115/pid_b3/official_$TAG  # B3_STORE: another trunk store (the 10-30 files: data/trunk/official)
FUS=$R2/results/pid_fusion3o_ref_trunk_emb_b3.json
EXC="--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]"
HEADS="--l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt"
CANDS="$T/player_identification_candidates_public.csv $T/player_identification_candidates_private.csv"
QS="$T/player_identification_test_public.csv $T/player_identification_test_private.csv"
[ -s "$FUS" ] || { echo "missing $FUS (pid_b3_pools.py eval --final-fusion-out)"; exit 2; }
ls "$STORE"/*.npz >/dev/null 2>&1 || { echo "missing trunk store $STORE (run pid_new_release.sh first)"; exit 2; }
cd "$R1"
sub() { local id; until id=$(sbatch --parsable $EXC "$@"); do sleep 60; done; echo "$id"; }
wait_ids() { for id in "$@"; do until [ -z "$(squeue -h -j "$id" 2>/dev/null)" ]; do sleep 60; done; done; }
B=$(sub -c 8 -t 00:30:00 -J b3o_build scripts/nano4/run.sbatch python -I $R2/scripts/pid_b3_official.py build --tests-dir $T --work $W)
wait_ids "$B"; grep -E "^(pub|priv):|Error|Traceback|SystemExit" logs/nano4/b3o_build-$B.out || true
caps=()
for P in pub priv; do
  IN="$W/$P/inputs/cands.csv $W/$P/inputs/quest.csv"
  EXP=$(python3 -c "import json; print(' '.join(json.load(open('$W/$P/meta.json'))['expect']))")
  mkdir -p "$W/$P/vec"
  S=$(sub --gres=gpu:8 -c 96 -t 08:00:00 -J b3o_s_$P scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --inputs $IN --out $W/$P/submission_rank.csv $HEADS --workers 10 $EXP)
  V=$(sub --gres=gpu:8 -c 96 -t 04:00:00 -J b3o_v_$P scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/trunk_predict.py extract --inputs $IN --vec $W/$P/vec --workers 10 $EXP)
  M=$(sub --dependency=afterok:$S -c 12 -t 01:00:00 -J b3o_m_$P scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8 --inputs $IN --out $W/$P/submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff_hd_r.json --allow-missing-hd $EXP)
  C=$(sub --dependency=afterok:$M:$V --gres=gpu:1 -c 12 -t 01:30:00 -J b3o_c_$P scripts/nano4/run.sbatch python -I $R2/scripts/pid_b3_official.py capture --work $W/$P)
  echo "$P: score $S vec $V merge $M capture $C"; caps+=("$C")
done
wait_ids "${caps[@]}"
for c in "${caps[@]}"; do grep -hE "passes over|SystemExit|Error|Traceback" logs/nano4/b3o_c_*-$c.out | tail -n 2 || true; done
ids=(); outs=()
for mode in off auto; do
  out=submission_${TAG}_emb_b3; [ $mode = auto ] && out=${out}_assign
  outs+=("$out")
  ids+=("$(sub -c 12 -t 01:30:00 -J p_$out scripts/nano4/run.sbatch python -I $R2/scripts/predict_pid.py --candidates $CANDS --questions $QS --fusion $FUS --out $R2/results/$out/submission_player.csv --workers 12 --assign $mode --trunk-store $STORE --b3-dir $W)")
done
wait_ids "${ids[@]}"
for i in "${!ids[@]}"; do echo "== ${outs[$i]} (job ${ids[$i]})"; grep -E "^wrote|b3 component|Error|Traceback" logs/nano4/p_${outs[$i]}-${ids[$i]}.out | tail -n 4 || true; done
cat > $R2/data/day_scripts/pid_b3_check_$TAG.py <<EOS
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
EOS
fin=$(sub -c 4 -t 00:20:00 -J p_b3fin scripts/nano4/run.sbatch python -I $R2/data/day_scripts/pid_b3_check_$TAG.py)
wait_ids "$fin"; grep -v "^+" logs/nano4/p_b3fin-$fin.out
