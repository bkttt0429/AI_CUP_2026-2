#!/usr/bin/env bash
set -uo pipefail
R2=~/BK_AICUP_2026-2
for v in submission submission_f3 submission_assign submission_f3_assign; do
  echo "== reconcile $v"
  python -I $R2/scripts/pid_reconcile.py --checks $R2/results/checks/official_checks.json --submission $R2/results/$v/submission_player.csv --out $R2/results/checks/official_${v}_reconcile.json
  echo "exit $?"
done
echo "== format check against sample_submission_player.csv"
python -I - <<'PY'
import pandas as pd
R2="/home/u2507115/BK_AICUP_2026-2"
sample=pd.read_csv(f"{R2}/data/tests/sample_submission_player.csv",dtype=str)
pools={"pub_":set(pd.read_csv(f"{R2}/data/tests/player_identification_candidates_public.csv",dtype=str,usecols=["player_id"])["player_id"]),
       "priv_":set(pd.read_csv(f"{R2}/data/tests/player_identification_candidates_private.csv",dtype=str,usecols=["player_id"])["player_id"])}
for v in ["submission","submission_f3","submission_assign","submission_f3_assign"]:
    s=pd.read_csv(f"{R2}/results/{v}/submission_player.csv",dtype=str)
    ok_cols=list(s.columns)==list(sample.columns); same_q=list(s.question_id)==list(sample.question_id)
    tops=s[[f"top_{i}" for i in range(1,6)]].values
    distinct=all(len(set(r))==5 for r in tops)
    inpool=all(all(p in pools[q[:q.index('_')+1]] for p in r) for q,r in zip(s.question_id,tops))
    print(v, "rows",len(s),"columns",ok_cols,"question order = sample",same_q,"five distinct",distinct,"in own pool",inpool,"na",int(s.isna().sum().sum()))
PY
