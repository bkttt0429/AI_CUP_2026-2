#!/usr/bin/env bash
# METHOD_NOTES 2.112: official networks on the SECOND supervised feature set ("js": projection fitted jointly over pooled /
# played-point / policy-expected vectors, 64 dims, + 4 policy scalars; probe P6), to be compared with o12 (first set) at the
# same length. One job does everything, so nothing depends on a conversation being alive:
#   1. the pools' features of the second set, if they are not there yet (8 shards, about 13 minutes)
#   2. eight official networks (competition folds 1-4, never fold 0; 116,420 external accounts; symmetry + ply masking 0.3,
#      half of an episode external, 20,000 steps on the 60,000-step cosine = the schedule of o12's 20,000-step snapshots, EMA):
#      o14As0-5  the pooled half and four more input dimensions replaced by the 68 features (o12's type A)
#      o14Cs0-1  the 68 features alone (type C)
#   3. their chain against the live composition (scripts/nano4/pid_chain_job3.sh new v2o14 o14F; live = pid_v2c2_bs.csv)
# Submit with -d afterok:<the extraction jobs>. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k L=""
  echo "second feature set: comp $(ls $O/comp 2>/dev/null | grep -c 'npz$') parts, ext $(ls $O/ext 2>/dev/null | grep -c 'npz$'), pools $(ls $O/pools 2>/dev/null | grep -c 'npz$'); $(du -sh $O | cut -f1)"
  if [ "$(ls $O/pools 2>/dev/null | grep -c 'npz$')" -lt 192 ]; then
    bash $R2/scripts/nano4/pid_exp_o14_x.sh pools || { echo "pools extraction FAILED"; return 1; }
  fi
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 20000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"
  for k in 0 1 2 3 4 5; do specs+=("-1 o14As$k $C --seed $((k + 60))"); L="$L o14F:$k=ckpt_-1_o14As${k}e.pt"; done
  for k in 0 1; do specs+=("-1 o14Cs$k $C --sup-only --seed $((k + 70))"); L="$L o14F:$((k + 6))=ckpt_-1_o14Cs${k}e.pt"; done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}" || echo "a trainer FAILED (the chain runs with the networks that exist)"
  LINKS="$L" bash $R2/scripts/nano4/pid_chain_job3.sh new v2o14 o14F
}
main "$@"; exit $?
