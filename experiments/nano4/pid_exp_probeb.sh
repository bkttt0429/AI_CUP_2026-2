#!/usr/bin/env bash
# METHOD_NOTES 2.88, probe B: do sequence networks get better when their per-ply input carries the SUPERVISED projection of the
# trunk vectors instead of the random one? Small scale: the simulated pools of folds 1 and 2 and up to 32 games of every
# player of folds 3 and 4 (about 130,000 games; never fold 0).
#   1. keys; the projection, fitted on the unprojected per-game rows of the players of folds 3 and 4 only
#   2. per-ply supervised features of those games at the subject's plies (8 shards)
#   3. eight small networks on those games only (symmetry + ply masking, 10,000 steps; nets 1 and 2 = the fold left out):
#        q0 the usual input                                   q1 the played-point half replaced by the supervised features
#        q2 the pooled half replaced                          q3 the supervised features alone (the rest set to its mean)
#   4. their fold vectors and the simulated-pool comparison against q0 (results/final/v2_sim14q.json)
# One job with 8 GPUs, from the AI_CUP_2026-1 checkout. The body is a function called on the last line with exit.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2/probeb S=/work/u2507115/pid_v2/seqnet
  local C="--steps 10000 --horizon 30000 --no-state --sym-aug --mask 0.3 --sup-require" specs=() v t
  mkdir -p $D
  [ -s $D/keys.txt ] || python -I $R2/scripts/pid_probe_fullrows.py --stage keys --per-player 32 --out $D/keys.txt
  [ -s $D/proj.npz ] || python -I $R2/scripts/pid_sup_fit.py --rows /work/u2507115/pid_v2/fullrows_train/rows /work/u2507115/pid_v2/fullrows/rows --fit-folds 3 4 --k 64 --out $D/proj.npz
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --keys $D/keys.txt --proj $D/proj.npz --out $D/ply --workers 10 || { echo "extraction FAILED"; return 1; }
  echo "supervised per-ply parts: $(ls $D/ply | grep -c npz), $(du -sh $D/ply | cut -f1)"
  for v in 1 2; do
    specs+=("$v q0 $C" "$v q1 $C --sup-dims 64 128" "$v q2 $C --sup-dims 0 64" "$v q3 $C --sup-dims 64 128 --sup-only")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply" "${specs[@]}" || echo "a trainer FAILED"
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _q0e --name _nq0e --out $R2/data/seq
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _q1e --name _nq1e --out $R2/data/seq --sup-ply $D/ply --sup-dims 64 128
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _q2e --name _nq2e --out $R2/data/seq --sup-ply $D/ply --sup-dims 0 64
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _q3e --name _nq3e --out $R2/data/seq --sup-ply $D/ply --sup-dims 64 128 --sup-only
  python -I $R2/scripts/pid_v2_sim14.py --names q0e=_nq0e q1e=_nq1e q2e=_nq2e q3e=_nq3e yse=_nyse --ref q0e --folds 1 2 --out $R2/results/final/v2_sim14q.json
}
main "$@"; exit $?
