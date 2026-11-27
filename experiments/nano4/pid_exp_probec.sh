#!/usr/bin/env bash
# METHOD_NOTES 2.98, probe C: does the advantage of supervised per-ply features hold when the training set grows? Probe B had
# 100,000 games; here every competition game of the training folds (no external accounts): 590,000 games per network.
#   pid_exp_probec.sh extract <first shard> <shards in total>   supervised features of the competition games not yet extracted
#                                                               (int8, the scale of the probe-B store); 8 shards per job
#   pid_exp_probec.sh train     eight networks on the competition games only (symmetry + ply masking, 20,000 steps; nets 1, 2):
#                               u0 the usual input, u1 the played-point half replaced, u2 the pooled half replaced, u3 the
#                               supervised features alone; then fold vectors, the simulated-pool comparison (v2_sim14u.json)
#                               and the comparison with the row-embedding fusion (v2_probe_c_fuse.json)
# From the AI_CUP_2026-1 checkout, 8 GPUs. The body is a function called on the last line with exit.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probec S=/work/u2507115/pid_v2/seqnet
  mkdir -p $D
  case $1 in
    extract)
      if [ ! -s $B/stats.npz ]; then
        python -I -c "import sys; sys.path.insert(0, '$R2/scripts'); import pid_sup; print('stats over', pid_sup.write_stats(['$B/ply'], '$B/stats.tmp.npz'), 'plies')" && mv $B/stats.tmp.npz $B/stats.npz
      fi
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --keys /work/u2507115/pid_v2/fullrows_train/keys.txt --skip-keys $B/keys.txt \
        --proj $B/proj.npz --stats $B/stats.npz --out $D/ply --workers 10
      echo "parts: $(ls $D/ply | grep -c npz), $(du -sh $D/ply | cut -f1)" ;;
    train)
      local SUP="--sup-ply $B/ply $D/ply --sup-stats $B/stats.npz" C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --sup-require" specs=() v
      for v in 1 2; do
        specs+=("$v u0 $C" "$v u1 $C --sup-dims 64 128" "$v u2 $C --sup-dims 0 64" "$v u3 $C --sup-dims 64 128 --sup-only")
      done
      bash $R2/scripts/nano4/pid_seq_train_fork.sh "$SUP" "${specs[@]}" || echo "a trainer FAILED"
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _u0e --name _nu0e --out $R2/data/seq
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _u1e --name _nu1e --out $R2/data/seq --sup-ply $B/ply $D/ply --sup-stats $B/stats.npz --sup-dims 64 128
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _u2e --name _nu2e --out $R2/data/seq --sup-ply $B/ply $D/ply --sup-stats $B/stats.npz --sup-dims 0 64
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _u3e --name _nu3e --out $R2/data/seq --sup-ply $B/ply $D/ply --sup-stats $B/stats.npz --sup-dims 64 128 --sup-only
      python -I $R2/scripts/pid_v2_sim14.py --names u0e=_nu0e u1e=_nu1e u2e=_nu2e u3e=_nu3e zme=_nzme yse=_nyse --ref u0e --folds 1 2 --out $R2/results/final/v2_sim14u.json
      python -I $R2/scripts/pid_probe_b_fuse.py --tags u0e u1e u2e u3e --ref u0e --out $R2/results/final/v2_probe_c_fuse.json ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
}
main "$@"; exit $?
