#!/usr/bin/env bash
# METHOD_NOTES 2.119, probe K: a STRONG KataGo network (kata1-b18c384nbt-s9996604416; already on disk, the rank session's M74
# read it) as a second feature extractor - at probe scale only (the probe-B games). Nothing here enters a submission:
# whether the network may be used for one is the participant's decision (an external resource to disclose).
#   1. per-game rows of the probe games of folds 3 / 4 under the strong network (for the projection fit; never fold 0)
#   2. the joint projection [pooled, played-point, policy-expected] -> 64 dims, as probe P6 on humanv0
#   3. per-ply features of all probe games: 64 dims + 4 policy scalars under the STRONG policy -> probek/ply_ks
#   4. small networks as in probe P6 (folds 1 and 2 left out, symmetry + ply masking, 10,000 steps):
#        pk / pkb   type A (the pooled half and four more input dimensions replaced by the 68 features), two seeds
#        pkc        type C (the 68 features alone)
#        p6sc       control: one more network on humanv0's second feature set (probep/ply_js), a new seed
#   5. fold vectors, then the simulated pools (scripts/pid_v2_sim14.py): single networks, and ensembles
#        e2 = q2e + p6se          e3k = e2 + pke          e3j = e2 + p6sce (a third member of an EXISTING representation)
#        e4k = e2 + pke + pkce    e4kb = e2 + pke + pkbe
#      primary reading: e3k against e3j (results/final/v2_sim14pk_j.json); also everything against e2 (v2_sim14pk.json)
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probek P=/work/u2507115/pid_v2/probep S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local K=/home/u2507115/BK_AICUP_2026-1/models/katago/kata1_b18c384nbt/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt
  local C="--steps 10000 --horizon 30000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only" G p pids=() rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -ge 8 ] || { echo "need 8 GPUs, have ${#G[@]}"; return 1; }
  mkdir -p $D
  if [ ! -s $D/proj3k.npz ]; then
    python -I $R2/scripts/pid_probe_k_keys.py --keys $B/keys.txt --folds 3 4 --out $D/keys_fit.txt || return 1
    bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_pol_extract.py --ckpt $K --rows-only --keys $D/keys_fit.txt --out $D --workers 10 || { echo "rows extraction FAILED"; tail -n 5 logs/nano4/shard0-$J.log; return 1; }
    echo "rows: $(ls $D/rows | grep -c 'npz$') parts at $(date +%H:%M:%S)"
    python -I $R2/scripts/pid_sup_fit.py --rows $D/rows --blocks g x e --fit-folds 3 4 --k 64 --out $D/proj3k.npz || { echo "projection fit FAILED"; return 1; }
  fi
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --ckpt $K --keys $B/keys.txt --proj $D/proj3k.npz --scalars --out $D/ply_ks --workers 10 || { echo "per-ply extraction FAILED"; tail -n 5 logs/nano4/shard0-$J.log; return 1; }
  echo "per-ply parts: $(ls $D/ply_ks | grep -c 'npz$'); $(du -sh $D | cut -f1) at $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=${G[0]},${G[1]},${G[2]},${G[3]},${G[4]},${G[5]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_ks" \
    "1 pk $C --sup-dims 0 68" "2 pk $C --sup-dims 0 68" "1 pkb $C --sup-dims 0 68 --seed 1" "2 pkb $C --sup-dims 0 68 --seed 1" \
    "1 pkc $C --sup-dims 0 68 --sup-only" "2 pkc $C --sup-dims 0 68 --sup-only" > logs/nano4/pktrain_k-$J.log 2>&1 &
  pids+=($!)
  CUDA_VISIBLE_DEVICES=${G[6]},${G[7]} bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $P/ply_js" \
    "1 p6sc $C --sup-dims 0 68 --seed 2" "2 p6sc $C --sup-dims 0 68 --seed 2" > logs/nano4/pktrain_j-$J.log 2>&1 &
  pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a trainer group FAILED"; rc=1; }; done
  tail -n 2 logs/nano4/pktrain_k-$J.log logs/nano4/pktrain_j-$J.log | cut -c1-200
  emb() {  # <GPU> <tag> <supervised options>
    CUDA_VISIBLE_DEVICES=$1 python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix "_$2" --name "_n$2" --out $R2/data/seq "${@:3}" > logs/nano4/pkemb_$2-$J.log 2>&1
  }
  pids=()
  emb ${G[0]} pke --sup-ply $D/ply_ks --sup-dims 0 68 & pids+=($!)
  emb ${G[1]} pkbe --sup-ply $D/ply_ks --sup-dims 0 68 & pids+=($!)
  emb ${G[2]} pkce --sup-ply $D/ply_ks --sup-dims 0 68 --sup-only & pids+=($!)
  emb ${G[3]} p6sce --sup-ply $P/ply_js --sup-dims 0 68 & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a fold-vector run FAILED"; rc=1; }; done
  local N="q2e=_nq2e p6se=_np6se pke=_npke pkbe=_npkbe pkce=_npkce p6sce=_np6sce e2=_nq2e+_np6se e3k=_nq2e+_np6se+_npke e3j=_nq2e+_np6se+_np6sce e4k=_nq2e+_np6se+_npke+_npkce e4kb=_nq2e+_np6se+_npke+_npkbe"
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_sim14.py --names $N --ref e3j --folds 1 2 --out $R2/results/final/v2_sim14pk_j.json || rc=1
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_sim14.py --names $N --ref e2 --folds 1 2 --out $R2/results/final/v2_sim14pk.json || rc=1
  return $rc
}
main "$@"; exit $?
