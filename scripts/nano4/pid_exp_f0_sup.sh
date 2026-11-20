#!/usr/bin/env bash
# METHOD_NOTES 2.201 (fold 0; the user's consent of 2026-11-18 00:29): the SECOND supervised feature set ("js", as
# scripts/nano4/pid_exp_o14_x.sh comp: the same projection, scalars and scale) for every training game of the players of
# fold 0 -> /work/u2507115/pid_v2/supj/comp0. The networks of the second feature set train with --sup-require, so a
# fold-0 game without these features would silently stay out of the training.
#   pid_exp_f0_sup.sh ext <first shard> <shards in total>    8 shards on the node's 8 GPUs (ONE GPU per process: nodes
#                                                            whose multi-GPU check fails can run it); resumable per part
#   pid_exp_f0_sup.sh check                                  every key of the list has its row in the parts -> comp0.CHECK_PASSED
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj rc=0 free
  local K=$O/fold0_keys.txt
  [ -s $P/stats_js.npz ] && [ -s $P/proj3.npz ] || { echo "the projection or the scale of the second feature set is missing"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 7000 ] || { echo "/work has only $free MB free: not started (the parts need about 1.2 GB, the floor is 4 GB)"; return 1; }
  [ -s $K ] || python -I $R2/scripts/pid_fold0_keys.py $K || return 1
  echo "key list: $(wc -l < $K) games of fold 0; node $(hostname); $(date '+%F %T')"
  case $1 in
    ext)  # shellcheck disable=SC2086
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --keys $K --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --out $O/comp0; rc=$? ;;
    check)
      python3 -I - $K $O/comp0 <<'PY' || return 1
import glob, sys
import numpy as np
keys = set(open(sys.argv[1]).read().split())
have, plies, empty = set(), 0, 0
for f in sorted(glob.glob(sys.argv[2] + "/sup_s*_p*.npz")):
    z = np.load(f)
    assert "zq" in z.files and z["zq"].shape[1] == 68 and z["zq"].shape[0] == int(z["n"].sum()) == len(z["t"]), f
    k = z["keys"].astype(str).tolist()
    assert not (set(k) & have), f"a key twice ({f})"
    have.update(k); plies += int(z["n"].sum()); empty += int((z["n"] == 0).sum())
print(f"{len(keys)} keys in the list, {len(have)} in the parts, {len(keys - have)} missing, {len(have - keys)} foreign; {plies} plies, {empty} games without plies")
assert have == keys and empty <= len(keys) // 200
PY
      echo "check passed $(date '+%F %T')" > $O/comp0.CHECK_PASSED ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
  echo "parts: comp0 $(ls $O/comp0 2>/dev/null | grep -c 'npz$'); $(du -sh $O/comp0 | cut -f1); /work free $(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9') MB; $(date '+%F %T')"
  return $rc
}
main "$@"; exit $?
