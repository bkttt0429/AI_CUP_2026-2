#!/usr/bin/env bash
# METHOD_NOTES 2.202: /home (100 GB quota, shared with the rank session) has no room for the vector stores of further
# families (424 MB per family: 75 MB real pools, 223 MB pseudo-pools, 126 MB old-pool simulation), /work has. This moves
# the PSEUDO-POOL and OLD-POOL vectors of the given families to /work and leaves absolute symlinks at the old paths, so
# every reader (np.load, find -name, os.path.exists) keeps working. The real-pool vectors (data/seq_ft) stay on /home.
#   pid_vec_overflow.sh <family> [<family> ...]            e.g. pid_vec_overflow.sh o40F
# Per file: copy, compare (cmp), rename into place on /work, then replace the original ATOMICALLY by a symlink
# (ln -s to a temporary name + mv -T), so a reader that has the file open keeps the old inode and a later open follows
# the link. Only regular files are touched (a second run does nothing). Every move is appended to
# results/final/moved_vectors_<date>.txt.
# NOT A DUPLICATE: the files under /work/u2507115/pid_v2/vec_overflow are the ONLY copies - never delete them as
# "duplicates of /home" (memory: delete-check-symlinks). Run only for families whose vector task has ended.
# Login node (file management only, no compute).
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 W=/work/u2507115/pid_v2/vec_overflow fam sub d f b n=0 kb=0 free list
  [ $# -ge 1 ] || { echo "usage: pid_vec_overflow.sh <family> [<family> ...]"; return 2; }
  list=$R2/results/final/moved_vectors_$(date +%Y%m%d).txt
  for fam in "$@"; do
    case "$fam" in o[0-9]*F) ;; *) echo "family name expected (o40F), got $fam"; return 2 ;; esac
    for sub in offsim oldsim; do
      d=$R2/data/pid_v2_seq_ft_$sub
      mkdir -p $W/$sub || return 1
      for f in $d/${sub}_*_net-1_${fam}s[0-9]*e_t*.npz; do
        { [ -f "$f" ] && [ ! -L "$f" ]; } || continue
        free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
        [ "$free" -ge "${MIN_WORK_MB:-6000}" ] || { echo "/work has only $free MB free: stopped after $n files"; return 1; }
        b=${f##*/}
        cp -p "$f" "$W/$sub/$b.part" && cmp -s "$f" "$W/$sub/$b.part" && mv -f "$W/$sub/$b.part" "$W/$sub/$b" \
          || { echo "copy of $b failed: stopped after $n files"; rm -f "$W/$sub/$b.part"; return 1; }
        kb=$((kb + $(du -k "$f" | cut -f1)))
        ln -s "$W/$sub/$b" "$f.lnk" && mv -T "$f.lnk" "$f" || { echo "link of $b failed: stopped after $n files"; rm -f "$f.lnk"; return 1; }
        echo "$(date '+%F %T') $f -> $W/$sub/$b" >> "$list"
        n=$((n + 1))
      done
    done
  done
  echo "moved $n files ($((kb / 1024)) MB) of $* to $W; list $list"
  echo "/home free $(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc '0-9') MB, /work free $(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9') MB (both lag by up to a minute)"
}
main "$@"; exit $?
