#!/usr/bin/env bash
# METHOD_NOTES 2.248: the /work half of the move-to-PC staging, COPIED into /home in parts so that the user can download it
# from the home directory (as the rank project did). Source: /work/u2507115/AICUP2_MOVE_20261121 (hard links; untouched).
#   part 1  README / LEFT_OUT / MANIFEST_work.sha256 / SEQNET_LINKS.txt, the pools' per-ply inputs, seqnet_final files 1-68
#   part 2  seqnet_final files 69-136
#   part 3  seqnet_fallback_only (64 files)
# Destination /home/u2507115/AICUP2_MOVE_20261121_work_part<N>/ with the same relative paths (the three parts merge into
# one tree) and MANIFEST_part<N>.sha256; every copy is checked against MANIFEST_work.sha256. A part is only started when
# /home keeps at least 4 GB free after it. Nothing is deleted here.
#   pid_move_part.sh <1|2|3> [<1|2|3> ...]
set -uo pipefail
main() {
  local W=/work/u2507115/AICUP2_MOVE_20261121 n D L need free f rc=0
  for n in "$@"; do
    D=/home/u2507115/AICUP2_MOVE_20261121_work_part$n
    L=$(mktemp "${TMPDIR:-/tmp}/movepart_XXXXXX")
    case "$n" in
      1) { printf '%s\n' ./README.txt ./LEFT_OUT.txt ./SYMLINKS_IN_STAGING.txt ./MANIFEST_work.sha256 ./work/pid_v2/SEQNET_LINKS.txt
           (cd $W && find ./work/pid_v2/ply_official ./work/pid_v2/sup_pools ./work/pid_v2/supj -type f | sort)
           (cd $W && find ./work/pid_v2/seqnet_final -type f | sort | sed -n '1,68p'); } > "$L" ;;
      2) (cd $W && find ./work/pid_v2/seqnet_final -type f | sort | sed -n '69,200p') > "$L" ;;
      3) (cd $W && find ./work/pid_v2/seqnet_fallback_only -type f | sort) > "$L" ;;
      *) echo "part $n: 1, 2 or 3"; rm -f "$L"; return 2 ;;
    esac
    need=$(cd $W && tr '\n' '\0' < "$L" | xargs -0 du -cm | tail -1 | cut -f1)
    free=$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc '0-9')
    if [ -e "$D/MANIFEST_part$n.sha256" ]; then echo "part $n exists already ($D)"; rm -f "$L"; continue; fi
    if [ "$free" -lt $(( need + 4000 )) ]; then echo "part $n needs $need MB, /home has $free MB free: NOT started (4 GB must stay free)"; rm -f "$L"; rc=3; break; fi
    echo "part $n: $(wc -l < "$L") files, $need MB -> $D (free on /home: $free MB), start $(date +%T)"
    mkdir -p "$D" || { rm -f "$L"; return 1; }
    (cd $W && rsync -a --files-from="$L" ./ "$D/") || { echo "part $n: the copy failed"; rm -f "$L"; return 1; }
    (cd $W && grep -F -f <(sed 's/^/  /' "$L") MANIFEST_work.sha256 | grep -v '  ./MANIFEST_work.sha256$') > "$D/MANIFEST_part$n.sha256.tmp"
    f=$(cd "$D" && sha256sum -c --quiet MANIFEST_part$n.sha256.tmp 2>&1 | grep -c -v '^$')
    if [ "$f" != 0 ]; then echo "part $n: $f files differ from the manifest - the copy is NOT valid"; (cd "$D" && sha256sum -c --quiet MANIFEST_part$n.sha256.tmp 2>&1 | head -5); rm -f "$L"; return 1; fi
    mv "$D/MANIFEST_part$n.sha256.tmp" "$D/MANIFEST_part$n.sha256"
    echo "part $n ready: $(find "$D" -type f | wc -l) files, $(du -sm "$D" | cut -f1) MB, every file matches the manifest ($(wc -l < "$D/MANIFEST_part$n.sha256") checked); free on /home now $(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc '0-9') MB; $(date +%T)"
    rm -f "$L"
  done
  return $rc
}
main "$@"; exit $?
