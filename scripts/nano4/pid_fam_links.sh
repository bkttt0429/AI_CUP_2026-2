#!/usr/bin/env bash
# The member links ckpt_-1_<family>s<k>e.pt of the families of my official networks whose members were trained under
# recipe tags (METHOD_NOTES 2.201, 2.204, 2.207, 2.211). One place for the chains and the fine-tune helpers.
#   pid_fam_links.sh <family> [<family> ...]      prints "<family> <linked checkpoints>" per family; rc 1 when one has < 7
#     o44F o50F o58F             members 0-3 = <base>Ss0-3, 4-7 = <base>SMs0-3
#     o45F-o49F o51F-o53F o56F o57F   members 0-5 = <base>As0-5, 6 / 7 = <base>Cs0 / Cs1
#     o62F o66F o70F as o44F; o60F o61F o63F o64F o65F o67F o68F o69F o71F as o45F (METHOD_NOTES 2.220, 2.229)
#     o53KF / o56KF              the SNAPSHOTS of o53 at 50,000 / of o56 at 45,000 steps (ckpt_-1_<tag>k50e.pt / k45e.pt)
# Any other family: nothing is linked, the existing links are counted. Links only (ln -sf of names in the same folder).
set -uo pipefail
main() {
  local SN=/work/u2507115/pid_v2/seqnet f b sfx i k n rc=0
  for f in "$@"; do
    b=${f%F}; sfx=""
    case "$f" in o53KF) b=o53; sfx=k50 ;; o56KF) b=o56; sfx=k45 ;; esac
    case "$f" in
      o44F|o50F|o58F|o62F|o66F|o70F)
        for i in 0 1 2 3; do
          [ -s $SN/ckpt_-1_${b}Ss${i}e.pt ] && ln -sf ckpt_-1_${b}Ss${i}e.pt $SN/ckpt_-1_${f}s${i}e.pt
          [ -s $SN/ckpt_-1_${b}SMs${i}e.pt ] && ln -sf ckpt_-1_${b}SMs${i}e.pt $SN/ckpt_-1_${f}s$((i + 4))e.pt
        done ;;
      o45F|o46F|o47F|o48F|o49F|o51F|o52F|o53F|o56F|o57F|o59F|o60F|o61F|o63F|o64F|o65F|o67F|o68F|o69F|o71F|o53KF|o56KF)
        for i in 0 1 2 3 4 5; do [ -s $SN/ckpt_-1_${b}As${i}${sfx}e.pt ] && ln -sf ckpt_-1_${b}As${i}${sfx}e.pt $SN/ckpt_-1_${f}s${i}e.pt; done
        for i in 0 1; do [ -s $SN/ckpt_-1_${b}Cs${i}${sfx}e.pt ] && ln -sf ckpt_-1_${b}Cs${i}${sfx}e.pt $SN/ckpt_-1_${f}s$((i + 6))e.pt; done ;;
    esac
    n=0; for k in $(seq 0 23); do [ -s $SN/ckpt_-1_${f}s${k}e.pt ] && n=$((n + 1)); done
    echo "$f $n"
    [ "$n" -ge 7 ] || rc=1
  done
  return $rc
}
main "$@"; exit $?
