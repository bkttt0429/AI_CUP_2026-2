#!/usr/bin/env bash
# After official networks trained with the symmetry augmentation: fine-tunes, both validation instruments, candidate files.
#   pid_v2_newnets_sym.sh <family> [<more families already processed> ...]
# e.g. `pid_v2_newnets_sym.sh o3S` after the first four, `pid_v2_newnets_sym.sh o3BS o3S` after the bigger ones.
# Reference: v2y_new (METHOD_NOTES 2.61: m101x + px + o1 + px2 + py*2 + pb). Candidates (rule in METHOD_NOTES 2.62):
#   <tag>_add  the reference's members + the official symmetry networks at weight 2
#   <tag>_sym  symmetry-trained networks only (official ones + the two held-out ones)
#   <tag>_mid  symmetry networks at weight 2 + the plain official networks (o1, o2) + x2e
#   <tag>_hi   symmetry networks at weight 4 + the plain official networks only
#   <tag>_all  symmetry networks at weight 3 + every plain network except the fifteen M90 ones
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
fam=$1
shift
sym=$fam; for f in "$@"; do sym=$sym+$f; done
w() { echo "$1" | sed "s/+/*$2+/g; s/\$/*$2/"; }  # every family of a +list at weight k
o2=""; ls $R2/data/seq_ft/official_pub2_net-1_o2s*e_t50.npz > /dev/null 2>&1 && o2="+o2"
tag=v2s; [ "$fam" != o3S ] && tag=v2${fam#o3}
tag=$(echo "$tag" | tr 'A-Z' 'a-z')
bash $R2/scripts/nano4/pid_v2_newnets.sh "$fam" v2y_new \
  "${tag}_add=m101x+px+o1+px2+py*2+pb+$(w "$sym" 2)" \
  "${tag}_sym=$sym+py" \
  "${tag}_mid=$(w "$sym" 2)+py*2+o1$o2+px2" \
  "${tag}_hi=$(w "$sym" 4)+py*4+o1$o2" \
  "${tag}_all=$(w "$sym" 3)+py*3+o1$o2+px2+pb+px+m101x"
