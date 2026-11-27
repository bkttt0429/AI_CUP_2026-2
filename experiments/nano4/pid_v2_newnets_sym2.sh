#!/usr/bin/env bash
# After official networks trained with the symmetry augmentation (parallel chain, METHOD_NOTES 2.67):
#   [TAG=<prefix of the candidate names>] pid_v2_newnets_sym2.sh <family> [<symmetry families already processed> ...]
# Reference: v2w_n8 (the file pid_v2w_n8s.csv). Candidates, by the share of the symmetry-trained networks:
#   <tag>_add  the reference's members + the official symmetry networks at weight 4
#   <tag>_mid  symmetry networks at weight 2 + every plain network except the fifteen M90 ones
#   <tag>_all  symmetry networks at weight 3 + the same plain networks
#   <tag>_hi   symmetry networks at weight 4 + the plain official networks only (o1, o2)
#   <tag>_sym  symmetry-trained networks only
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
fam=$1
shift
sym=$fam; for f in "$@"; do sym=$sym+$f; done
w() { echo "$1" | sed "s/+/*$2+/g; s/\$/*$2/"; }  # every family of a +list at weight k
o2=""; ls $R2/data/seq_ft/official_pub2_net-1_o2s*e_t50.npz > /dev/null 2>&1 && ls /work/u2507115/pid_v2/seq_ft_offsim/offsim_pub2_h3s1_net-1_o2s*e_t50.npz > /dev/null 2>&1 && ls /work/u2507115/pid_v2/seq_ft_oldsim/oldsim_priv_s0_net-1_o2s*e_t50.npz > /dev/null 2>&1 && o2="+o2"
tag=${TAG:-v2$(echo "${fam#o3}" | tr 'A-Z' 'a-z')}
plain=o1$o2+px2+pb+px+m101x
OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$fam" v2w_n8 \
  "${tag}_add=m101x+px+o1+px2+pb+py*8+$(w "$sym" 4)" \
  "${tag}_mid=$(w "$sym" 2)+py*2+$plain" \
  "${tag}_all=$(w "$sym" 3)+py*3+$plain" \
  "${tag}_hi=$(w "$sym" 4)+py*4+o1$o2" \
  "${tag}_sym=$sym+py"
