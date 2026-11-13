#!/usr/bin/env bash
# After official networks trained with the symmetry augmentation: fine-tunes, both validation instruments, candidate files.
#   pid_v2_newnets_sym.sh o3S     the four of the usual size        pid_v2_newnets_sym.sh o3BS   the four bigger ones (after o3S)
# Reference: the best composition known when this runs (v2k_eq if it exists, else v2j_eq, else v2i_eq).
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
fam=$1
ref=v2i_eq; base=m90+m101x+px+o1+px2
[ -s $R2/results/pid_fusion_v2j_eq.json ] && { ref=v2j_eq; base=$base+o2; }
[ -s $R2/results/pid_fusion_v2k_eq.json ] && { ref=v2k_eq; base=$base+o3; }
new=${base#m90+}
if [ "$fam" = o3S ]; then
  bash $R2/scripts/nano4/pid_v2_newnets.sh o3S $ref "v2s_eq=$base+o3S" "v2s_w=$base+o3S*3" "v2s_new=$new+o3S*2" "v2s_only=o3S"
else
  bash $R2/scripts/nano4/pid_v2_newnets.sh o3BS $ref "v2t_eq=$base+o3S+o3BS" "v2t_w=$base+o3S*3+o3BS*3" "v2t_new=$new+o3S*2+o3BS*2" "v2t_only=o3S+o3BS"
fi
