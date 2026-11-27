#!/usr/bin/env bash
# After the eight bigger official networks (o3B): fine-tunes, both validation instruments and candidate files.
# Reference: the best composition known when this runs (v2k_eq if it exists, else v2j_eq, else v2i_eq).
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
ref=v2i_eq; base=m90+m101x+px+o1+px2
[ -s $R2/results/pid_fusion_v2j_eq.json ] && { ref=v2j_eq; base=$base+o2; }
[ -s $R2/results/pid_fusion_v2k_eq.json ] && { ref=v2k_eq; base=$base+o3; }
new=${base#m90+}
bash $R2/scripts/nano4/pid_v2_newnets.sh o3B $ref "v2b_eq=$base+o3B" "v2b_w=$base+o3B*2" "v2b_new=$new+o3B" "v2b_only=o3B"
