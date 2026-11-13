#!/usr/bin/env bash
# The o3 families after job p_seqoff3 (scripts/nano4/pid_exp_official8.sh): o3 (as before), o3hm and o3hb (rank-matched
# episodes). Each step is scripts/nano4/pid_v2_newnets.sh; the last check holds every family side by side.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
N=$R2/scripts/nano4/pid_v2_newnets.sh
base=m90+m101x+px+o1+px2+o2
bash $N o3 v2i_eq,v2j_eq "v2k_eq=$base+o3" "v2k_new=m101x+px+o1+px2+o2+o3" || echo "o3 step failed"
bash $N o3hm v2k_eq "v2m_hm=$base+o3+o3hm" || echo "o3hm step failed"
bash $N o3hb v2k_eq "v2m_hb=$base+o3+o3hb" "v2m_all=$base+o3+o3hm+o3hb" "v2m_new=m101x+px+o1+px2+o2+o3+o3hm+o3hb" || echo "o3hb step failed"
