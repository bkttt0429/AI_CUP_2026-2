#!/usr/bin/env bash
# Chain for a family of official networks trained with a NEW recipe (opening dropout, ply masking, ...): the same steps as
# pid_v2_newnets2.sh (fine-tunes, check on the official candidates, pseudo-pools, old-pool simulation, files), with candidates
# that give the new-recipe networks an increasing share next to the 20 networks of the plain symmetry recipe (METHOD_NOTES 2.87):
#   base        o3BS*2 + o3S*2 + o4S*2 + py*2 + o1 + o2       (the make-up of the current candidate, 49 entries)
#   <tag>_h2    new families x 4  + base        <tag>_h5   new families x 10 + base        <tag>_h10  new families x 20 + base
#   <tag>_new   the new families alone
# usage: [TAG=<tag>] pid_v2_newnets_recipe.sh <new family> [new-recipe families already processed ...]
# Reference: v2w_n8 (the live file's composition), as every chain of 2026-11-14. Never run two chains at the same time.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 fam=$1 tag=${TAG:-v2r} base="o3BS*2+o3S*2+o4S*2+py*2+o1+o2" f
  mult() { local k=$1 s="" g; shift; for g in "$@"; do s="$s+$g*$k"; done; echo "${s#+}"; }
  local alone=""
  for f in "$@"; do alone="$alone+$f"; done
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$fam" v2w_n8 \
    "${tag}_h2=$(mult 4 "$@")+$base" "${tag}_h5=$(mult 10 "$@")+$base" "${tag}_h10=$(mult 20 "$@")+$base" "${tag}_new=${alone#+}"
}
main "$@"; exit $?
