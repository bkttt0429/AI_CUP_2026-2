#!/usr/bin/env bash
# Chain for families of official networks trained with a NEW recipe (opening dropout, ply masking, ...): the same steps as
# pid_v2_newnets2.sh (fine-tunes of the first family, check on the official candidates, pseudo-pools, old-pool simulation,
# files). Candidates (METHOD_NOTES 2.87, revised 2.92 after the first reading: a higher share of the new networks is WORSE,
# an even mix of every kind is best):
#   <tag>_all   every family x 3 (the new ones, the 40,000-step o3SL, o3BS, o3S, o4S), py x 3, the plain networks
#   <tag>_h     new families x 4, o3SL x 4, the 20,000-step symmetry families x 2, py x 2, o1, o2
#   <tag>_x     symmetry-augmented networks only: new families x 2, o3SL x 2, the others once, py
#   <tag>_p     <tag>_all WITHOUT the first family (what that family adds); only when more than one family is given
# usage: [TAG=<tag>] pid_v2_newnets_recipe.sh <new family> [new-recipe families already processed ...]
# Reference: v2w_n8 (the live file's composition), as every chain of 2026-11-14. Never run two chains at the same time.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 fam=$1 tag=${TAG:-v2r} plain="o1+o2+px2+pb+px+m101x"
  mult() { local k=$1 s="" g; shift; for g in "$@"; do s="$s+$g*$k"; done; echo "${s#+}"; }
  local cands=("${tag}_all=$(mult 3 "$@" o3SL o3BS o3S o4S)+py*3+$plain"
               "${tag}_h=$(mult 4 "$@" o3SL)+$(mult 2 o3BS o3S o4S)+py*2+o1+o2"
               "${tag}_x=$(mult 2 "$@" o3SL)+o3BS+o3S+o4S+py")
  if [ $# -gt 1 ]; then cands+=("${tag}_p=$(mult 3 "${@:2}" o3SL o3BS o3S o4S)+py*3+$plain"); fi
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$fam" v2w_n8 "${cands[@]}"
}
main "$@"; exit $?
