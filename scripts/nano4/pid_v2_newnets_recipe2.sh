#!/usr/bin/env bash
# Chain for NEW families of official networks, compared directly with the LIVE file's composition (v2sl_all = pid_v2sl_alls.csv,
# Public 0.9016519 since 2026-11-14 13:42): the steps of pid_v2_newnets2.sh (fine-tunes of the first family, check on the
# official candidates, pseudo-pools, old-pool simulation, files). Candidates:
#   <tag>_all   the live composition + every new family x 3 (the weight of its other symmetry families)
#   <tag>_h     the live composition + every new family x 6
#   <tag>_p     <tag>_all WITHOUT the first family (what that family adds); only when more than one family is given
# usage: [TAG=<tag>] pid_v2_newnets_recipe2.sh <new family> [new families already processed ...]
# Rule (METHOD_NOTES 2.104): a file is offered for upload when its pseudo-pool difference from the live composition is at
# least +0.002 and its old-pool simulation is not below the live composition's.
# Chains of different families may run at the same time (the vector files are written atomically, METHOD_NOTES 2.103).
# Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 fam=$1 tag=${TAG:-v2r} live="o3SL*3+o3S*3+o4S*3+o3BS*3+py*3+o1+o2+px2+pb+px+m101x"
  mult() { local k=$1 s="" g; shift; for g in "$@"; do s="$s+$g*$k"; done; echo "${s#+}"; }
  local cands=("${tag}_all=$(mult 3 "$@")+$live" "${tag}_h=$(mult 6 "$@")+$live")
  if [ $# -gt 1 ]; then cands+=("${tag}_p=$(mult 3 "${@:2}")+$live"); fi
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$fam" v2sl_all "${cands[@]}"
}
main "$@"; exit $?
