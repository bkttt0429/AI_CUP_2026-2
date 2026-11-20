#!/usr/bin/env bash
# METHOD_NOTES 2.247: staging of the player-ID project for the move to the user's PC, as the rank project did
# (/work/u2507115/AICUP1_MOVE_20261121). HARD LINKS only: no file is copied, moved or deleted, and no quota is used.
#   /work/u2507115/AICUP2_MOVE_20261121/   files that live on /work   (+ MANIFEST_work.sha256, LEFT_OUT.txt, README.txt)
#   /home/u2507115/AICUP2_MOVE_20261121/   files that live on /home   (+ MANIFEST_home.sha256)
# Run again safely: existing links are kept, the manifests are rewritten. From anywhere; no GPU needed.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2 W=/work/u2507115/AICUP2_MOVE_20261121 H=/home/u2507115/AICUP2_MOVE_20261121 f t n rc=0
  mkdir -p $W/work/pid_v2/seqnet_final $W/work/pid_v2/seqnet_fallback_only $H/home/BK_AICUP_2026-2/data || return 1
  # 1. checkpoints: the networks of the final (document v2fin11_c) and those only the fallback v2fin7_a needs
  python3 -I - "$R2" "$P/seqnet" "$W/work/pid_v2" <<'PY' || return 1
import json, os, re, sys
R, SN, out = sys.argv[1:4]
def nets(d):
    return {re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", os.path.basename(x[0])).groups() for x in json.load(open(f"{R}/results/pid_fusion_{d}.json"))["seq_official"]}
fin, fb = nets("v2fin11_c"), nets("v2fin7_a")
done = 0
with open(f"{out}/SEQNET_LINKS.txt", "w") as fh:
    fh.write("# family member name (a symlink in seqnet on nano4) -> checkpoint file; folder it is staged in\n")
    for v, tg in sorted(fin | fb, key=lambda x: x[1]):
        name = f"ckpt_{v}{tg}.pt"
        real = os.path.realpath(f"{SN}/{name}")
        sub = "seqnet_final" if (v, tg) in fin else "seqnet_fallback_only"
        dst = f"{out}/{sub}/{os.path.basename(real)}"
        if not os.path.exists(dst):
            os.link(real, dst)
        done += 1
        fh.write(f"{name} -> {os.path.basename(real)}\t{sub}\n")
print(f"checkpoints staged: {done} (final {len(fin)}, fallback only {len(fb - fin)})")
PY
  # 2. the pools' per-ply inputs of the fine-tunes (on /work)
  for f in ply_official sup_pools; do [ -d $W/work/pid_v2/$f ] || cp -al $P/$f $W/work/pid_v2/$f || rc=1; done
  mkdir -p $W/work/pid_v2/supj && { [ -d $W/work/pid_v2/supj/pools ] || cp -al $P/supj/pools $W/work/pid_v2/supj/pools || rc=1; }
  # 3. /home: submission files, readings and documents, the pools' vectors and stores
  for f in upload results METHOD_NOTES.md; do [ -e $H/home/BK_AICUP_2026-2/$f ] || cp -al $R2/$f $H/home/BK_AICUP_2026-2/$f || rc=1; done
  for f in seq_ft trunk pid_v2_ply_original; do [ -e $H/home/BK_AICUP_2026-2/data/$f ] || cp -al $R2/data/$f $H/home/BK_AICUP_2026-2/data/$f || rc=1; done
  # symbolic links inside the staged trees point to places on nano4: list them
  find $W $H -type l -printf '%p -> %l\n' > $W/SYMLINKS_IN_STAGING.txt 2> /dev/null
  cat > $W/LEFT_OUT.txt <<'TXT'
Left on nano4, NOT staged (MB) - all of it can be made again from the code on GitHub, the competition files and the open
external data; none of it is needed to rebuild the final file from the staged checkpoints and vectors:
   31350  /work/u2507115/ext_fox                 external open games (Fox Go dataset) and their per-ply features (training input)
   12258  /work/u2507115/pid_v2/supj             supervised per-ply features of the TRAINING games (pools/ IS staged)
    1541  /work/u2507115/pid_v2/probep           projection, statistics and features of the supervised feature set
    3512  BK_AICUP_2026-2/data/pid_v2_f0tmp      pseudo-pool vectors of the verdicts (hold-out seeds 4-7)
    2448  BK_AICUP_2026-2/data/pid_v2_seq_ft_offsim   pseudo-pool vectors (hold-out seeds 0-3)
    1213  BK_AICUP_2026-2/data/pid_v2_seq_ft_oldsim   old-pool simulation vectors
    1108  /work/u2507115/pid_v2/vtmp8            temporary verdict vectors of 11-19 / 11-20
     710  /work/u2507115/pid_v2/orig_ft          vectors of the original-pool cross-check (METHOD_NOTES 2.241)
     634  /work/u2507115/pid_v2/ply_origq        per-ply vectors of the original pools' question games
     137  BK_AICUP_2026-2/data/tests_1111        the organisers' test files (the user has the originals)
          BK_AICUP_2026-2/.venv, data/ext_ply*, data/supj_ext, the rank project's trunk network and per-ply stores
          (results/trunk_vec* of BK_AICUP_2026-1: the rank project's own staging)
TXT
  (cd $W && find . -type f ! -name 'MANIFEST_work.sha256*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST_work.sha256.tmp && mv MANIFEST_work.sha256.tmp MANIFEST_work.sha256) || rc=1
  (cd $H && find . -type f ! -name 'MANIFEST_home.sha256*' -print0 | sort -z | xargs -0 sha256sum > MANIFEST_home.sha256.tmp && mv MANIFEST_home.sha256.tmp MANIFEST_home.sha256) || rc=1
  echo "staged on /work: $(find $W -type f | wc -l) files, $(du -sm $W | cut -f1) MB; on /home: $(find $H -type f | wc -l) files, $(du -sm $H | cut -f1) MB; symbolic links inside: $(wc -l < $W/SYMLINKS_IN_STAGING.txt)"
  for f in seqnet_final seqnet_fallback_only ply_official sup_pools supj; do echo "  $(du -sm $W/work/pid_v2/$f | cut -f1) MB  work/pid_v2/$f ($(find $W/work/pid_v2/$f -type f | wc -l) files)"; done
  for f in upload results data/seq_ft data/trunk data/pid_v2_ply_original; do echo "  $(du -sm $H/home/BK_AICUP_2026-2/$f | cut -f1) MB  home/BK_AICUP_2026-2/$f ($(find $H/home/BK_AICUP_2026-2/$f -type f | wc -l) files)"; done
  echo "done at $(date '+%F %T'), rc $rc"
  return $rc
}
main "$@"; exit $?
