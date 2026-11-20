#!/usr/bin/env python
"""A fusion document for pseudo-pool / old-pool READINGS whose new families have no real-pool vectors on /home yet
(METHOD_NOTES 2.201; /home has no room for them before a candidate has passed).

As data/day_scripts/build_v2h.py for families of my official networks (tags <family>s<k>e), with one difference: a family
named after the composition counts the members that have a linked checkpoint (/work/u2507115/pid_v2/seqnet/
ckpt_-1_<family>s<k>e.pt) instead of the members that have a real-pool vector file. The readings only use the file NAMES of
`seq_official` (scripts/pid_v2_offsim_par.py, scripts/pid_v2_oldsim.py); a submission file needs the real-pool vectors and
a document built by build_v2h.py.

  pid_build_doc_names.py <name> <family>[*k][+<family>[*k]...] [<new family> ...]
"""
import json
import re
import sys
from pathlib import Path

R, SN = Path("/home/u2507115/BK_AICUP_2026-2"), Path("/work/u2507115/pid_v2/seqnet")
name, spec, new = sys.argv[1], sys.argv[2].split("+"), set(sys.argv[3:])
ref = json.loads((R / "results/pid_fusion_v2ge.json").read_text())
F = lambda tag: [str(R / f"data/seq_ft/official_{pool}_net-1{tag}_t50.npz") for pool in ("pub2", "priv2")]  # noqa: E731
groups = {}
for g in {x.split("*")[0] for x in spec}:
    if not re.fullmatch(r"o\d\w*", g):
        sys.exit(f"{g}: only families of my official networks")
    if g in new:
        groups[g] = [F(f"_{g}s{k}e") for k in range(24) if (SN / f"ckpt_-1_{g}s{k}e.pt").exists()]
    else:
        groups[g] = [F(f"_{g}s{k}e") for k in range(24) if Path(F(f"_{g}s{k}e")[0]).exists()]
    if len(groups[g]) < 7:
        sys.exit(f"family {g}: only {len(groups[g])} members ({'linked checkpoints' if g in new else 'real-pool vectors'})")
files = [x for g in spec for _ in range(int(g.split("*")[1]) if "*" in g else 1) for x in groups[g.split("*")[0]]]
doc = {**ref, "seq_official": files, "arm": "ft50", "composition": spec,
       "seq_model": f"composition {'+'.join(spec)} ({len(files)} networks), each after 50 steps of the pool fine-tune; mean of the row-standardised matrices; weights of v2ge",
       "note": "pseudo-pool / old-pool reading only (scripts/pid_build_doc_names.py): the real-pool vector files of "
               + (" ".join(sorted(new & set(groups))) or "-") + " may not exist"}
(R / f"results/pid_fusion_{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
print(name, spec, "networks", len(files), "| members:", " ".join(f"{g}={len(v)}" for g, v in sorted(groups.items())))
