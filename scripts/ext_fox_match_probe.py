#!/usr/bin/env python
"""What a hit of the overlap gate is (METHOD_NOTES 2.50): for every competition training game (fold 0 excluded, test sets
never read) whose first --prefix moves equal an external game's up to a board symmetry, describe the pair WITHOUT any account:
which rank file and which external folder, both lengths, how many moves agree from the start (under the symmetry that aligns
the prefix) and whether the whole record is the same, the external record's year and the two ranks written in it, and how many
external records share that prefix. Account keys are not read."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_fox_ingest import SYMS, moves_of, prefix_hash  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", nargs="+", required=True, help="name=dir")
    ap.add_argument("--prefix", type=int, default=60)
    ap.add_argument("--aicup1", default=str(Path.home() / "BK_AICUP_2026-1"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ext = {}
    for spec in a.parts:
        name, d = spec.split("=", 1)
        for f in sorted(Path(d).glob("games_*.npz")):
            with np.load(f) as z:
                mv, off, ln, year, br, wr, ha = z["moves"], z["off"], z["len"], z["year"], z["br"], z["wr"], z["ha"]
            for i, (o, n) in enumerate(zip(off, ln)):
                if n >= a.prefix:
                    h = prefix_hash([(int(v) // 512, int(v) % 512) for v in mv[o:o + a.prefix]], a.prefix)
                    if h is not None:
                        ext.setdefault(h, []).append((name, [(int(v) // 512, int(v) % 512) for v in mv[o:o + n]], int(year[i]), int(br[i]), int(wr[i]), int(ha[i])))
        print(f"{name}: {len(ext)} distinct prefixes so far", flush=True)
    folds = pd.read_csv(Path(a.aicup1) / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
    out = []
    for f in sorted((Path(a.aicup1) / "data/training").glob("train_*.csv")):
        c = pd.read_csv(f, usecols=["player_id", "sgf_content"], dtype=str)
        c = c[c.player_id.map(folds).fillna(-1) != 0]
        for s in c.sgf_content:
            mv = moves_of(s.encode())
            h = prefix_hash(mv, a.prefix)
            if h is None or h not in ext:
                continue
            for name, em, year, br, wr, ha in ext[h]:
                best = 0
                for sym in SYMS:  # the symmetry that aligns the two records
                    t = [(c_, 361 if p == 361 else 19 * sym(p // 19, p % 19)[0] + sym(p // 19, p % 19)[1]) for c_, p in em]
                    k = 0
                    while k < min(len(t), len(mv)) and t[k] == mv[k]:
                        k += 1
                    best = max(best, k)
                out.append({"competition_file": f.name, "competition_moves": len(mv), "external_folder": name, "external_moves": len(em), "moves_equal_from_start": best,
                            "whole_record_equal": bool(best == len(mv) == len(em)), "external_year": year, "external_ranks_black_white": [br, wr], "external_handicap": ha,
                            "external_records_with_this_prefix": len(ext[h])})
                print(out[-1], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} pair(s); wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
