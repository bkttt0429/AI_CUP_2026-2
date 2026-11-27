#!/usr/bin/env python
"""Compare two releases of the official player-identification files (e.g. data/tests vs data/tests_<date>).

Answers, per pool: did the candidate library change (same games? same players? same row order?), did the questions
change (same ids? same games? same n mix?), and what kind of change it is: colours other than Black, HD values, board
orientation (first-move corner mix and raw first-move concentration per candidate, the habit that the orientation /
raw-opening / trunk components rely on), game lengths, candidates per question structure. Whole-game hashes are
symmetry-canonical (aicup_pid.dedup.game_hash), so a rotated or mirrored copy of an old game still matches.

  python -I scripts/pid_compare_releases.py --old data/tests --new data/tests_20261108 --out results/checks/release_compare.json
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402,F401
from aicup.sgf import parse_moves  # noqa: E402
from aicup_pid.dedup import game_hash  # noqa: E402

POOLS = (("public", "player_identification_candidates_public.csv", "player_identification_test_public.csv"),
         ("private", "player_identification_candidates_private.csv", "player_identification_test_private.csv"))
HD = re.compile(r"HD\[(\d+)\]")
FIRST_B = re.compile(r";B\[([a-s]{2})\]")


def corner(pt):
    x, y = ord(pt[0]) - 97, ord(pt[1]) - 97
    return ("L" if x < 9 else "C" if x == 9 else "R") + ("T" if y < 9 else "M" if y == 9 else "B")


def load(d, c_name, q_name):
    c = pd.read_csv(Path(d) / c_name, dtype=str, keep_default_na=False)
    q = pd.read_csv(Path(d) / q_name, dtype=str, keep_default_na=False)
    games = []
    for _, r in q.iterrows():
        n = int(r["num_games"])
        for i in range(1, n + 1):
            games.append((r["question_id"], r[f"sgf_{i}"], r.get(f"color_{i}", "B")))
    return c, q, games


def profile(c, q, qgames):
    hs = lambda s: game_hash(parse_moves(s))
    first = c.sgf_content.map(lambda s: (FIRST_B.search(s) or [None, None])[1])
    modal = c.assign(f=first).groupby("player_id").f.agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0.0)
    qfirst = [m.group(1) for _, s, _ in qgames if (m := FIRST_B.search(s))]
    lens = [len(parse_moves(s)) for s in c.sgf_content.head(2000)]
    return {
        "candidates": int(c.player_id.nunique()), "candidate_games": int(len(c)),
        "games_per_candidate": dict(collections.Counter(c.groupby("player_id").size().tolist()).most_common(5)),
        "questions": int(len(q)), "n_mix": dict(sorted(collections.Counter(q.num_games.astype(int)).items())),
        "colours_candidates": dict(collections.Counter(c.color)), "colours_questions": dict(collections.Counter(col for *_, col in qgames)),
        "hd_candidates": dict(collections.Counter((HD.search(s) or [None, "none"])[1] for s in c.sgf_content)),
        "hd_questions": dict(collections.Counter((HD.search(s) or [None, "none"])[1] for _, s, _ in qgames)),
        "first_corner_candidates": dict(collections.Counter(first.dropna().map(corner)).most_common(6)),
        "first_corner_questions": dict(collections.Counter(map(corner, qfirst)).most_common(6)),
        "first_move_modal_share_median": float(modal.median()), "first_move_modal_share_ge_0.9": int((modal >= 0.9).sum()),
        "length_quantiles_candidates_first2000": [int(x) for x in np.quantile(lens, [0.1, 0.5, 0.9])],
        "_hash_c": [hs(s) for s in c.sgf_content], "_hash_q": [hs(s) for _, s, _ in qgames],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {}
    for name, c_name, q_name in POOLS:
        oc, oq, og = load(a.old, c_name, q_name)
        nc, nq, ng = load(a.new, c_name, q_name)
        po, pn = profile(oc, oq, og), profile(nc, nq, ng)
        hc_o, hc_n, hq_o, hq_n = set(po.pop("_hash_c")), pn.pop("_hash_c"), set(po.pop("_hash_q")), pn.pop("_hash_q")
        same_rows = len(oc) == len(nc) and bool((oc.sgf_content.values == nc.sgf_content.values).all())
        cmp = {
            "candidate_files_identical": same_rows,
            "new_candidate_games_seen_in_old_candidates": int(sum(h in hc_o for h in hc_n)),
            "new_candidate_games_seen_in_old_questions": int(sum(h in hq_o for h in hc_n)),
            "new_question_games_seen_in_old_candidates": int(sum(h in hc_o for h in hq_n)),
            "new_question_games_seen_in_old_questions": int(sum(h in hq_o for h in hq_n)),
            "same_candidate_ids": sorted(set(oc.player_id)) == sorted(set(nc.player_id)),
            "same_question_ids": list(oq.question_id) == list(nq.question_id),
        }
        out[name] = {"old": po, "new": pn, "compare": cmp}
        print(f"== {name}")
        for k, v in cmp.items():
            print(f"  {k}: {v}")
        for k in pn:
            if po[k] != pn[k]:
                print(f"  CHANGED {k}: old {po[k]}  ->  new {pn[k]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
