#!/usr/bin/env python
"""External players for the identity metric (METHOD_NOTES 2.46): which accounts and which of their games.
Input: the compact parts of scripts/ext_fox_ingest.py (one directory per rank folder of the Fox Go Dataset). An account key
is a salted hash; no name exists in these files.
  eligible games   19x19 (ingest), no handicap, at least --min-moves moves, no pass inside the first 256 plies
  accounts         at least --need eligible games in the folder and not among the --cap-pct busiest (bots, shared accounts);
                   --accounts of them per folder, drawn at random
  games            --games per account, each game given to ONE of its chosen accounts (the one still missing more), so a
                   record is extracted once and has one subject, as a competition record has
Output <out>/select.csv: game_key (x:<folder>:<record>), account (hex key), color (the subject's), sgf_content in the
competition's own form "(;SZ[19]HD[0];B[..];W[..]...)" cut after 256 plies; <out>/select.json: the counts.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_fox_ingest import moves_of  # noqa: E402

T_MAX = 256
ABC = "abcdefghijklmnopqrs"


def sgf_of(codes):
    return "(;SZ[19]HD[0]" + "".join(f";{'BW'[c // 512]}[{ABC[(c % 512) // 19]}{ABC[(c % 512) % 19]}]" for c in codes) + ")"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", nargs="+", required=True, help="name=dir of one rank folder's parts, e.g. 6d=/work/.../parts_6d")
    ap.add_argument("--accounts", type=int, default=7500, help="per folder")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--need", type=int, default=30)
    ap.add_argument("--min-moves", type=int, default=80)
    ap.add_argument("--cap-pct", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, n_rows = {"args": vars(a), "folders": {}}, 0
    with open(out / "select.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_key", "account", "color", "sgf_content"])
        for spec in a.parts:
            name, d = spec.split("=", 1)
            Z = []
            for f in sorted(Path(d).glob("games_*.npz")):  # the arrays in memory: an .npz re-reads an array at every access
                with np.load(f) as z:
                    Z.append({k: z[k] for k in ("pb", "pw", "ha", "len", "br", "wr", "year", "moves", "off", "key")})
            pb, pw = np.concatenate([z["pb"] for z in Z]), np.concatenate([z["pw"] for z in Z])
            ha, ln = np.concatenate([z["ha"] for z in Z]), np.concatenate([z["len"] for z in Z])
            br, wr, year = np.concatenate([z["br"] for z in Z]), np.concatenate([z["wr"] for z in Z]), np.concatenate([z["year"] for z in Z])
            part = np.concatenate([np.full(len(z["len"]), i) for i, z in enumerate(Z)])
            local = np.concatenate([np.arange(len(z["len"])) for z in Z])
            ok = (ha <= 0) & (ln >= a.min_moves)
            for i, z in enumerate(Z):  # no pass inside the plies that are used
                mv, off = z["moves"], z["off"]
                is_pass = (mv % 512) == 361
                cs = np.concatenate([[0], np.cumsum(is_pass)])
                m = part == i
                end = off + np.minimum(z["len"], T_MAX)
                ok[m] &= (cs[end] - cs[off]) == 0
            idx = np.flatnonzero(ok)
            acc, cnt = np.unique(np.concatenate([pb[idx], pw[idx]]), return_counts=True)
            cap = np.percentile(cnt, 100 - a.cap_pct)
            pool = acc[(cnt >= a.need) & (cnt <= cap)]
            chosen = rng.choice(pool, min(a.accounts, len(pool)), replace=False)
            need = dict.fromkeys(chosen.tolist(), a.games)
            got = dict.fromkeys(chosen.tolist(), 0)
            rows, ranks, years = 0, [], []
            for g in rng.permutation(idx):
                b, wkey = int(pb[g]), int(pw[g])
                nb, nw = need.get(b, 0), need.get(wkey, 0)
                if nb <= 0 and nw <= 0:
                    continue
                side = 0 if nb >= nw else 1
                who = b if side == 0 else wkey
                need[who] -= 1
                got[who] += 1
                z = Z[part[g]]
                o = int(z["off"][local[g]])
                codes = z["moves"][o:o + min(int(ln[g]), T_MAX)].tolist()
                text = sgf_of(codes)
                if rows < 200:  # the written record reads back as the same moves (the overlap gate and the replay parse this form)
                    assert [c * 512 + q for c, q in moves_of(text.encode())] == codes, "record does not read back"
                w.writerow([f"x:{name}:{Path(str(z['key'][local[g]])).stem}", f"{who:016x}", "BW"[side], text])
                rows += 1
                ranks.append(int(br[g] if side == 0 else wr[g])); years.append(int(year[g]))
            g_ = np.array(list(got.values()))
            summary["folders"][name] = {"games_in_parts": int(len(ok)), "eligible_games": int(len(idx)), "accounts_with_eligible_games": int(len(acc)), "accounts_meeting_need": int(len(pool)),
                                        "busiest_cut_at_games": float(cap), "accounts_chosen": int(len(chosen)), "games_written": rows, "accounts_with_8_or_more": int((g_ >= 8).sum()),
                                        "accounts_with_all_games": int((g_ >= a.games).sum()), "subject_rank_counts": {str(k): int(v) for k, v in zip(*np.unique(ranks, return_counts=True))},
                                        "year_counts": {str(k): int(v) for k, v in zip(*np.unique(years, return_counts=True))}}
            n_rows += rows
            s = summary["folders"][name]
            print(f"{name}: {s['games_in_parts']} games, {s['eligible_games']} eligible; accounts {s['accounts_with_eligible_games']}, with >= {a.need} games {s['accounts_meeting_need']}; chosen {s['accounts_chosen']} "
                  f"-> {rows} games ({s['accounts_with_all_games']} accounts with all {a.games}, {s['accounts_with_8_or_more']} with >= 8); subject ranks {s['subject_rank_counts']} ({time.time() - t0:.0f} s)", flush=True)
    summary["games_written"] = n_rows
    (out / "select.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {out / 'select.csv'}: {n_rows} games ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
