#!/usr/bin/env python
"""Census of the external accounts, and the second side of the games already selected (METHOD_NOTES 2.73).

Input: the compact parts of scripts/ext_fox_ingest.py for every rank folder used so far (name=dir ...) and every select.csv
written so far. An account is a salted hash; no name exists in these files, and nothing here looks one up.
  1. eligible games per account, counted over all the folders together (the rules of ext_fox_select.py: no handicap, at least
     --min-moves moves, no pass inside the first 256 plies; the --cap-pct busiest accounts set aside): how many accounts have
     at least k games, for all accounts and for those in no selection
  2. the other player of every selected game (those games have per-ply vectors already). <out>/second.csv gets a row
     (game_key, account, color) when the other player is a selected account, or an account that is the other player of at
     least --min-second selected games: the same stored game then serves a second player
  3. what a selection of the accounts with few games would cost: for the accounts in no selection with at least k eligible
     games (k = 2 .. 8), the number of games needed to give each of them k games when a game can serve both of its players
     (greedy, both sides counted), next to the number needed with one subject per game
Output: <out>/census.json, <out>/second.csv.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_fox_select import T_MAX, load  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", nargs="+", required=True, help="name=dir of one rank folder's parts")
    ap.add_argument("--selections", nargs="+", required=True, help="every select.csv so far")
    ap.add_argument("--min-moves", type=int, default=80)
    ap.add_argument("--cap-pct", type=float, default=0.5)
    ap.add_argument("--min-second", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10 ** 9)
    sel = {}  # game_key -> (account, colour) of its subject
    for f in a.selections:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                sel[r["game_key"]] = (int(r["account"], 16), r["color"])
    sel_acc = np.array(sorted({v[0] for v in sel.values()}), dtype=np.uint64)
    print(f"{len(sel)} selected games of {len(sel_acc)} accounts in {len(a.selections)} selections ({time.time() - t0:.0f} s)", flush=True)

    PB, PW, OK, second = [], [], [], {}  # second: game_key -> (other account, its colour)
    n_games = 0
    for spec in a.parts:
        name, d = spec.split("=", 1)
        for z in load(d):
            ok = (z["ha"] <= 0) & (z["len"] >= a.min_moves)
            is_pass = (z["moves"] % 512) == 361
            cs = np.concatenate([[0], np.cumsum(is_pass)])
            ok &= (cs[z["off"] + np.minimum(z["len"], T_MAX)] - cs[z["off"]]) == 0
            PB.append(z["pb"].astype(np.uint64)); PW.append(z["pw"].astype(np.uint64)); OK.append(ok)
            n_games += len(ok)
            for i, k in enumerate(z["key"]):
                key = f"x:{name}:{str(k).rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
                s = sel.get(key)
                if s is not None:
                    b, w = int(z["pb"][i]), int(z["pw"][i])
                    if s[1] == "B" and b == s[0]:
                        second[key] = (w, "W")
                    elif s[1] == "W" and w == s[0]:
                        second[key] = (b, "B")
        print(f"  {name}: read, {n_games} games so far, {len(second)} selected games found ({time.time() - t0:.0f} s)", flush=True)
    pb, pw, ok = np.concatenate(PB), np.concatenate(PW), np.concatenate(OK)
    del PB, PW, OK
    idx = np.flatnonzero(ok)
    acc, cnt = np.unique(np.concatenate([pb[idx], pw[idx]]), return_counts=True)
    cap = float(np.percentile(cnt, 100 - a.cap_pct))
    free = ~np.isin(acc, sel_acc) & (cnt <= cap)
    ks = list(range(1, 21)) + [30, 50, 100]
    rep = {"games_in_parts": int(n_games), "eligible_games": int(len(idx)), "accounts_with_eligible_games": int(len(acc)), "busiest_cut_at_games": cap,
           "selected_accounts": int(len(sel_acc)), "selected_games": len(sel), "selected_games_found": len(second),
           "accounts_with_at_least": {str(k): int((cnt >= k).sum()) for k in ks},
           "unselected_accounts_with_at_least": {str(k): int((free & (cnt >= k)).sum()) for k in ks}}
    print(f"{n_games} games, {len(idx)} eligible; accounts {len(acc)}; busiest {a.cap_pct} % above {cap:.0f} games", flush=True)
    print("accounts with at least k eligible games (all / in no selection): " + "  ".join(f"{k}: {rep['accounts_with_at_least'][str(k)]} / {rep['unselected_accounts_with_at_least'][str(k)]}" for k in ks), flush=True)

    # 2. the second side of the selected games
    other = np.array([v[0] for v in second.values()], dtype=np.uint64)
    is_sel = np.isin(other, sel_acc)
    u, c = np.unique(other[~is_sel], return_counts=True)
    new_acc = u[c >= a.min_second]
    keep = is_sel | np.isin(other, new_acc)
    views = dict(zip(*np.unique(np.array([v[0] for v in sel.values()], dtype=np.uint64), return_counts=True)))
    extra = dict(zip(*np.unique(other[is_sel], return_counts=True)))
    before = np.array([views[x] for x in sel_acc.tolist()])
    after = before + np.array([extra.get(x, 0) for x in sel_acc.tolist()])
    with open(out / "second.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_key", "account", "color"])
        for (key, (o, col)), kp in zip(second.items(), keep):
            if kp:
                w.writerow([key, f"{o:016x}", col])
    rep["second_side"] = {"other_player_is_selected": int(is_sel.sum()), "share": float(is_sel.mean()), "other_accounts_not_selected": int(len(u)),
                          f"new_accounts_with_{a.min_second}_or_more": int(len(new_acc)), "rows_written": int(keep.sum()),
                          "games_per_selected_account_before": {"mean": float(before.mean()), "quantiles_10_50_90": [float(x) for x in np.percentile(before, [10, 50, 90])]},
                          "games_per_selected_account_after": {"mean": float(after.mean()), "quantiles_10_50_90": [float(x) for x in np.percentile(after, [10, 50, 90])]}}
    s2 = rep["second_side"]
    print(f"second side: the other player is a selected account in {s2['other_player_is_selected']} of {len(second)} selected games ({s2['share']:.1%}); games per selected account "
          f"{s2['games_per_selected_account_before']['mean']:.1f} -> {s2['games_per_selected_account_after']['mean']:.1f}; other accounts with >= {a.min_second} such games (new players) {len(new_acc)}; "
          f"{s2['rows_written']} rows in second.csv ({time.time() - t0:.0f} s)", flush=True)

    # 3. the cost of selecting the accounts with few games
    rep["few_games"] = {}
    order = rng.permutation(idx)
    for k in (2, 3, 4, 5, 6, 8):
        target = acc[free & (cnt >= k)]
        if not len(target):
            continue
        need = dict.fromkeys(target.tolist(), k)
        b_in, w_in = np.isin(pb[order], target), np.isin(pw[order], target)
        both = b_in & w_in
        used_two = used_one = 0
        for mask in (both, (b_in | w_in) & ~both):  # first the games whose two players are both wanted, then the rest
            for g in order[mask].tolist():
                b, w_ = int(pb[g]), int(pw[g])
                nb, nw = need.get(b, 0), need.get(w_, 0)
                if nb <= 0 and nw <= 0:
                    continue
                if nb > 0:
                    need[b] -= 1
                if nw > 0:
                    need[w_] -= 1
                if nb > 0 and nw > 0:
                    used_two += 1
                else:
                    used_one += 1
        short = sum(1 for v in need.values() if v > 0)
        rep["few_games"][str(k)] = {"accounts": int(len(target)), "games_both_sides": int(used_two + used_one), "of_which_serve_two": int(used_two), "games_one_subject_each": int(len(target) * k),
                                    "accounts_left_short": int(short), "eligible_games_with_both_players_wanted": int(both.sum())}
        r = rep["few_games"][str(k)]
        print(f"accounts in no selection with >= {k} games: {r['accounts']}; {k} games each = {r['games_one_subject_each']} with one subject per game, {r['games_both_sides']} games when a game serves both "
              f"players ({r['of_which_serve_two']} serve two); accounts left short {r['accounts_left_short']} ({time.time() - t0:.0f} s)", flush=True)
    (out / "census.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"wrote {out / 'census.json'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
