"""Structure of the 11-08 v2 release's added pools (pub2_ / priv2_): counts, colours, sizes, handicap, duplicates against the
10-30 pools and the training games. Usage: pid_inspect_v2.py <tests dir> <training dir>"""
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

T, TRAIN = Path(sys.argv[1]), Path(sys.argv[2])
h = lambda s: hashlib.md5(s.encode()).hexdigest()
moves = lambda s: re.findall(r";([BW])\[([a-s]{0,2})\]", s)
seen = {}
for tag in ("public", "private"):
    for kind in ("candidates", "candidates2"):
        c = pd.read_csv(T / f"player_identification_{kind}_{tag}.csv", dtype=str, keep_default_na=False)
        per = c.groupby("player_id").size()
        hd = Counter(re.search(r"HD\[(\d+)\]", s).group(1) if "HD[" in s else "none" for s in c.sgf_content)
        first = Counter(moves(s)[0][0] if moves(s) else "-" for s in c.sgf_content)
        ln = pd.Series([len(moves(s)) for s in c.sgf_content])
        print(f"{kind}_{tag}: {len(c)} games, {c.player_id.nunique()} players, games/player {per.min()}-{per.max()} (mean {per.mean():.1f}); colour {dict(Counter(c.color))}; "
              f"B games per player {c[c.color == 'B'].groupby('player_id').size().reindex(per.index, fill_value=0).describe()[['min', '50%', 'max']].to_dict()}; HD {dict(hd)}; "
              f"first mover mark {dict(first)}; moves median {ln.median():.0f}; ids {c.player_id.iloc[0]} {c.game_id.iloc[0]}; columns {list(c.columns)}", flush=True)
        if kind == "candidates2":
            hdc = Counter((col, re.search(r"HD\[(\d+)\]", s).group(1) if "HD[" in s else "none", moves(s)[0][0] if moves(s) else "-") for col, s in zip(c.color, c.sgf_content))
            print(f"  (player colour, HD, first mover mark): {dict(hdc)}")
        seen[f"{kind}_{tag}"] = {h(s) for s in c.sgf_content}
    for kind in ("test", "test2"):
        q = pd.read_csv(T / f"player_identification_{kind}_{tag}.csv", dtype=str, keep_default_na=False)
        n = q.num_games.astype(int)
        games, cols = [], []
        for _, r in q.iterrows():
            for i in range(1, int(r.num_games) + 1):
                games.append(r[f"sgf_{i}"]); cols.append(r[f"color_{i}"])
        mixed = sum(len({r[f"color_{i}"] for i in range(1, int(r.num_games) + 1)}) > 1 for _, r in q.iterrows())
        print(f"{kind}_{tag}: {len(q)} questions, num_games {dict(n.value_counts().sort_index())}; {len(games)} games; colour {dict(Counter(cols))}; questions with both colours {mixed}; "
              f"ids {q.question_id.iloc[0]}..{q.question_id.iloc[-1]}; columns {len(q.columns)}", flush=True)
        seen[f"{kind}_{tag}"] = {h(s) for s in games}
names = list(seen)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        k = len(seen[a] & seen[b])
        if k:
            print(f"identical records: {a} & {b}: {k}")
print("identical-record check between the eight files done")
tr = set()
for f in sorted(TRAIN.glob("train_*.csv")):
    tr |= {h(s) for s in pd.read_csv(f, usecols=["sgf_content"], dtype=str).sgf_content}
print(f"training records {len(tr)}; " + "; ".join(f"{k} in training: {len(v & tr)}" for k, v in seen.items()))
s = pd.read_csv(T / "sample_submission_player_v2.csv", dtype=str)
print("sample:", len(s), list(s.columns), dict(Counter(q.split('_')[0] for q in s.question_id)), s.iloc[0].tolist(), s.iloc[-1].tolist())
