"""Label-free probes for structure others might exploit: SGF properties left in the official files, question-order vs
predicted-player-order correlation, and shared opening prefixes between question and candidate games (near-duplicates)."""
import collections, re, sys, time
sys.path.insert(0, "/home/u2507115/BK_AICUP_2026-2/src")
import aicup_pid  # noqa: E402  (puts AI_CUP_2026-1/src on sys.path)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from aicup.sgf import parse_moves
from aicup_pid.dedup import game_hash

R2 = "/home/u2507115/BK_AICUP_2026-2"
prop = re.compile(r"([A-Z]{1,3})\[")
hd = re.compile(r"HD\[(\d+)\]")
strict = re.compile(r"^\(;SZ\[\d+\]HD\[\d+\](;[BW]\[[a-t]{0,2}\])*\)$")
subs = {v: pd.read_csv(f"{R2}/results/{v}/submission_player.csv", dtype=str) for v in ("submission_f3", "submission_f3_assign")}
for pool, cf, qf in (("pub_", "player_identification_candidates_public.csv", "player_identification_test_public.csv"),
                     ("priv_", "player_identification_candidates_private.csv", "player_identification_test_private.csv")):
    t0 = time.time()
    cands = pd.read_csv(f"{R2}/data/tests/{cf}", dtype=str, keep_default_na=False)
    qs = pd.read_csv(f"{R2}/data/tests/{qf}", dtype=str, keep_default_na=False)
    qsgfs = [(r.question_id, r[f"sgf_{i}"]) for _, r in qs.iterrows() for i in range(1, int(r.num_games) + 1)]
    # (1) what is left in the SGF strings
    cnt, qcnt = collections.Counter(), collections.Counter()
    for s in cands.sgf_content: cnt.update(set(prop.findall(s)))
    for _, s in qsgfs: qcnt.update(set(prop.findall(s)))
    print(pool, "candidate SGF properties", dict(cnt))
    print(pool, "question SGF properties", dict(qcnt))
    print(pool, "HD candidates", dict(collections.Counter(hd.search(s).group(1) if hd.search(s) else "none" for s in cands.sgf_content)),
          "HD questions", dict(collections.Counter(hd.search(s).group(1) if hd.search(s) else "none" for _, s in qsgfs)))
    bad_c = [s for s in cands.sgf_content if not strict.match(s)]; bad_q = [s for _, s in qsgfs if not strict.match(s)]
    print(pool, "not move-only strings: candidates", len(bad_c), "questions", len(bad_q), "examples", [b[:120] for b in (bad_c + bad_q)[:3]])
    # (2) order leak: question index vs predicted player index
    ng = qs.set_index("question_id").num_games.astype(int)
    for v, sub in subs.items():
        s = sub[sub.question_id.str.startswith(pool)]
        qi = s.question_id.str.extract(r"(\d+)$")[0].astype(int).values; pi = s.top_1.str.extract(r"(\d+)$")[0].astype(int).values
        rho, p = spearmanr(qi, pi)
        print(pool, v, f"spearman(question index, predicted player index) = {rho:+.3f} (p={p:.2g}); equal index {int((qi == pi).sum())}/400;",
              "predicted player index by n:", {n: (int(pi[ng.loc[s.question_id].values == n].min()), int(pi[ng.loc[s.question_id].values == n].max()), round(float(pi[ng.loc[s.question_id].values == n].mean()), 1)) for n in (5, 10, 15, 20)})
    # candidate game order within a player: are game ids chronological-looking? report mean length by position in the player's 100
    # (3) shared opening prefixes / near-duplicates between question games and candidate games
    cm = [parse_moves(s) for s in cands.sgf_content]; qm = [parse_moves(s) for _, s in qsgfs]
    print(pool, "parsed", len(cm), "candidate and", len(qm), "question games in", round(time.time() - t0), "s;",
          "lengths: candidates median", int(np.median([len(m) for m in cm])), "questions median", int(np.median([len(m) for m in qm])))
    for k in (12, 20, 30, 40, 60, 100):
        ch = {}
        for j, m in enumerate(cm):
            if len(m) >= k: ch.setdefault(game_hash(m[:k]), []).append(j)
        hits = [(i, ch[game_hash(m[:k])]) for i, m in enumerate(qm) if len(m) >= k and game_hash(m[:k]) in ch]
        same_player_as_pred = 0
        if hits:
            # does the matching candidate belong to the player we predicted for that question?
            qid_of = [q for q, _ in qsgfs]; pred = subs["submission_f3"].set_index("question_id").top_1
            same_player_as_pred = sum(1 for i, js in hits if pred[qid_of[i]] in set(cands.player_id.iloc[js]))
        print(pool, f"question games sharing the first {k} moves with >=1 candidate game: {len(hits)}/{len(qm)}; "
              f"matching candidate owned by our predicted player: {same_player_as_pred}/{len(hits)}")
    # full-game containment for short question games (<= 80 moves): question game == prefix of a candidate game
    short = [(i, m) for i, m in enumerate(qm) if 0 < len(m) <= 80]
    lengths = sorted(set(len(m) for _, m in short))
    contained = 0
    for L in lengths:
        ch = set(game_hash(m[:L]) for m in cm if len(m) >= L)
        contained += sum(1 for _, m in short if len(m) == L and game_hash(m) in ch)
    print(pool, f"short question games (<=80 moves, {len(short)}) that equal the prefix of a candidate game: {contained}; {round(time.time() - t0)} s")
