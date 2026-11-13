#!/usr/bin/env python
"""Move-level matching instead of one vector per game: a zero-shot probe (METHOD_NOTES 2.56, atom 3).

The present score compares ONE vector per game (the network's [CLS] output) with the candidate's prototype. What is averaged
away there: "in the situation of this very move, what did the candidate do in his own games?". Retrieval systems keep the
tokens and match them one by one (late interaction). Here, with no training: every own move of the subject is a token, and a
question game is scored against a candidate by how well its tokens are matched by the tokens of the candidate's ten games.
  token kinds   enc  the network's contextual state at the move (last encoder layer)
                gl   the stored input of the move (board-pooled + played-point trunk vectors, standardised): position and
                     move, no identity training at all
  scores        maxsim  mean over the question's tokens of the best cosine among the candidate's tokens
                near    the same, the candidate's tokens restricted to plies within +-8 of the question token's ply
                vote    every question token votes for the owners of its 20 nearest candidate tokens (cosine weights)
Simulated pools of the training folds (pid_v2_sim2.py), one held-out network per fold, no pool fine-tune; every question game
is a 1-game query, and the questions as given (1 / 2 / 3 / 5 games). Reported: each score alone, and added to the pooled score
(both row-standardised; the weight chosen on the OTHER fold's pools), against the pooled score alone, with a bootstrap over
players. Run from the AI_CUP_2026-1 checkout."""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AICUP1 = Path.cwd()
R2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AICUP1 / "src"), str(AICUP1 / "logs/nano4"), str(R2 / "src"), str(R2 / "scripts")]
import m88_seqid as m88  # noqa: E402
import pid_v2_seq_finetune as FT  # noqa: E402

WEIGHTS = (0.1, 0.2, 0.35, 0.5, 1.0)
SIZES = (1, 2, 3, 5)
MAXTOK = 128  # own moves kept per game (the network reads 256 plies)


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def per_question(S, truth):
    rank = (S > S[np.arange(len(truth)), truth][:, None]).sum(1)
    return np.where(rank < 5, np.exp(-rank.astype(float)), 0.0)


@torch.no_grad()
def tokens(net, A, rows, batch=512):
    """Per game: pooled vector [128]; per own move of the subject: encoder state [D] and standardised input [128]; ply index."""
    grab = {}
    orig = net.enc.forward
    net.enc.forward = lambda x, **kw: grab.__setitem__("H", orig(x, **kw)) or grab["H"]
    net.eval()
    pooled, enc, gl, ply, game = [], [], [], [], []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        g = torch.as_tensor(A["GL"][r]).cuda()
        mov = torch.as_tensor(A["MOV"][r].astype(np.int64)).cuda()
        ln = torch.as_tensor(A["LEN"][r].astype(np.int64)).cuda()
        sj = torch.as_tensor(A["SUBJ"][r].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(g, torch.as_tensor(A["POS"][r].astype(np.int64)).cuda(), mov, ln, sj)
        H = grab["H"][:, 1:].float()  # [B, T, D], position 0 is [CLS]
        T = H.shape[1]
        own = (mov == sj[:, None]) & (torch.arange(T, device=H.device)[None] < ln[:, None])
        b, t = own.nonzero(as_tuple=True)
        keep = (own.cumsum(1)[b, t] <= MAXTOK)
        b, t = b[keep], t[keep]
        pooled.append(z.float().cpu())
        enc.append(torch.nn.functional.normalize(H[b, t], dim=-1).half().cpu())
        gl.append(torch.nn.functional.normalize((g[b, t].float() - net.mu) / net.sd, dim=-1).half().cpu())
        ply.append(t.cpu())
        game.append((b + i).cpu())
    net.enc.forward = orig
    return torch.cat(pooled).numpy(), torch.cat(enc), torch.cat(gl), torch.cat(ply), torch.cat(game)


@torch.no_grad()
def match(q_tok, q_ply, q_game, n_q, c_tok, c_ply, c_owner, C, chunk=768):
    """[n_q games, C] for maxsim / near / vote. Candidate tokens are laid out in a padded [C, L] grid for the grouped maximum."""
    cnt = torch.bincount(c_owner, minlength=C)
    L = int(cnt.max())
    order = torch.argsort(c_owner, stable=True)
    slot = torch.arange(len(order)) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    grid = torch.zeros(C * L, c_tok.shape[1], dtype=torch.half)
    gply = torch.full((C * L,), -10 ** 6, dtype=torch.long)
    gown = torch.arange(C).repeat_interleave(L)
    idx = c_owner[order] * L + slot
    grid[idx], gply[idx] = c_tok[order], c_ply[order]
    valid = torch.zeros(C * L, dtype=torch.bool)
    valid[idx] = True
    grid, gply, gown, valid = grid.cuda(), gply.cuda(), gown.cuda(), valid.cuda()
    out = {k: torch.zeros(n_q, C, device="cuda") for k in ("maxsim", "near", "vote")}
    ntok = torch.bincount(q_game, minlength=n_q).clamp(min=1).cuda()
    for i in range(0, len(q_tok), chunk):
        qt, qp, qg = q_tok[i:i + chunk].cuda(), q_ply[i:i + chunk].cuda(), q_game[i:i + chunk].cuda()
        S = (qt @ grid.T).float()
        S[:, ~valid] = -1.0
        out["maxsim"].index_add_(0, qg, S.view(len(qt), C, L).max(-1).values)
        Sn = S.masked_fill((gply[None] - qp[:, None]).abs() > 8, -1.0)
        out["near"].index_add_(0, qg, Sn.view(len(qt), C, L).max(-1).values)
        top = S.topk(20, dim=1)
        v = torch.zeros(len(qt), C, device="cuda").scatter_add_(1, gown[top.indices], top.values.clamp(min=0))
        out["vote"].index_add_(0, qg, v)
    return {k: (x / ntok[:, None]).cpu().numpy() for k, x in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--ckpt-dir", default="/work/u2507115/pid_v2/seqnet")
    ap.add_argument("--ckpt-suffix", default="_x2e")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import aicup_pid
    from aicup.ladder import to_json
    from aicup_pid import v2
    from aicup_pid.paths import FOLDS_CSV
    from pid_v2_sim2 import build
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")
    rows = v2.load_side_rows(AICUP1)
    runs = []
    for v in a.folds:
        z = np.load(R2 / "data" / "seq" / f"fold{v}.npz")
        g = games[(games.fold == v) & games.game_key.isin(rows.keys()) & games.game_key.isin(set(z["key"].astype(str)))]
        net = FT.load_net(Path(a.ckpt_dir) / f"ckpt_{v}{a.ckpt_suffix}.pt")
        for s in a.seeds:
            cands, questions, owner, _block, _sgfs, _info = build(g, s)
            players = sorted(cands)
            pidx = {p: i for i, p in enumerate(players)}
            need = {k for ks in cands.values() for k, _ in ks} | {k for q in questions for k, _ in q}
            files = [f for d in m88.DIRS for f in sorted((d / f"fold{v}").glob("games_s*_p*.csv"))]
            parts = {x: [] for x in ("key", "GL", "POS", "MOV", "LEN", "SUBJ")}
            with mp.get_context("fork").Pool(12) as pool:
                for i in range(0, len(files), 48):
                    for key, _pid, GL, POS, MOV, LEN, SUBJ in pool.map(m88.read_part, files[i:i + 48], chunksize=2):
                        m = np.array([k in need for k in key])
                        if m.any():
                            for name, arr in (("key", key), ("GL", GL), ("POS", POS), ("MOV", MOV), ("LEN", LEN), ("SUBJ", SUBJ)):
                                parts[name].append(arr[m])
            A = {k: np.concatenate(x) for k, x in parts.items()}
            at = {k: i for i, k in enumerate(A["key"])}
            c_rows = np.array([at[k] for p in players for k, _ in cands[p]])
            c_own = np.array([pidx[p] for p in players for _ in cands[p]])
            q_rows = np.array([at[k] for q in questions for k, _ in q])
            q_of = np.array([j for j, q in enumerate(questions) for _ in q])  # question of every query game
            truth_q = np.array([pidx[p] for p in owner])
            n_q = np.array([len(q) for q in questions])
            C = len(players)
            cp, c_enc, c_gl, c_ply, c_game = tokens(net, A, c_rows)
            qp, q_enc, q_gl, q_ply, q_game = tokens(net, A, q_rows)
            P = np.zeros((C, cp.shape[1]), np.float32)
            np.add.at(P, c_own, cp)
            P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
            c_owner_tok = torch.as_tensor(c_own)[c_game]
            S_game = {"pooled": qp @ P.T}
            for kind, ct, qt in (("enc", c_enc, q_enc), ("gl", c_gl, q_gl)):
                for name, M in match(qt, q_ply, q_game, len(q_rows), ct, c_ply, c_owner_tok, C).items():
                    S_game[f"{kind}_{name}"] = M
            # 1-game queries: every question game on its own; questions: the mean of their games' rows (pooled: summed vectors)
            Q = np.zeros((len(questions), cp.shape[1]), np.float32)
            np.add.at(Q, q_of, qp)
            S_q = {"pooled": (Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)) @ P.T}
            for name, M in S_game.items():
                if name != "pooled":
                    acc = np.zeros((len(questions), C), np.float32)
                    np.add.at(acc, q_of, M)
                    S_q[name] = acc / n_q[:, None]
            runs.append({"fold": v, "seed": s, "game": {k: zrow(x) for k, x in S_game.items()}, "truth_game": truth_q[q_of], "q": {k: zrow(x) for k, x in S_q.items()}, "truth_q": truth_q, "n_q": n_q})
            print(f"fold {v} seed {s}: {len(q_rows)} query games, {len(c_enc)} candidate tokens, {len(q_enc)} question tokens ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(0)
    out = {"network": a.ckpt_suffix, "one_game": {}, "questions": {}}
    names = [k for k in runs[0]["game"] if k != "pooled"]

    def report(level, key_s, key_t, sizes):
        base = np.concatenate([per_question(r[key_s]["pooled"], r[key_t]) for r in runs])
        cl = np.concatenate([np.char.add(f"{r['fold']}|", r[key_t].astype(str)) for r in runs])
        ids, inv = np.unique(cl, return_inverse=True)
        print(f"== {level}: pooled vectors alone {base.mean():.4f} ({len(base)} queries)", flush=True)
        for nm in names:
            alone = np.concatenate([per_question(r[key_s][nm], r[key_t]) for r in runs])
            # the weight for a fold's pools is the best one on the other folds' pools
            comb = []
            for r in runs:
                others = [o for o in runs if o["fold"] != r["fold"]]
                w = max(WEIGHTS, key=lambda w_: np.mean([per_question(o[key_s]["pooled"] + w_ * o[key_s][nm], o[key_t]).mean() for o in others]))
                comb.append((per_question(r[key_s]["pooled"] + w * r[key_s][nm], r[key_t]), w))
            d = np.concatenate([c for c, _ in comb]) - base
            per = np.bincount(inv, weights=d, minlength=len(ids)) / np.bincount(inv, minlength=len(ids))
            bs = per[rng.integers(0, len(ids), (a.boot, len(ids)))].mean(1)
            rec = {"alone": float(alone.mean()), "added_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "weights": [w for _, w in comb]}
            out[level][nm] = rec
            print(f"   {nm:11s} alone {alone.mean():.4f}; added to the pooled score {d.mean():+.4f} [{rec['ci95'][0]:+.4f}, {rec['ci95'][1]:+.4f}] (weights {rec['weights']})", flush=True)
        out[level]["pooled"] = float(base.mean())

    report("one_game", "game", "truth_game", None)
    report("questions", "q", "truth_q", SIZES)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
