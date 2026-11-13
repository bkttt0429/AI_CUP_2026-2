"""Move-level matching (METHOD_NOTES 2.56, atom 3): every own move of a player is a token, the stored input of that move
(board-pooled + played-point trunk vectors of the frozen network, 128 numbers: the situation and the move, no identity
training). A question game is scored against a candidate by how its tokens are matched among the tokens of the candidate's
games. Network-independent: computed once per pool from the per-ply parts."""
import numpy as np
import torch

MAXTOK = 128  # own moves kept per game (the per-ply parts hold 256 plies)


def game_tokens(GL, MOV, LEN, SUBJ):
    """(tokens [n_tok, 128] float32 un-normalised, game index [n_tok], ply [n_tok]) of the subject's own moves."""
    GL, MOV, LEN, SUBJ = (torch.as_tensor(np.asarray(x)) for x in (GL, MOV, LEN, SUBJ))
    T = GL.shape[1]
    own = (MOV.long() == SUBJ.long()[:, None]) & (torch.arange(T)[None] < LEN.long()[:, None])
    b, t = own.nonzero(as_tuple=True)
    keep = own.cumsum(1)[b, t] <= MAXTOK
    b, t = b[keep], t[keep]
    return GL[b, t].float(), b, t


@torch.no_grad()
def scores(c_tok, c_owner, c_ply, C, q_tok, q_game, q_ply, n_q, ks=(20, 5), split=40, chunk=1024, device="cuda"):
    """{name: [n_q games, C]} per question game, each row the mean over the game's tokens:
    vote<k>       every token votes for the owners of its k nearest candidate tokens (cosine, weights = cosine clipped at 0)
    vote<k0>_open / _mid   the first k, only the question tokens before / from ply `split`
    maxsim        the best cosine among the candidate's tokens
    Tokens are standardised with the candidate tokens' mean and deviation, then normalised."""
    mu, sd = c_tok.mean(0), c_tok.std(0) + 1e-6
    ct = torch.nn.functional.normalize((c_tok - mu) / sd, dim=-1).half().to(device)
    qt_all = torch.nn.functional.normalize((q_tok - mu) / sd, dim=-1).half()
    own = torch.as_tensor(np.asarray(c_owner)).long().to(device)
    cnt = torch.bincount(own, minlength=C)
    L = int(cnt.max())
    order = torch.argsort(own, stable=True)
    slot = torch.arange(len(order), device=device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    grid_at = torch.full((C * L,), -1, dtype=torch.long, device=device)
    grid_at[own[order] * L + slot] = order  # which candidate token sits in each cell of the padded [C, L] grid
    k0 = ks[0]
    out = {f"vote{k}": torch.zeros(n_q, C, device=device) for k in ks}
    out.update({f"vote{k0}_open": torch.zeros(n_q, C, device=device), f"vote{k0}_mid": torch.zeros(n_q, C, device=device), "maxsim": torch.zeros(n_q, C, device=device)})
    q_game_d, q_ply_d = torch.as_tensor(np.asarray(q_game)).long().to(device), torch.as_tensor(np.asarray(q_ply)).long().to(device)
    for i in range(0, len(qt_all), chunk):
        qt, qg, qp = qt_all[i:i + chunk].to(device), q_game_d[i:i + chunk], q_ply_d[i:i + chunk]
        S = (qt @ ct.T).float()  # [chunk, n candidate tokens]
        top = S.topk(max(ks), dim=1)
        for k in ks:
            v = torch.zeros(len(qt), C, device=device).scatter_add_(1, own[top.indices[:, :k]], top.values[:, :k].clamp(min=0))
            out[f"vote{k}"].index_add_(0, qg, v)
            if k == k0:
                out[f"vote{k0}_open"].index_add_(0, qg, v * (qp < split)[:, None])
                out[f"vote{k0}_mid"].index_add_(0, qg, v * (qp >= split)[:, None])
        G = torch.cat([S, torch.full((len(qt), 1), -1.0, device=device)], 1)[:, grid_at]  # empty cells read the appended -1 column
        out["maxsim"].index_add_(0, qg, G.view(len(qt), C, L).max(-1).values)
    ntok = torch.bincount(q_game_d, minlength=n_q).clamp(min=1)[:, None]
    return {k: (x / ntok).cpu().numpy() for k, x in out.items()}


def component(arrays, cands, questions, players=None, **kw):
    """{name: [Q, C]} for one pool. arrays: {key, GL, MOV, LEN, SUBJ} of the pool's games; cands {player: [(key, colour)]};
    questions [[(key, colour)]]. A question's row is the mean of its games' rows."""
    players = players or sorted(cands)
    at = {k: i for i, k in enumerate(np.asarray(arrays["key"]).astype(str))}
    c_rows = np.array([at[k] for p in players for k, _ in cands[p]])
    c_own_game = np.array([i for i, p in enumerate(players) for _ in cands[p]])
    q_rows = np.array([at[k] for q in questions for k, _ in q])
    q_of = np.array([j for j, q in enumerate(questions) for _ in q])
    pick = lambda rows: (arrays["GL"][rows], arrays["MOV"][rows], arrays["LEN"][rows], arrays["SUBJ"][rows])  # noqa: E731
    c_tok, c_game, c_ply = game_tokens(*pick(c_rows))
    q_tok, q_game, q_ply = game_tokens(*pick(q_rows))
    per_game = scores(c_tok, c_own_game[c_game.numpy()], c_ply, len(players), q_tok, q_game, q_ply, len(q_rows), **kw)
    n = np.bincount(q_of, minlength=len(questions))[:, None]
    out = {}
    for name, M in per_game.items():
        acc = np.zeros((len(questions), len(players)), np.float32)
        np.add.at(acc, q_of, M)
        out[name] = acc / n
    return out
