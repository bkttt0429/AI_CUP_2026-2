#!/usr/bin/env python
"""Line B4 (trunk variant): per-game trunk-style features of the pristine humanv0 trunk (METHOD_NOTES 2.21).

Every game is replayed exactly as AI_CUP_2026-1's scripts/trunk_extract.py replays it (same Features, rules, handicap fix,
random projections to 64 + 64 dims with the same seed); the trunk is run on the TARGET player's positions only (Black in
this task), and each game is stored as one 256-dimensional row: mean and standard deviation of the board-pooled vector
and of the played-point vector over the target's plies (passes have no played point). Rows are written in parts
(<out>/<tag>_s<global shard>_p<k>.npz: keys, feats float32 [G, 256], n_target) and the run resumes per part.

Inputs: --official (both official pools: candidate games keyed c:<row>:<game_id> as predict_pid.py keys them, question
games q:<question_id>:<i>) or --keys FILE (training game keys, one per line or a CSV with a game_key column). Sharding:
shards.sh appends --shard i --nshards N per GPU; --node k --nodes M spreads one key list over M such jobs.

  sbatch -p 8gpus --gres=gpu:8 -c 96 -t 03:00:00 -J trunkx scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 \\
      python -I ../BK_AICUP_2026-2/scripts/pid_trunk_extract.py --official --out ../BK_AICUP_2026-2/data/trunk/official --node 0 --nodes 2 --workers 10
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
import aicup_pid  # noqa: E402  (puts AI_CUP_2026-1/src on sys.path)
from aicup_pid import pools  # noqa: E402
from aicup_pid.symmetry import game_sym, transform_sgf  # noqa: E402
from aicup_pid.paths import AICUP1  # noqa: E402,F401  (its scripts directory is on sys.path)
import trunk_extract as te  # noqa: E402  (AI_CUP_2026-1's replay, projections and vectors)
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.submission import check_game  # noqa: E402

DIM = 4 * te.PROJ_DIM
T = ROOT / "data" / "tests"


def official_tasks(tests_dir=T):
    """{key: (sgf, colour)} of every usable game of both official pools in tests_dir (default data/tests)."""
    out = {}
    for c_name, q_name in (("player_identification_candidates_public.csv", "player_identification_test_public.csv"),
                           ("player_identification_candidates_private.csv", "player_identification_test_private.csv")):
        cands = pools.read_candidates(Path(tests_dir) / c_name)
        for i, r in enumerate(cands.itertuples()):
            out[f"c:{i}:{r.game_id}"] = (r.sgf, "B")
        for q in pools.read_questions(Path(tests_dir) / q_name):
            for i, (sgf, colour) in enumerate(q["games"]):
                if check_game(sgf, colour or "B")[0]:
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")
    return out


def official2_tasks(tests_dir):
    """{key: (sgf, colour)} of the 11-08 v2 release's ADDED pools (candidates2 / test2, public and private): both colours, the
    colour column is the player's side. Keys as official_tasks: c:<row in its file>:<game_id>, q:<question_id>:<i>."""
    out = {}
    for tag in ("public", "private"):
        cands = pd.read_csv(Path(tests_dir) / f"player_identification_candidates2_{tag}.csv", dtype=str, keep_default_na=False)
        for i, r in enumerate(cands.itertuples()):
            if check_game(r.sgf_content, r.color)[0]:
                out[f"c:{i}:{r.game_id}"] = (r.sgf_content, r.color)
        for q in pools.read_questions(Path(tests_dir) / f"player_identification_test2_{tag}.csv"):
            for i, (sgf, colour) in enumerate(q["games"]):
                if check_game(sgf, colour or "B")[0]:
                    out[f"q:{q['question_id']}:{i}"] = (sgf, colour or "B")
    return out


def read_keys(path):
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p, dtype=str)["game_key"].tolist()
    return [x.strip() for x in p.read_text().splitlines() if x.strip()]


def aggregate(g, l, pos, lens):
    """[G, 256]: per game mean / std of the pooled and of the played-point vectors over its target plies."""
    out = np.zeros((len(lens), DIM), np.float32)
    start = 0
    for i, n in enumerate(lens):
        if n:
            gi = g[start:start + n].astype(np.float32)
            li = l[start:start + n][pos[start:start + n] < te.PASS_POS].astype(np.float32)
            out[i, 0:64], out[i, 64:128] = gi.mean(axis=0), gi.std(axis=0)
            if len(li):
                out[i, 128:192], out[i, 192:256] = li.mean(axis=0), li.std(axis=0)
        start += n
    return out


def replay_target(task):
    """trunk_extract.replay for the TARGET side's plies only: the game is played through as there, but the input features of the
    other side's positions are never computed (they are most of the replay's cost; the rank session's replay_side, AI_CUP_2026-1
    experiments/nano4/m118_fullrows_extract.py, checked against the full replay). Returns (key, bins, globs, pos, mover, t) for the
    target's plies, t their ply numbers; bins and globs are None when the target has no ply."""
    idx, sgf, color = task
    stones, moves = te.setup_and_moves(sgf)
    gs = te.start_state(stones, te.RULES)
    bins, globs, pos, mover, ts = [], [], [], [], []
    for t, (c, x, y) in enumerate(moves):
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE
        loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if c == color:
            b, g = gs.get_input_features(te._FEATURES)
            bins.append(b[0].astype(np.uint8))
            globs.append(g[0].astype(np.float32))
            pos.append(te.PASS_POS if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board))
            mover.append(0 if c == "B" else 1)
            ts.append(t)
        gs.play(pla, loc)
    if not pos:
        return idx, None, None, np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.int64)
    return idx, np.stack(bins), np.stack(globs), np.asarray(pos, np.int16), np.asarray(mover, np.int8), np.asarray(ts, np.int64)


def full_vectors(model, bins, globs, pos, mover, rows, batch):
    """[n, C] float32 board-pooled and played-point trunk vectors WITHOUT the random projection (C = 384; a pass has a zero
    played-point vector), computed as trunk_extract.vectors computes them before it projects (METHOD_NOTES 2.69, probe)."""
    n = len(pos)
    out_g = out_l = None
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b, g = torch.as_tensor(bins[i:j]).cuda(), torch.as_tensor(globs[i:j]).cuda()
            c = torch.as_tensor(mover[i:j]).cuda().long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)
            trunk = trunk.float()
            p = torch.as_tensor(pos[i:j]).cuda().long()
            on = p < te.PASS_POS
            local = torch.zeros(j - i, trunk.shape[1], device=trunk.device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            if out_g is None:
                out_g, out_l = np.empty((n, trunk.shape[1]), np.float32), np.empty((n, trunk.shape[1]), np.float32)
            out_g[i:j], out_l[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy()
    return out_g, out_l


def policy_vectors(model, bins, globs, pos, mover, rows, batch):
    """full_vectors plus what the network itself EXPECTED at every position (METHOD_NOTES 2.102, probe P). One forward pass
    gives the trunk and the policy (channel 0, the side to move, the neutral metadata row); with the policy over the legal
    board points (on board, empty, not the ko point; pass left out and the rest renormalised):
      g [n, C]  board-pooled trunk vector               x [n, C]  trunk vector at the played point (zero for a pass)
      e [n, C]  policy-weighted mean of the trunk vectors of the board points = the played-point vector the network expects
                (zero for a pass, so that x - e is zero there)
      s [n, 5]  the played move's surprisal (-log p, pass included in that distribution), its rank in the policy (0 = the top
                move), the policy's entropy, its top probability, the Chebyshev distance between the played point and the
                policy's top board point (0 for a pass)
    Also returns the number of played moves my legality mask called illegal (their surprisal is clipped; should be ~0)."""
    from aicup.l1b import _GraphExtraOutputs
    n = len(pos)
    C = None
    S = np.zeros((n, 5), np.float32)
    bad = 0
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device="cuda")
    meta = model.get_has_metadata_encoder()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b, g = torch.as_tensor(bins[i:j]).cuda().float(), torch.as_tensor(globs[i:j]).cuda().float()
            c = torch.as_tensor(mover[i:j]).cuda().long()
            extra = _GraphExtraOutputs(["trunkfinal"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(b, g, input_meta=rows_t[c] if meta else None, extra_outputs=extra)
                logits = model.postprocess_output(outputs)[0][0][:, 0, :]
            trunk, logits = extra.returned["trunkfinal"].float(), logits.float()
            if C is None:
                C = trunk.shape[1]
                G, X, E = np.empty((n, C), np.float32), np.empty((n, C), np.float32), np.empty((n, C), np.float32)
            legal_b = ((b[:, 0] - b[:, 1] - b[:, 2] - b[:, 6]) > 0.5).flatten(1)  # [B, 361]
            legal = torch.cat([legal_b, torch.ones(j - i, 1, dtype=torch.bool, device=b.device)], dim=1)
            lp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
            p = torch.as_tensor(pos[i:j].astype(np.int64)).cuda()
            on = p < te.PASS_POS
            lpp = lp.gather(1, p[:, None])[:, 0]
            miss = ~torch.isfinite(lpp)
            bad += int(miss.sum())
            lpp = torch.where(miss, torch.full_like(lpp, -20.0), lpp).clamp(min=-20.0)
            rank = (lp > lpp[:, None]).sum(1).float()
            pr = lp.exp()
            ent = -(pr * torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))).sum(1)
            pb = torch.softmax(logits[:, :361].masked_fill(~legal_b, float("-inf")), dim=1)
            pb = torch.where(torch.isfinite(pb), pb, torch.zeros_like(pb))  # a full board: no legal point
            e = torch.einsum("bp,bcp->bc", pb, trunk.flatten(2))
            local = torch.zeros(j - i, C, device=trunk.device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            e = e * on[:, None].float()
            top = pb.argmax(1)
            dist = torch.maximum((p // 19 - top // 19).abs(), (p % 19 - top % 19).abs()).float() * on.float()
            G[i:j], X[i:j], E[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy(), e.cpu().numpy()
            S[i:j] = torch.stack([-lpp, rank, ent, pr.max(1).values, dist], dim=1).cpu().numpy()
    return G, X, E, S, bad


def aggregate_full(g, l, pos, lens):
    """[G, 4C]: per game mean / std of the unprojected pooled vectors and of the unprojected played-point vectors (as aggregate)."""
    C = g.shape[1]
    out = np.zeros((len(lens), 4 * C), np.float32)
    start = 0
    for i, n in enumerate(lens):
        if n:
            gi = g[start:start + n]
            li = l[start:start + n][pos[start:start + n] < te.PASS_POS]
            out[i, 0:C], out[i, C:2 * C] = gi.mean(axis=0), gi.std(axis=0)
            if len(li):
                out[i, 2 * C:3 * C], out[i, 3 * C:4 * C] = li.mean(axis=0), li.std(axis=0)
        start += n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--official", action="store_true")
    src.add_argument("--official2", action="store_true", help="the v2 release's added pools (candidates2 / test2), both colours")
    ap.add_argument("--tests-dir", default=str(T), help="folder of the official CSVs for --official (a re-released set goes in its own folder)")
    src.add_argument("--keys", help="training game keys (text file or CSV with game_key)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="part")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-games", type=int, default=0, help="smoke tests only")
    ap.add_argument("--fast", action="store_true", help="replay with the input features of the target side only (replay_target): the same rows, about twice as fast")
    ap.add_argument("--max-parts", type=int, default=0, help="stop after this many parts (a check run)")
    ap.add_argument("--full", action="store_true", help="also store the per-game mean / std of the UNPROJECTED trunk vectors (array `full`, [G, 4 x 384]); the 256-dim rows are then computed from the same vectors")
    ap.add_argument("--sym-seed", type=int, default=None, help="rotate / mirror every game by aicup_pid.symmetry.game_sym(key, sgf, seed) before the replay (simulated re-release; HD >= 2 games stay as they are)")
    a = ap.parse_args(argv)
    t0 = time.time()
    shard, nshards = a.node * a.nshards + a.shard, a.nodes * a.nshards
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.official or a.official2:
        tasks_all = official2_tasks(a.tests_dir) if a.official2 else official_tasks(a.tests_dir)
        keys = sorted(tasks_all)
    else:
        tasks_all, keys = None, sorted(set(read_keys(a.keys)))
    if a.max_games:
        keys = keys[: a.max_games]
    mine = keys[shard::nshards]
    parts = [mine[i:i + a.part_size] for i in range(0, len(mine), a.part_size)]
    todo = [(k, p) for k, p in enumerate(parts) if not (out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz").exists()]
    print(f"shard {shard}/{nshards}: {len(mine)} of {len(keys)} games, {len(parts)} parts, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows = neutral_rows()
        proj_g = proj_l = None
        games_done = plies_done = 0
        for k, part in todo:
            sgf = {q: tasks_all[q] for q in part} if tasks_all is not None else load_games(part, aicup_pid.TRAIN_DIR)
            if a.sym_seed is not None:
                sgf = {q: (transform_sgf(v[0], game_sym(q, v[0], a.sym_seed)), *v[1:]) for q, v in sgf.items()}
            if a.fast:  # only the target's plies come back, with their ply numbers
                done = list(pool.imap(replay_target, [(q, *sgf[q]) for q in part], chunksize=1))
                tsel = [d[5] for d in done]
                some = [d for d in done if len(d[3])]
                bins, globs = np.concatenate([d[1] for d in some]), np.concatenate([d[2] for d in some])
                pos, mover = np.concatenate([d[3] for d in done]), np.concatenate([d[4] for d in done])
                lens = [len(d[3]) for d in done]
            else:
                done = list(pool.imap(te.replay, [(q, *sgf[q]) for q in part], chunksize=1))
                sel = [d[5] for d in done]  # target plies of each game
                tsel = [np.flatnonzero(m) for m in sel]
                bins = np.concatenate([d[1][m] for d, m in zip(done, sel)])
                globs = np.concatenate([d[2][m] for d, m in zip(done, sel)])
                pos = np.concatenate([d[3][m] for d, m in zip(done, sel)])
                mover = np.concatenate([d[4][m] for d, m in zip(done, sel)])
                lens = [int(m.sum()) for m in sel]
            if proj_g is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(bins[:1]).cuda(), torch.as_tensor(globs[:1]).cuda(),
                                                    torch.as_tensor(mover[:1]).cuda().long(), rows)
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)
            full = None
            if a.full:  # one trunk pass: the unprojected vectors, and the usual projected ones from them
                fg, fl = full_vectors(model, bins, globs, pos, mover, rows, a.batch)
                full = aggregate_full(fg, fl, pos, lens)
                g = (fg @ proj_g.float().cpu().numpy()).astype(np.float16)
                l = (fl @ proj_l.float().cpu().numpy()).astype(np.float16)
            else:
                g, l = te.vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, a.batch)
            feats = aggregate(g, l, pos, lens)
            if not np.isfinite(feats).all():
                raise ValueError(f"part {k}: non-finite features")
            # phase means of the target's plies (t < 60, 60 <= t < 150, t >= 150; 128 dims each, zeros when a phase is empty): the same
            # definition as AI_CUP_2026-1 scripts/m_series/m84_rows_extract.py side_rows (a pass keeps a zero played-point vector)
            phase = np.zeros((len(lens), 384), np.float16)
            X = np.hstack([g.astype(np.float32), l.astype(np.float32)])
            start = 0
            for i, (t_, n_) in enumerate(zip(tsel, lens)):
                for j, pm in enumerate((t_ < 60, (t_ >= 60) & (t_ < 150), t_ >= 150)):
                    if pm.any():
                        phase[i, 128 * j:128 * (j + 1)] = X[start:start + n_][pm].mean(axis=0)
                start += n_
            path = out / f"{a.tag}_s{shard:03d}_p{k:04d}.npz"
            tmp = path.with_name(path.name + ".tmp.npz")
            extra = {}
            if full is not None:  # unprojected: the whole-game mean / std [4 x 384] and the three phase means [3 x 768]; 7.7 KB per game
                XF = np.hstack([fg, fl])
                fullp = np.zeros((len(lens), 3 * XF.shape[1]), np.float32)
                start = 0
                for i, (t_, n_) in enumerate(zip(tsel, lens)):
                    for j, pm in enumerate((t_ < 60, (t_ >= 60) & (t_ < 150), t_ >= 150)):
                        if pm.any():
                            fullp[i, XF.shape[1] * j:XF.shape[1] * (j + 1)] = XF[start:start + n_][pm].mean(axis=0)
                    start += n_
                extra = {"full": full.astype(np.float16), "fullp": fullp.astype(np.float16)}
            np.savez(tmp, keys=np.array([d[0] for d in done]), feats=feats, n_target=np.array(lens, np.int16), phase=phase, **extra)
            os.replace(tmp, path)
            games_done += len(part)
            plies_done += sum(len(d[3]) for d in done)
            if a.max_parts and games_done >= a.max_parts * a.part_size:
                break
            el = time.time() - t0
            print(f"part {k + 1}/{len(parts)}: {games_done} games, {games_done / el:.1f} games/s, {plies_done / el:.0f} plies/s replayed", flush=True)
    finally:
        pool.close()
        pool.join()
    print(f"shard {shard} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
