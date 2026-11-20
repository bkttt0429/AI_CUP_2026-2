#!/usr/bin/env python
"""Verifier scores of the frozen conditional behaviour policy on one pseudo-pool split (METHOD_NOTES 2.209, feasibility probe).

The split is the one of scripts/pid_v2_offsim_par.py --holdout 3 --only POOL:SEED: per candidate of an official added pool
aicup_pid.v2.candidate_holdout takes 3 games out; the other 7 are its reference games, each held-out game is a 1-game
question. Only the labelled CANDIDATE games are used (no question file game, no training fold).

The model is the research agent's (read-only; its modules are imported from their own directories, nothing is copied):
/work/u2507115/behavior_probe_20261116/models/contextual_<seed>.pt on top of the pristine humanv0 trunk. Per game its
replay keeps at most 16 of the player's own moves, evenly spaced over the game (behavior_data.position_indices). A candidate's
memory = the positions of its reference games (<= 7 x 16 = 112); for a query game the policy gives, per candidate, the
log-probability of every played move, with the memory intact and with ablation='permute' (the value vectors shifted by one
position inside the candidate, the summary of the candidate unchanged). Stored per query game and model seed: the mean
log-probability per ply under both, for all candidates. The verifier score of the pilot is intact - permute, mean over seeds.

  --out FILE.npz   keys [q], row [q] (index of the 1-game question in offsim's question list), truth [q] (index in players),
                   true_id [q], plies [q], intact / permute [q, model seed, candidate] float32, neutral [q] (the trunk's own
                   policy), players [C], model_seeds, ref_games / ref_positions [C], rejected (+ _why): games the replay refused
  --shard K --nshards N   every N-th query game; EVERY shard builds the memory of all candidates itself (nothing is shared)
Writes nothing but --out (through --out + '.tmp'). Compute node only.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
PILOT = Path("/work/u2507115/behavior_probe_20261116")  # frozen models, protocol.json, selection.json
HEAD = Path("/home/u2507115/.cache/pidbehavior_wt/scripts")  # behavior_model, behavior_data, pid_behavior_extract, pid_behavior_train
CHOICE = Path("/home/u2507115/.cache/pidchoice_wt/scripts")  # pid_choice_extract.forward_choices
EXAMPLE = Path("/home/u2507115/BK_AICUP_2026-1/research_notes/behavior_rank_20261118")  # score_games.features / checked_replay


def pseudo_pool(pool, seed, fusion, tests_dir, holdout=3):
    """(players, ref {player: [(key, colour)]}, questions [[(key, colour)]], truth [Q], sgfs of the candidate games): the same
    calls in the same order as pid_v2_offsim_par.main, lines 117-136 (the draw itself is v2.candidate_holdout)."""
    import aicup_pid  # noqa: F401  (puts AI_CUP_2026-1's src on sys.path)
    from aicup.submission import check_game
    from aicup_pid import v2
    from aicup_pid.trunkstyle import load_store
    T, tag = Path(tests_dir), {"pub2": "public", "priv2": "private"}[pool]
    store = load_store(str(ROOT / json.loads(Path(fusion).read_text())["trunk_store"]))
    cands, _q, _ids, sgfs = v2.official_pool(T / f"player_identification_candidates2_{tag}.csv", T / f"player_identification_test2_{tag}.csv")
    players = sorted(cands)
    ckeys = {k: c for g in cands.values() for k, c in g}
    bad = {k for k, c in ckeys.items() if not check_game(sgfs[k], c)[0] or k not in store}
    hold = v2.candidate_holdout(cands, holdout, seed)
    held = {k for g in hold.values() for k, _ in g}
    ref = {p: [(k, c) for k, c in cands[p] if k not in held and k not in bad] for p in players}
    questions, truth = [], []
    for i, p in enumerate(players):
        h = [(k, c) for k, c in hold[p] if k not in bad]
        for g in h:
            questions.append([g]); truth.append(i)
        for m in (2, 3):
            if len(h) >= m:
                questions.append(h[:m]); truth.append(i)
    return players, ref, questions, np.array(truth), {k: sgfs[k] for k in ckeys}


def ref_value(phi, lp, played):
    """The value vector of a reference position as the pilot stored it (pid_behavior_extract.extract, verify_scores.condition):
    feature of the played move minus the policy-weighted mean feature, and the trunk's log-probability of the played move."""
    import torch
    ix = torch.arange(len(played), device=played.device)
    return torch.cat([phi[ix, played] - (lp.exp()[..., None] * phi).sum(1), lp[ix, played, None]], -1)


def save(path, **arrays):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", required=True, choices=("pub2", "priv2"))
    ap.add_argument("--holdout-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit-queries", type=int, default=None, help="smoke test: only the first M 1-game questions (all shards together); the memory is still built for every candidate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fusion", default=str(ROOT / "results" / "pid_fusion_v2fin_x.json"), help="only its trunk_store is read: the games offsim leaves out")
    ap.add_argument("--tests-dir", default=str(ROOT / "data" / "tests_1111"))
    ap.add_argument("--workers", type=int, default=7, help="replay processes (the pilot's number)")
    ap.add_argument("--batch", type=int, default=8, help="query games per forward of the policy")
    a = ap.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run inside a Slurm allocation (compute node)")
    assert a.nshards >= 1 and 0 <= a.shard < a.nshards and a.out.endswith(".npz")
    t0 = time.time()
    players, ref, questions, truth, sgfs = pseudo_pool(a.pool, a.holdout_seed, a.fusion, a.tests_dir)
    C = len(players)
    single = [(r, q[0][0], q[0][1], int(truth[r])) for r, q in enumerate(questions) if len(q) == 1]
    todo = single[:a.limit_queries][a.shard::a.nshards]
    pilot = json.loads((PILOT / "protocol.json").read_text())
    seeds = pilot["seeds"]
    print(f"{a.pool} split {a.holdout_seed}: {C} candidates, {sum(map(len, ref.values()))} reference games, {len(single)} 1-game questions, "
          f"{len(todo)} in shard {a.shard}/{a.nshards}; model seeds {seeds} ({time.time() - t0:.0f} s)", flush=True)
    common = dict(pool=a.pool, seed=a.holdout_seed, shard=a.shard, nshards=a.nshards, limit=a.limit_queries or 0, players=np.array(players, dtype=str),
                  model_seeds=np.array(seeds), n_questions=len(questions), n_single=len(single))
    if not todo:
        e = np.zeros((0, len(seeds), C), np.float32)
        save(a.out, **common, keys=np.zeros(0, dtype=str), row=np.zeros(0, np.int64), truth=np.zeros(0, np.int64), true_id=np.zeros(0, dtype=str),
             plies=np.zeros(0, np.int64), intact=e, permute=e, neutral=np.zeros(0, np.float32), rejected=np.zeros(0, dtype=str), rejected_why=np.zeros(0, dtype=str))
        print(f"no query game in this shard; wrote {a.out}", flush=True)
        return 0

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import torch
    import aicup_pid
    sys.path.insert(0, str(HEAD))
    sys.path.append(str(EXAMPLE))
    from behavior_data import digest
    from behavior_model import BehaviorPolicy
    from pid_behavior_extract import init_worker
    from pid_behavior_train import references
    from score_games import checked_replay, features
    # pid_choice_extract imports aicup_pid.choice_signal, which only the research worktree's copy of the package has: that one
    # directory is added BEHIND this repository's, so every other aicup_pid module (v2 above included) stays this repository's
    aicup_pid.__path__.append(str(CHOICE.parent / "src" / "aicup_pid"))
    sys.path.insert(0, str(CHOICE))
    import pid_choice_extract
    from aicup.humansl import load_humansl
    assert callable(pid_choice_extract.forward_choices)
    torch.set_num_threads(1)
    # the frozen pilot: its code, its extractor sources and the trunk are unchanged; the two policies are the selected ones
    pinned = {**json.loads((PILOT / "training_protocol.json").read_text())["code"], **pilot["sources"], pilot["checkpoint"]["path"]: pilot["checkpoint"]["sha256"]}
    for path, sha in pinned.items():
        assert digest(path) == sha, f"{path} changed since the pilot"
    frozen = json.loads((PILOT / "selection.json").read_text())["checkpoint_sha256"]
    nets = []
    for s in seeds:
        f = PILOT / "models" / f"contextual_{s}.pt"
        assert digest(f) == frozen[f.name], f"{f} is not the selected checkpoint"
        net = BehaviorPolicy("contextual").to(a.device)
        net.load_state_dict(torch.load(f, map_location=a.device, weights_only=True))
        nets.append(net.eval())
    ckpt = pilot["checkpoint"]["path"]
    config = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    katago, _ = load_humansl(ckpt, a.device)
    katago.eval()
    projection = np.eye(384, dtype=np.float32)
    rejected = {}
    with torch.inference_mode(), ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(config,)) as pool:
        def stream(games, what):
            """(key, (phi, logp, state, history, played)) of every game that replays; the others go to `rejected`."""
            for at in range(0, len(games), 28):
                items = [(k, {"sgf": sgfs[k], "color": c}) for k, c in games[at:at + 28]]
                for k, rep, err in pool.map(checked_replay, items, chunksize=1):
                    if err is None:
                        yield k, features(katago, rep, projection)
                    else:
                        rejected[k] = err
                if at % 560 == 0:
                    print(f"  {what}: {min(at + 28, len(games))}/{len(games)} games, {len(rejected)} refused ({time.time() - t0:.0f} s)", flush=True)

        # the pilot's own check: the streamed features reproduce its stored tensors of its first game
        _, rep, err = pool.submit(checked_replay, ("check", pilot["rows"][0])).result()
        assert err is None, err
        got = features(katago, rep, projection)
        with np.load(PILOT / "games" / "0000.npz", allow_pickle=False) as z:
            for name, x in zip(("phi", "logp", "state", "history", "played", "ref_value"), (*got, ref_value(got[0], got[1], got[4]))):
                np.testing.assert_allclose(x.cpu().numpy(), z[name], atol=.025, rtol=.005, err_msg=name)
        # the memory: one owner per candidate, every position of its reference games
        data = {}
        for k, (phi, lp, state, history, played) in stream([g for p in players for g in ref[p]], "memory"):
            data[k] = dict(state=state, history=history, played=played, ref_value=ref_value(phi, lp, played))
        groups = [[k for k, _ in ref[p] if k in data] for p in players]
        assert all(groups), "a candidate has no reference game that replays"
        mem = references(data, groups)
        del data
        print(f"memory: state {tuple(mem[0].shape)}, history {tuple(mem[1].shape)}, value {tuple(mem[2].shape)}; positions per candidate "
              f"{int(mem[3].sum(1).min())}-{int(mem[3].sum(1).max())} ({time.time() - t0:.0f} s)", flush=True)
        keys, plies, neutral, blocks = [], [], [], []
        got = stream([(k, c) for _, k, c, _ in todo], "queries")
        for buf in iter(lambda: list(itertools.islice(got, a.batch)), []):
            phi, lp, state, history, played = [torch.cat(f) for f in zip(*(v for _, v in buf))]
            n = [len(v[-1]) for _, v in buf]
            neutral.extend(x.mean().item() for x in lp[torch.arange(len(played), device=played.device), played].split(n))
            per = []
            for net in nets:
                for ablation in (None, "permute"):
                    pred = net(phi, lp, state, history, *mem, ablation=ablation)  # [plies, C, 362] log-probabilities
                    own = pred.gather(2, played[:, None, None].expand(-1, C, 1)).squeeze(-1)
                    per.append(torch.stack([x.mean(0) for x in own.split(n)]))  # [games, C]
            blocks.append(torch.stack(per, 1).reshape(len(buf), len(nets), 2, C).cpu().numpy())
            keys.extend(k for k, _ in buf)
            plies.extend(n)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    S = np.concatenate(blocks) if blocks else np.zeros((0, len(nets), 2, C), np.float32)
    assert np.isfinite(S).all() and np.isfinite(neutral).all() and len(set(keys)) == len(keys)
    at = {k: (r, t) for r, k, _, t in todo}
    bad = sorted(rejected)
    save(a.out, **common, keys=np.array(keys, dtype=str), row=np.array([at[k][0] for k in keys], np.int64), truth=np.array([at[k][1] for k in keys], np.int64),
         true_id=np.array([players[at[k][1]] for k in keys], dtype=str), plies=np.array(plies, np.int64), intact=S[:, :, 0].astype(np.float32),
         permute=S[:, :, 1].astype(np.float32), neutral=np.array(neutral, np.float32), ref_games=np.array([len(g) for g in groups]),
         ref_positions=mem[3].sum(1).cpu().numpy(), rejected=np.array(bad, dtype=str), rejected_why=np.array([rejected[k] for k in bad], dtype=str),
         code_sha=digest(Path(__file__)), example_sha=digest(EXAMPLE / "score_games.py"))
    print(f"wrote {a.out}: {len(keys)} of {len(todo)} query games x {len(nets)} model seeds x {C} candidates; {len(bad)} games refused by the replay; "
          f"peak GPU memory {peak:.1f} GB ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
