"""Replay the candidate games and retain only the tactical contingency tables."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_tactical_inputs import CACHE, PLAN, POOLS, RUN, atomic_json, atomic_npz, digest_file, guard, read_manifest
from aicup.sgf import setup_and_moves
from katago.game.board import Board
from aicup_pid.tactical_replies import counts, opportunity, play_checked, play_response


def extract_game(game):
    stones, moves = setup_and_moves(game["sgf"])
    sig = hashlib.sha256(repr((sorted(stones), moves)).encode()).hexdigest()
    board = Board(19)
    for x, y in stones:
        board.set_stone(Board.BLACK, board.loc(x, y))
    if stones:
        board.set_pla(Board.WHITE)
    events, actions, phases = [], [], []
    prev = Board.PASS_LOC
    for t, (colour, x, y) in enumerate(moves[:256]):
        player = Board.BLACK if colour == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else board.loc(x, y)
        if colour == game["color"] and t >= 12 and prev != Board.PASS_LOC:
            events.append(opportunity(board, player, prev))
            actions.append(play_response(board, player, loc, prev))
            phases.append(int(t >= 80) + int(t >= 160))
        else:
            play_checked(board, player, loc)
        prev = loc
    seed = int(hashlib.sha256(("tactical-shuffle-v1|" + sig).encode()).hexdigest()[:16], 16)
    table, perm = counts(events, actions, phases, seed)
    return game["key"], sig, table, perm


def main():
    budget = guard()
    m = read_manifest(RUN, verify=False)
    games = [g for g in m["games"] if g["pool"] in POOLS]
    if len(games) != 12000 or any(not g["original_key"].startswith("c:") for g in games):
        raise ValueError("expected only the two 6,000-game candidate pools")
    with ProcessPoolExecutor(max_workers=10) as workers:
        res = list(workers.map(extract_game, games, chunksize=32))
    keys, signatures, intact, shuffled = map(np.asarray, zip(*res))
    if len(set(signatures)) != len(games) or len(set(keys)) != len(games):
        raise ValueError("physical game duplication")
    if not np.array_equal(intact[:, :, 0], shuffled[:, :, 0]) or not np.array_equal(intact.sum(1), shuffled.sum(1)):
        raise ValueError("shuffled reference control changed the margins")
    path = CACHE / "tactics.npz"
    atomic_npz(path, keys=keys, sig=signatures, intact=intact, shuffled=shuffled)
    atomic_json(CACHE / "extract.json", {"plan": digest_file(PLAN), "data": digest_file(path),
                "games": len(keys), "unique_full_histories": len(set(signatures)), "budget_start": budget,
                "observations": int(intact[:, :, 0].sum()), "games_without_observations": int((intact[:, :, 0].sum(1) == 0).sum()),
                "contexts": intact[:, :, 0].sum(0).tolist(), "responses": intact[:, :, 1:].sum((0, 1)).tolist()})
    print("verified physical-game separation and intact response margins", len(keys), flush=True)


if __name__ == "__main__":
    main()
