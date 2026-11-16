#!/usr/bin/env python
"""Mandatory turn-order gate for immutable v1 extraction inputs.

v1 used explicit-color legality with board-relative features. Validating all
manifest moves establishes that the two colors coincide for existing shards.
New wrapper invocations also check this invariant before extraction.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pid_choice_manifest import atomic_json, digest_file, read_manifest
from pid_choice_extract import replay_choices, te


def validate_turn_order(sgf):
    stones, moves = te.setup_and_moves(sgf)
    state = te.start_state(stones, te.RULES)
    expected = state.board.pla
    for t, (c, _, _) in enumerate(moves[:256]):
        pla = te.Board.BLACK if c == "B" else te.Board.WHITE
        if pla != expected:
            raise ValueError(f"turn mismatch at ply {t}: recorded {c}, expected {expected}")
        expected = te.Board.get_opp(pla)
    return min(len(moves), 256)


def replay_choices_checked(key, sgf, color):
    validate_turn_order(sgf)
    return replay_choices(key, sgf, color)


def audit_manifest(run_dir):
    m = read_manifest(run_dir)
    failures, positions = [], 0
    for g in m["games"]:
        try:
            positions += validate_turn_order(g["sgf"])
        except ValueError as e:
            failures.append({"key": g["key"], "reason": str(e)})
    result = {"run_id": m["run_id"], "games": len(m["games"]), "plies_checked": positions, "failures": failures,
              "verdict": "passed" if not failures else "input_invalid", "code_sha256": digest_file(__file__)}
    atomic_json(Path(run_dir) / "audit_input.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    if audit_manifest(a.run_dir)["verdict"] != "passed":
        raise SystemExit(2)
