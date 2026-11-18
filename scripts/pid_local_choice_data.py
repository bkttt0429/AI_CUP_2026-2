"""Select fresh players and audit the prior study data used for the local-choice pilot."""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pid_reference_policy import (
    FOLDS_CSV, LEDGER, ROOT, TRAIN_DIR, atomic_json, check, physical_signature,
    read_manifest, read_protocol, setup_and_moves, source_record, stable_hash, verify_source,
)

RUN = Path("/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8")
DEST = RUN / "local_choice_20261118"
SIZES = {"train": 200, "dev": 50, "test": 100}


def stable_order(value):
    return stable_hash("local-choice-20261118|" + value)


def validate(rows):
    if len({g["key"] for g in rows}) != len(rows):
        raise ValueError("duplicate game key")
    if len({physical_signature(g) for g in rows}) != len(rows):
        raise ValueError("duplicate physical game")
    if len({g["owner"] for g in rows}) != sum(SIZES.values()):
        raise ValueError("owner crosses splits")
    for split, size in SIZES.items():
        owners = defaultdict(list)
        for g in rows:
            if g["split"] == split:
                owners[g["owner"]].append(g["slot"])
                allowed = {"train": {1, 2}, "dev": {3}, "test": {4}}[split]
                if g["fold"] not in allowed:
                    raise ValueError("wrong player fold")
        if len(owners) != size or any(sorted(slots) != list(range(10)) for slots in owners.values()):
            raise ValueError(f"wrong {split} games or owners")


def data_id(manifest):
    return stable_hash(json.dumps({k: v for k, v in manifest.items() if k != "id"}, sort_keys=True))[:12]


def read_data():
    m = json.loads((DEST / "data.json").read_text())
    if data_id(m) != m["id"]:
        raise ValueError("modified data manifest")
    verify_source(m["checkpoint"])
    verify_source(m["selection_code"])
    validate(m["games"])
    return m


def prepare():
    if (DEST / "data.json").exists():
        return read_data()
    old = read_protocol(RUN)
    archive = read_manifest(RUN, verify=False)
    if old["id"] != "7b991b270173" or archive["run_id"] != old["parent_run"]:
        raise ValueError("wrong parent source")
    past_rows = old["games"] + archive["games"]
    blocked = {g["owner"] for g in past_rows}
    seen = {physical_signature(g) for g in past_rows}
    with open(FOLDS_CSV) as file:
        table = list(csv.DictReader(file))
    eligible = sorted({r["player_id"] for r in table if int(r["fold"]) == 4 and
                       int(r["n_games"]) >= 10 and r["player_id"] not in blocked}, key=stable_order)
    selected_pool = set(eligible[:300])
    candidates = defaultdict(list)
    for path in sorted(TRAIN_DIR.glob("train_*.csv")):
        with open(path) as file:
            for r in csv.DictReader(file):
                if r["player_id"] in selected_pool:
                    candidates[r["player_id"]].append((stable_order(r["rank"] + ":" + r["game_id"]), {
                        "key": r["rank"] + ":" + r["game_id"], "sgf": r["sgf_content"], "owner": r["player_id"],
                        "fold": 4, "split": "test", "color": r["color"],
                    }))
    rows = [g for g in old["games"] if g["split"] != "test"]
    counts = Counter()
    for p in eligible[:300]:
        fresh, physical = [], set()
        for _, g in sorted(candidates[p], key=lambda x: x[0]):
            _, moves = setup_and_moves(g["sgf"])
            if sum(c == g["color"] for c, _, _ in moves[:256]) < 32:
                counts["too_short_examined"] += 1
                continue
            sig = physical_signature(g)
            if sig in seen or sig in physical:
                counts["duplicate_examined"] += 1
                continue
            physical.add(sig)
            fresh.append(dict(g, slot=len(fresh)))
            if len(fresh) == 10:
                break
        if len(fresh) != 10:
            continue
        rows.extend(fresh)
        seen.update(physical)
        counts["test_owners"] += 1
        if counts["test_owners"] == SIZES["test"]:
            break
    validate(rows)
    fresh_rows = [g for g in rows if g["split"] == "test"]
    if {g["owner"] for g in fresh_rows} & blocked:
        raise ValueError("previously inspected owner in fresh test")
    counts.update(eligible_by_fold_table=len(eligible), prior_physical_games=len({physical_signature(g) for g in past_rows}))
    manifest = {"games": rows, "audit": dict(counts), "sizes": SIZES, "checkpoint": old["checkpoint"],
                "prior_data": [source_record(RUN / "manifest.json"), source_record(RUN / "refpolicy/manifest.json")],
                "selection_code": source_record(__file__), "training_sources": old["sources"],
                "test_game_ranks": dict(Counter(g["key"].split(":")[0] for g in fresh_rows))}
    manifest["id"] = data_id(manifest)
    atomic_json(DEST / "data.json", manifest)
    return manifest


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("use production Slurm")
    check(RUN, LEDGER)
    manifest = prepare()
    print(json.dumps({"id": manifest["id"], "sizes": SIZES, "audit": manifest["audit"]}), flush=True)
