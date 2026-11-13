#!/usr/bin/env python
"""External game records for the identity metric (METHOD_NOTES 2.46): the Fox Go Dataset (github.com/featurecat/go-dataset,
GPL-3.0) -> compact parts, and the count-only overlap gate against the competition's training games.

  --stage ingest --archives <dir with the .7z / .7z.NNN files of one rank folder> --out <dir>
      Every SGF is unpacked to the node's RAM disk, read as data (nothing in it is executed), removed again, and reduced to
      what the competition's own records hold - board size, handicap, the moves - plus the fields used only for filtering
      and grouping: the two ACCOUNT KEYS (a salted hash of PB / PW; the names themselves are never written), ranks, year,
      time settings, result type. Written as <out>/games_NNNN.npz.
  --stage overlap --parts <dirs of ingest outputs ...> --out <json>
      The gate: how many competition training games (fold 0 excluded; test sets never read) have the same first --prefix
      moves as an external game, up to the eight board symmetries. COUNTS ONLY: no account key or name is read for a match.
      Any overlap means the competition data and this source share games: stop and report, do not train on the source.

Moves are stored as uint16 codes colour * 512 + (19 * x + y), pass = 361; a game key is the archive's file name.
Run with python -I (the archives are untrusted data).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MOVE = re.compile(rb";\s*([BW])\[([a-s]{2})?\]")
PROP = re.compile(rb"([A-Z]{1,2})\[([^\]]*)\]")
SYMS = [lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y),
        lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)]


def moves_of(raw):
    """[(colour 0/1, point 0-360 or 361)] of an SGF's main line (bytes)."""
    out = []
    for c, p in MOVE.findall(raw):
        out.append((0 if c == b"B" else 1, 361 if not p else 19 * (p[0] - 97) + (p[1] - 97)))
    return out


def prefix_hash(mv, k):
    """uint64 of the first k moves, the smallest over the eight symmetries; None for a shorter game or a pass inside."""
    if len(mv) < k or any(p == 361 for _, p in mv[:k]):
        return None
    best = None
    for f in SYMS:
        b = bytes(v for c, p in mv[:k] for v in (c, *f(p // 19, p % 19)))
        best = b if best is None or b < best else best
    return int.from_bytes(hashlib.blake2b(best, digest_size=8).digest(), "little")


def rank_code(b):
    """Fox rank text -> int (kyu negative: 18k = -18 ... 1k = -1; dan 1-9; pro 10+; unknown 0)."""
    s = b.decode("utf-8", "replace")
    m = re.match(r"(\d+)\s*(级|級|k|K)", s)
    if m:
        return -int(m.group(1))
    m = re.match(r"(\d+)\s*(段|d|D)", s)
    if m:
        return int(m.group(1))
    m = re.match(r"[Pp职職]?(\d+)\s*[Pp段]?", s)
    return 10 + int(m.group(1)) if m and ("P" in s or "p" in s or "职" in s or "職" in s) else 0


def ingest(a):
    import py7zr
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    salt_file = Path(a.salt_file)
    if not salt_file.exists():
        salt_file.write_bytes(os.urandom(32))
        salt_file.chmod(0o600)
    salt = salt_file.read_bytes()
    key_of = lambda name: int.from_bytes(hashlib.blake2b(name, key=salt, digest_size=8).digest(), "little")  # noqa: E731
    files = sorted(Path(a.archives).iterdir())
    groups = {}
    for f in files:  # x.7z -> one archive; x.7z.001, x.7z.002 ... -> one archive split in volumes
        m = re.fullmatch(r"(.+\.7z)(\.\d+)?", f.name)
        if m:
            groups.setdefault(m.group(1), []).append(f)
    part, n_all, n_bad = 0, 0, 0
    for name, vols in groups.items():
        src = vols[0]
        if len(vols) > 1 or vols[0].name != name:  # join the volumes (plain concatenation) next to the output, removed afterwards
            src = out / f"_{name}"
            with open(src, "wb") as w:
                for v in sorted(vols):
                    w.write(v.read_bytes())
        tmp = Path(a.tmp) / f"extfox_{os.getpid()}_{name}"  # RAM disk of the node: half a million small files do not go to the network file system
        try:
            with py7zr.SevenZipFile(src) as z:
                names = [n for n in z.getnames() if n.endswith(".sgf")]
                bad = [n for n in z.getnames() if n.startswith("/") or ".." in n.split("/")]
                if bad:
                    raise ValueError(f"{name}: entries leave the target folder, e.g. {bad[:3]}")
                tmp.mkdir(parents=True)
                z.extractall(tmp)
            print(f"{name}: {len(names)} records unpacked to {tmp} ({time.time() - t0:.0f} s)", flush=True)
            for i in range(0, len(names), a.chunk):
                rec = {k: [] for k in ("key", "pb", "pw", "br", "wr", "year", "ha", "tm", "tc", "tt", "res", "len", "off")}
                mv_all, off = [], 0
                for n in names[i:i + a.chunk]:
                    raw = (tmp / n).read_bytes()
                    head = dict(PROP.findall(re.split(rb";\s*[BW]\[", raw, maxsplit=1)[0]))
                    mv = moves_of(raw)
                    if head.get(b"SZ", b"19") != b"19" or len(mv) < a.min_moves or b"PB" not in head or b"PW" not in head:
                        n_bad += 1
                        continue
                    num = lambda k: int(head[k]) if head.get(k, b"").lstrip(b"-").isdigit() else -1  # noqa: E731
                    re_ = head.get(b"RE", b"")
                    rec["key"].append(n); rec["pb"].append(key_of(head[b"PB"])); rec["pw"].append(key_of(head[b"PW"]))
                    rec["br"].append(rank_code(head.get(b"BR", b""))); rec["wr"].append(rank_code(head.get(b"WR", b"")))
                    rec["year"].append(int(head[b"DT"][:4]) if head.get(b"DT", b"")[:4].isdigit() else 0)
                    rec["ha"].append(min(num(b"HA"), 99)); rec["tm"].append(num(b"TM")); rec["tc"].append(min(num(b"TC"), 30000)); rec["tt"].append(min(num(b"TT"), 30000))
                    rec["res"].append(1 if b"+R" in re_ else 2 if b"+T" in re_ else 3 if b"+" in re_ else 0)  # resign / time / points / other
                    rec["len"].append(len(mv)); rec["off"].append(off)
                    mv_all.extend(c * 512 + p for c, p in mv)
                    off += len(mv)
                np.savez(out / f"games_{part:04d}.npz", key=np.array(rec["key"]), pb=np.array(rec["pb"], np.uint64), pw=np.array(rec["pw"], np.uint64), br=np.array(rec["br"], np.int8),
                         wr=np.array(rec["wr"], np.int8), year=np.array(rec["year"], np.int16), ha=np.array(rec["ha"], np.int8), tm=np.array(rec["tm"], np.int32), tc=np.array(rec["tc"], np.int16),
                         tt=np.array(rec["tt"], np.int16), res=np.array(rec["res"], np.int8), len=np.array(rec["len"], np.int32), off=np.array(rec["off"], np.int64), moves=np.array(mv_all, np.uint16))
                part += 1
                n_all += len(rec["key"])
                print(f"  part {part}: {n_all} games kept, {n_bad} left out ({time.time() - t0:.0f} s)", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if src.name.startswith("_"):
            src.unlink()
    print(f"ingest done: {n_all} games in {part} parts, {n_bad} left out (not 19x19, under {a.min_moves} moves, or no account) ({time.time() - t0:.0f} s)")


def overlap(a):
    import multiprocessing as mp

    import pandas as pd
    t0 = time.time()
    ext, n_ext = set(), 0
    for d in a.parts:
        for f in sorted(Path(d).glob("games_*.npz")):
            z = np.load(f)
            mv, off, ln = z["moves"], z["off"], z["len"]
            for o, n in zip(off, ln):
                h = prefix_hash([(int(v) // 512, int(v) % 512) for v in mv[o:o + a.prefix]], a.prefix) if n >= a.prefix else None
                n_ext += 1
                if h is not None:
                    ext.add(h)
        print(f"external {d}: {n_ext} games so far, {len(ext)} distinct prefixes ({time.time() - t0:.0f} s)", flush=True)
    folds = pd.read_csv(Path(a.aicup1) / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
    res = {"prefix_moves": a.prefix, "external_games": n_ext, "external_distinct_prefixes": len(ext), "by_file": {}, "note": "counts only; fold 0 excluded; test sets not read; no account read for a match"}
    tot = hit = 0
    for f in sorted((Path(a.aicup1) / "data/training").glob("train_*.csv")):
        c = pd.read_csv(f, usecols=["player_id", "sgf_content"], dtype=str)
        c = c[c.player_id.map(folds).fillna(-1) != 0]
        with mp.get_context("fork").Pool(a.workers) as pool:
            hs = pool.map(_h, [(s, a.prefix) for s in c.sgf_content], chunksize=2000)
        hs = [h for h in hs if h is not None]
        m = sum(h in ext for h in hs)
        res["by_file"][f.name] = {"games_with_prefix": len(hs), "matches": int(m)}
        tot += len(hs); hit += m
        print(f"{f.name}: {len(hs)} games with {a.prefix} moves, {m} with an external game of the same first {a.prefix} moves ({time.time() - t0:.0f} s)", flush=True)
    res.update(competition_games=tot, matches=int(hit))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"OVERLAP: {hit} of {tot} competition training games match one of {len(ext)} external prefixes -> {'STOP: shared games' if hit else 'no shared game in these parts'}; wrote {a.out}")


def _h(x):
    return prefix_hash(moves_of(x[0].encode()), x[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("ingest", "overlap"))
    ap.add_argument("--archives")
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--salt-file", default="/work/u2507115/ext_fox/account_salt.bin", help="local secret for the account keys; never in the repository")
    ap.add_argument("--tmp", default="/dev/shm", help="where an archive is unpacked while it is read (removed afterwards)")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--min-moves", type=int, default=30)
    ap.add_argument("--prefix", type=int, default=60)
    ap.add_argument("--aicup1", default=str(Path.home() / "BK_AICUP_2026-1"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    return ingest(a) if a.stage == "ingest" else overlap(a)


if __name__ == "__main__":
    sys.exit(main())
