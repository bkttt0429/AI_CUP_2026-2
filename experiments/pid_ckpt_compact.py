#!/usr/bin/env python
"""Halve the checkpoints written before pid_seq_train.py stored one copy of the weights (METHOD_NOTES 2.101).

The older files hold the same state twice ("best_state" and "net": two separate copies of identical tensors). Every file of
the directory that is not a link and holds two EQUAL states is rewritten with one copy under both names (what the trainer
writes now); every other key is kept. A file is replaced only after the rewritten one has been read back and compared with the
original state tensor by tensor; files whose two states differ, or that have only one of them, are left alone.
"""
import argparse
import os
import sys
from pathlib import Path

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--min-mb", type=float, default=100.0, help="files below this size are not opened")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    done = skipped = 0
    saved = 0
    for f in sorted(Path(a.dir).glob("ckpt_*.pt")):
        if f.is_symlink() or f.stat().st_size < a.min_mb * 1e6:
            continue
        st = torch.load(f, map_location="cpu", weights_only=False)
        b, n = st.get("best_state"), st.get("net")
        if not (isinstance(b, dict) and isinstance(n, dict)) or b.keys() != n.keys():
            print(f"  {f.name}: left alone (keys {sorted(st.keys())})", flush=True)
            skipped += 1
            continue
        if all(x.data_ptr() == n[k].data_ptr() for k, x in b.items() if x.numel()):
            skipped += 1  # one copy already
            continue
        if not all(torch.equal(x, n[k]) for k, x in b.items()):
            print(f"  {f.name}: left alone (the two states differ)", flush=True)
            skipped += 1
            continue
        before = f.stat().st_size
        if a.dry:
            print(f"  {f.name}: two equal states, {before / 1e6:.0f} MB", flush=True)
            continue
        tmp = f.with_name(f.name + ".compact_tmp")
        torch.save({**st, "net": b}, tmp)
        back = torch.load(tmp, map_location="cpu", weights_only=False)
        ok = (back.keys() == st.keys() and all(torch.equal(x, back["best_state"][k]) and torch.equal(x, back["net"][k]) for k, x in b.items())
              and all(repr(back[k]) == repr(st[k]) for k in st if k not in ("best_state", "net")))
        if not ok:
            tmp.unlink()
            print(f"  {f.name}: read-back differs, left alone", flush=True)
            skipped += 1
            continue
        os.replace(tmp, f)
        after = f.stat().st_size
        saved += before - after
        done += 1
        print(f"  {f.name}: {before / 1e6:.0f} -> {after / 1e6:.0f} MB", flush=True)
    print(f"{done} files rewritten, {skipped} left as they were, {saved / 1e9:.2f} GB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
