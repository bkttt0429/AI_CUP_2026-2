#!/usr/bin/env python
"""Lossless re-compression of .npz archives in place (METHOD_NOTES 2.163).

Every large feature archive of this project was written with np.savez: the members are "Stored" (no compression).
np.load reads a compressed archive exactly as an uncompressed one, so re-writing the same arrays with deflate frees
space without touching a single value and without changing any reader.
  --sample N   measure only: N evenly spaced files per directory, the size under each method and the decompression
               speed; nothing is written
  default      convert: per file  load every array  ->  write a hidden temporary file in the same directory
               (.<name>.rctmp: no "*.npz" pattern of any loader matches it)  ->  load the temporary file again and compare
               member order, dtype, shape and every byte with the original arrays  ->  check that the original has not
               changed meanwhile  ->  os.replace (atomic; a reader gets the old or the new file, both complete)  ->
               restore the modification time and the mode.
Left alone: symbolic links, files with more than one hard link (re-writing would un-share them), archives that are
already compressed, archives with object arrays, files that would shrink by less than --min-gain.
The manifest (--manifest, JSON lines) records per file the old and the new size and a SHA-256 of every array (dtype,
shape, bytes): from now on a file of these directories is identified by its content, not by its size.
Safe to interrupt and to run again (converted files are skipped). From any directory; CPU only.
"""
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

METHODS = {"deflate": zipfile.ZIP_DEFLATED, "bz2": zipfile.ZIP_BZIP2, "lzma": zipfile.ZIP_LZMA}
CFG = {}


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_npz(target, arrays, method, level):
    """As np.savez_compressed (zip members <key>.npy), with the compression method given."""
    kw = {} if method == "lzma" else {"compresslevel": level}
    with zipfile.ZipFile(target, "w", compression=METHODS[method], allowZip64=True, **kw) as zf:
        for k, a in arrays.items():
            with zf.open(k + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(a), allow_pickle=False)


def load_all(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def convert(path):
    path = Path(path)
    out = {"path": str(path)}
    try:
        st = os.lstat(path)
        out["old"] = st.st_size
        if not stat.S_ISREG(st.st_mode):
            return {**out, "status": "skipped", "why": "not a regular file"}
        if st.st_nlink > 1:
            return {**out, "status": "skipped", "why": "hard-linked"}
        with zipfile.ZipFile(path) as zf:
            if all(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist()):
                return {**out, "status": "skipped", "why": "already compressed"}
        try:
            arrays = load_all(path)
        except ValueError as e:  # object arrays need pickle: left alone
            return {**out, "status": "skipped", "why": f"not loadable without pickle ({str(e)[:60]})"}
        sums = {k: digest(a) for k, a in arrays.items()}
        # The quota of this file system gives the space of a replaced file back only after some seconds (job 787891:
        # 8.5 GiB re-written in 12 s took the free space from 7.3 to 0.7 GiB; it was back at 9.3 GiB half a minute
        # later). So: below the floor every worker WAITS for the space to come back; it never writes into the reserve.
        waited = 0
        while shutil.disk_usage(path.parent).free < CFG["min_free"]:
            if waited >= CFG["max_wait"]:
                return {**out, "status": "failed", "why": "free space stayed below the floor"}
            time.sleep(5)
            waited += 5
        tmp = path.with_name(f".{path.name}.rctmp")
        try:
            write_npz(tmp, arrays, CFG["method"], CFG["level"])
            new = os.stat(tmp).st_size
            if new > st.st_size * (1 - CFG["min_gain"]):
                os.unlink(tmp)
                return {**out, "status": "kept", "new": new, "why": "no gain"}
            back = load_all(tmp)
            if list(back) != list(arrays) or any(digest(back[k]) != sums[k] for k in arrays):
                os.unlink(tmp)
                return {**out, "status": "failed", "why": "the re-written archive differs"}
            now = os.lstat(path)
            if (now.st_ino, now.st_size, now.st_mtime_ns) != (st.st_ino, st.st_size, st.st_mtime_ns):
                os.unlink(tmp)
                return {**out, "status": "skipped", "why": "changed while it was read"}
            os.chmod(tmp, stat.S_IMODE(st.st_mode))
            os.replace(tmp, path)
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        finally:
            if tmp.exists():
                os.unlink(tmp)
        return {**out, "status": "converted", "new": new, "arrays": {k: {"dtype": arrays[k].dtype.str, "shape": list(arrays[k].shape), "sha256": sums[k]} for k in arrays}}
    except Exception as e:  # noqa: BLE001
        return {**out, "status": "failed", "why": f"{type(e).__name__}: {str(e)[:120]}"}


def sample_one(path):
    """Sizes of one archive under every method (in memory) and the time to read it back."""
    arrays = load_all(path)
    raw = sum(a.nbytes for a in arrays.values())
    res = {"stored": os.path.getsize(path), "raw": raw, "by_array": {k: [int(a.nbytes), len(zlib_bytes(a))] for k, a in arrays.items()}}
    for name, method, level in (("deflate1", "deflate", 1), ("deflate6", "deflate", 6), ("deflate9", "deflate", 9), ("bz2", "bz2", 9), ("lzma", "lzma", None)):
        buf = io.BytesIO()
        t = time.time()
        write_npz(buf, arrays, method, level)
        tw = time.time() - t
        buf.seek(0)
        t = time.time()
        with np.load(buf, allow_pickle=False) as z:
            for k in z.files:
                z[k]
        res[name] = [buf.getbuffer().nbytes, tw, time.time() - t]
    return res


def zlib_bytes(a):
    import zlib
    return zlib.compress(np.ascontiguousarray(a).tobytes(), 6)


def files_of(d, pattern):
    d = Path(d)
    return sorted(str(f) for f in d.glob(pattern) if not f.name.startswith(".") and not f.name.endswith(".tmp.npz"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--method", choices=sorted(METHODS), default="deflate")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--min-gain", type=float, default=0.03, help="leave a file alone when it would shrink by less than this share")
    ap.add_argument("--min-free-mb", type=int, default=5000, help="floor of the free space: below it the workers wait (the space of replaced files comes back with a delay)")
    ap.add_argument("--max-wait", type=int, default=1800, help="seconds a worker waits for the floor before it gives the file up")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    CFG.update(method=a.method, level=a.level, min_gain=a.min_gain, min_free=a.min_free_mb * 2 ** 20, max_wait=a.max_wait)
    if a.sample:
        with mp.get_context("fork").Pool(min(a.workers, 16)) as pool:
            for d in a.dirs:
                fs = files_of(d, a.pattern)
                if not fs:
                    print(f"{d}: no files", flush=True)
                    continue
                total = sum(os.path.getsize(f) for f in fs)
                pick = [fs[i] for i in sorted({int(j * (len(fs) - 1) / max(a.sample - 1, 1)) for j in range(min(a.sample, len(fs)))})]
                rs = pool.map(sample_one, pick)
                st = sum(r["stored"] for r in rs)
                line = f"{d}: {len(fs)} files, {total / 2 ** 30:.2f} GiB; sample of {len(pick)} ({st / 2 ** 20:.0f} MiB):"
                for name in ("deflate1", "deflate6", "deflate9", "bz2", "lzma"):
                    sz = sum(r[name][0] for r in rs)
                    line += f"  {name} {sz / st:.3f} (write {st / 2 ** 20 / max(sum(r[name][1] for r in rs), 1e-9):.0f}, read {st / 2 ** 20 / max(sum(r[name][2] for r in rs), 1e-9):.0f} MiB/s)"
                print(line, flush=True)
                keys = list(rs[0]["by_array"])
                print("      by array (deflate 6): " + "  ".join(f"{k} {sum(r['by_array'][k][0] for r in rs if k in r['by_array']) / 2 ** 20:.1f} -> {sum(r['by_array'][k][1] for r in rs if k in r['by_array']) / 2 ** 20:.1f} MiB" for k in keys)
                      + f" | would free about {total * (1 - sum(r['deflate6'][0] for r in rs) / st) / 2 ** 30:.2f} GiB", flush=True)
        return 0
    man = open(a.manifest, "a") if a.manifest else None
    bad = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for d in a.dirs:
            fs = files_of(d, a.pattern)
            before = shutil.disk_usage(d).free if fs else 0
            tally, old, new = {}, 0, 0
            for r in pool.imap_unordered(convert, fs, chunksize=4):
                tally[r["status"]] = tally.get(r["status"], 0) + 1
                if r["status"] == "converted":
                    old += r["old"]
                    new += r["new"]
                if r["status"] == "failed":
                    bad += 1
                    print(f"  FAILED {r['path']}: {r.get('why')}", flush=True)
                if man:
                    man.write(json.dumps(r) + "\n")
            if man:
                man.flush()
            why = {}
            print(f"{d}: {len(fs)} files -> {tally}; converted {old / 2 ** 30:.2f} -> {new / 2 ** 30:.2f} GiB (freed {(old - new) / 2 ** 30:.2f} GiB, ratio {new / max(old, 1):.3f}); "
                  f"free space of the file system {before / 2 ** 30:.1f} -> {shutil.disk_usage(d).free / 2 ** 30:.1f} GiB ({time.time() - t0:.0f} s)", flush=True)
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
