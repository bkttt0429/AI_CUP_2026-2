#!/usr/bin/env python
"""How two submission files differ, per pool (the prefix of question_id): questions whose Top-1 differs and questions whose
five answers differ in any place. Also the first 12 hex digits of each file's MD5 (the name the notes use for a file)."""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


def read(p):
    with open(p, newline="") as fh:
        return {r["question_id"]: [r[f"top_{i}"] for i in range(1, 6)] for r in csv.DictReader(fh)}


def main(argv=None):
    a, b = (Path(x) for x in (argv or sys.argv[1:])[:2])
    A, B = read(a), read(b)
    assert A.keys() == B.keys(), "the two files answer different questions"
    top1, any_ = Counter(), Counter()
    for q in A:
        pool = q.split("_q_")[0]
        top1[pool] += A[q][0] != B[q][0]
        any_[pool] += A[q] != B[q]
    for p in (a, b):
        print(f"{p.name}: {hashlib.md5(p.read_bytes()).hexdigest()[:12]}, {len(read(p))} questions")
    print("questions with a different Top-1: " + ", ".join(f"{k} {top1[k]}" for k in sorted(any_)) + f" (all {sum(top1.values())})")
    print("questions with any different answer: " + ", ".join(f"{k} {any_[k]}" for k in sorted(any_)) + f" (all {sum(any_.values())})")


if __name__ == "__main__":
    sys.exit(main())
