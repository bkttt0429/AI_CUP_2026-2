#!/usr/bin/env python
"""How two submission files differ, per pool (the prefix of question_id): questions whose Top-1 differs and questions whose
five answers differ in any place. Also the first 12 hex digits of each file's MD5 (the name the notes use for a file)."""
import csv  # 讀取提交檔
import hashlib  # 計算檔案的 MD5
import sys  # 命令列參數與結束碼
from collections import Counter  # 逐題池計數
from pathlib import Path  # 以路徑物件處理檔案


def read(p):  # 讀入一個提交檔
    with open(p, newline="") as fh:  # 依 csv 模組的要求開檔
        return {r["question_id"]: [r[f"top_{i}"] for i in range(1, 6)] for r in csv.DictReader(fh)}  # 題目編號 → 前 5 名


def main(argv=None):  # 命令列進入點
    a, b = (Path(x) for x in (argv or sys.argv[1:])[:2])  # 要比較的兩個檔
    A, B = read(a), read(b)  # 讀入兩個檔的答案
    assert A.keys() == B.keys(), "the two files answer different questions"  # 兩個檔必須回答同一組題目
    top1, any_ = Counter(), Counter()  # 各題池 Top-1 不同與任一答案不同的題數
    for q in A:  # 逐題比較
        pool = q.split("_q_")[0]  # 題池＝question_id 的前綴
        top1[pool] += A[q][0] != B[q][0]  # 第一名不同
        any_[pool] += A[q] != B[q]  # 五個答案有任何一處不同
    for p in (a, b):  # 逐檔列出識別資訊
        print(f"{p.name}: {hashlib.md5(p.read_bytes()).hexdigest()[:12]}, {len(read(p))} questions")  # MD5 前 12 碼與題數
    print("questions with a different Top-1: " + ", ".join(f"{k} {top1[k]}" for k in sorted(any_)) + f" (all {sum(top1.values())})")  # 各題池 Top-1 不同的題數與合計
    print("questions with any different answer: " + ", ".join(f"{k} {any_[k]}" for k in sorted(any_)) + f" (all {sum(any_.values())})")  # 各題池任一答案不同的題數與合計


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
