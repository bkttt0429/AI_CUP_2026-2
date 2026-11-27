#!/usr/bin/env python3
"""METHOD_NOTES 2.187: the scale4 round's jobs, submitted in order within the queue limits (login node: submit and query only).

  selection (already submitted)  ->  10 extraction jobs (6 x ply, 4 x sup)  ->  check  ->  8 trainings (o36-o39)
  ->  4 x family vectors (FAST)  ->  the 'all' job (candidates v2fin2_x / v2fin2_a)

A task is submitted when the tasks it needs are COMPLETED and the partition has room (submitted jobs of this session, names
p_*: at most 10 on 8gpus - three are left to the rank session - and 9 on 16gpus). A job that ends in another state is sent
again, up to three times, with its node added to the exclude list. State in results/final/s4_orchestrate_state.json: the
script can be stopped and started again.
  python3 scripts/pid_s4_orchestrate.py --select-job <id> [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = "/home/u2507115/BK_AICUP_2026-1"
R2 = "/home/u2507115/BK_AICUP_2026-2"
X = "/work/u2507115/ext_fox"
STATE = R2 + "/results/final/s4_orchestrate_state.json"
EXCLUDE = [1, 3, 4, 5, 6, 7, 8, 9, 10, 39, 62, 69, 82, 83, 145, 146, 147, 155, 159, 161, 166, 167, 177]
LIMIT = {"8gpus": 10, "16gpus": 9}
RUNCAP = {"8gpus": 8, "16gpus": 4}
BAD = ("FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def tasks():
    t = {}
    for k in (0, 8, 16, 24, 32, 40):
        t[f"ply{k}"] = dict(cmd=f"bash {R2}/scripts/nano4/ext_fox_scale4.sh ply {k} {X}/scale4/ply", gpus=8, time="02:00:00", needs=["select"], env="")
    for k in (0, 8, 16, 24):
        t[f"sup{k}"] = dict(cmd=f"bash {R2}/scripts/nano4/ext_fox_scale4.sh sup {k} {X}/scale4/supj", gpus=8, time="02:00:00", needs=["select"], env="")
    ext = [n for n in t]
    t["check"] = dict(cmd=f"bash {R2}/scripts/nano4/ext_fox_scale4.sh check {X}/scale4/ply {X}/scale4/supj", gpus=1, time="00:30:00", needs=ext, env="", part="8gpus")
    for fam in ("o36", "o37", "o38", "o39"):
        for g in (0, 1):
            t[f"{fam}_g{g}"] = dict(cmd=f"bash {R2}/scripts/nano4/pid_exp_s4w_train.sh {fam} {g}", gpus=8, time="10:00:00", needs=["check"], env="WIN=9")
        t[f"vec_{fam}"] = dict(cmd=f"bash {R2}/scripts/nano4/pid_exp_s4w_chain.sh {fam}", gpus=8, time="02:00:00", needs=[f"{fam}_g0", f"{fam}_g1"], env="FAST=1")
    t["chain_v2fin2"] = dict(cmd=f"bash {R2}/scripts/nano4/pid_exp_s4w_chain.sh all", gpus=8, time="04:00:00", needs=[f"vec_{f}" for f in ("o36", "o37", "o38", "o39")], env="")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-job", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    T = tasks()
    st = json.load(open(STATE)) if os.path.exists(STATE) and not a.dry else {}
    st.setdefault("exclude", list(EXCLUDE))
    st.setdefault("task", {})
    for n in T:
        st["task"].setdefault(n, {"status": "waiting", "job": None, "tries": 0})
    st["task"]["select"] = {"status": "submitted", "job": a.select_job, "tries": 1}

    def save():
        if not a.dry:
            json.dump(st, open(STATE + ".tmp", "w"), indent=1)
            os.replace(STATE + ".tmp", STATE)

    def counts(part):
        names = sh(f"squeue -u u2507115 -h -p {part} -o '%j %t'").splitlines()
        mine = [x for x in names if x.startswith("p_")]
        return len(mine), sum(1 for x in names if x.split()[-1] == "R")

    def submit(name):
        d = T[name]
        order = [d["part"]] if d.get("part") else sorted(RUNCAP, key=lambda p: (counts(p)[1] >= RUNCAP[p], counts(p)[0] / LIMIT[p]))
        for part in order:
            if d["gpus"] < 8 and part == "16gpus":
                continue
            if counts(part)[0] >= LIMIT[part]:
                continue
            ex = "25a-hgpn[" + ",".join(f"{x:03d}" for x in sorted(set(st["exclude"]))) + "]"
            cmd = (f"{d['env']} sbatch --parsable -p {part} --gres=gpu:{d['gpus']} -c {12 * d['gpus']} -t {d['time']} -J p_s4_{name} "
                   f"--exclude='{ex}' scripts/nano4/run.sbatch {d['cmd']}")
            if a.dry:
                log(f"DRY {name} -> {part}: {cmd[:150]}")
                return "dry"
            out = sh(cmd)
            if out.split(";")[0].isdigit():
                log(f"submitted {name}: job {out} on {part} (try {st['task'][name]['tries'] + 1})")
                return out.split(";")[0]
            log(f"sbatch refused {name} on {part}: {out[:200]}")
        return None

    last = ""
    while True:
        for name, s in st["task"].items():
            if s["status"] != "submitted":
                continue
            row = sh(f"sacct -j {s['job']} -X -n -P -o State,NodeList,Elapsed").split("\n")[0].split("|")
            state = row[0].split()[0] if row and row[0] else ""
            if state == "COMPLETED":
                if name == "select" and not os.path.exists(f"{X}/scale4/select.csv"):
                    s["status"] = "failed"; log("ALERT: the selection job completed but there is no select.csv")
                elif name == "check" and not os.path.exists(f"{X}/scale4/CHECK_PASSED"):
                    s["status"] = "failed"; log("ALERT: the check job completed but CHECK_PASSED is missing")
                else:
                    s["status"] = "done"; log(f"done: {name} (job {s['job']}, {row[2] if len(row) > 2 else ''})")
            elif any(state.startswith(b) for b in BAD):
                node = row[1] if len(row) > 1 else ""
                if node.startswith("25a-hgpn") and node[8:].isdigit():
                    st["exclude"].append(int(node[8:]))
                if name == "select" or s["tries"] >= 3:
                    s["status"] = "failed"; log(f"ALERT: {name} (job {s['job']}) ended {state} on {node}; not sent again")
                else:
                    s["status"] = "waiting"; log(f"{name} (job {s['job']}) ended {state} on {node}: will be sent again without that node")
        for name in T:
            s = st["task"][name]
            if s["status"] == "waiting" and all(st["task"][n]["status"] == "done" for n in T[name]["needs"]):
                job = submit(name)
                if job == "dry":
                    s["status"] = "done"
                elif job:
                    s.update(status="submitted", job=job, tries=s["tries"] + 1)
        save()
        summary = " ".join(f"{k}:{sum(1 for s in st['task'].values() if s['status'] == k)}" for k in ("waiting", "submitted", "done", "failed"))
        if summary != last:
            log("tasks " + summary); last = summary
        if st["task"]["chain_v2fin2"]["status"] in ("done", "failed"):
            log("the last job has ended: " + st["task"]["chain_v2fin2"]["status"]); return 0
        blocked = [n for n, s in st["task"].items() if s["status"] == "failed"]
        if blocked and not any(s["status"] == "submitted" for s in st["task"].values()):
            log(f"stopped: failed tasks {blocked} and nothing is running"); return 1
        if a.dry:
            if all(s["status"] == "done" for s in st["task"].values() if s is not st["task"]["select"]):
                return 0
            st["task"]["select"]["status"] = "done"
            continue
        time.sleep(45)


if __name__ == "__main__":
    sys.exit(main())
