#!/usr/bin/env python3
"""METHOD_NOTES 2.188: the scale4 round and the verdict of the merged retrain as single-node steps INSIDE one multi-node job.

Since 2026-11-17 (about 13:00) the small partitions take no new jobs (a group cap of 64 GPUs for all users) and the larger
ones want more than one node per job. This driver runs on the first node of a multi-node job (scripts/nano4/run_multi.sbatch)
and starts every task as an srun step on one free node (8 GPUs, 96 cores):

  selection of scale4 (job given by --select-out, already running)  ->  10 extraction steps (6 x ply, 4 x sup)  ->  check
  ->  8 trainings (o36-o39)  ->  4 x family vectors (FAST)  ->  'all' (candidates v2fin2_x / v2fin2_a)
  and, as soon as the trainings of o32-o35 (their own jobs) have written their checkpoints:
  4 x family vectors (FAST)  ->  'all' (candidates v2fin_x / v2fin_a)

A step that ends with an error is started again on another node (at most three starts); a node on which a step failed
within ten minutes is not used again; a training that has not finished loading within --load-limit minutes is stopped and
started elsewhere. Every step gets its own job-id string (the scripts name their logs and temporary folders after it).
"""
import argparse
import os
import subprocess
import sys
import time

R2 = "/home/u2507115/BK_AICUP_2026-2"
X = "/work/u2507115/ext_fox"
S = "/work/u2507115/pid_v2/seqnet"
N4 = R2 + "/scripts/nano4"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def ckpts(fam):
    """The eight checkpoint files of a family of the merged retrain / the scale4 round."""
    if fam in ("o32", "o36"):
        return [f"{S}/ckpt_-1_{fam}Ss{i}e.pt" for i in range(4)] + [f"{S}/ckpt_-1_{fam}SMs{i}e.pt" for i in range(4)]
    return [f"{S}/ckpt_-1_{fam}As{i}e.pt" for i in range(6)] + [f"{S}/ckpt_-1_{fam}Cs{i}e.pt" for i in range(2)]


def written(files, age=90):
    return all(os.path.exists(f) and os.path.getsize(f) > 0 and time.time() - os.path.getmtime(f) > age for f in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-out", required=True, help="the .out file of the selection job: the extraction starts when it has its 'selection:' line")
    ap.add_argument("--load-limit", type=float, default=30.0)
    ap.add_argument("--skip", nargs="*", default=[], help="tasks counted as done (a restart)")
    a = ap.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    nodes = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], capture_output=True, text=True).stdout.split()
    log(f"driver on {os.uname().nodename}; job {job}; {len(nodes)} nodes: {' '.join(nodes)}")

    def selected():
        try:
            return os.path.exists(f"{X}/scale4/select.csv") and any(line.startswith("selection:") for line in open(a.select_out))
        except OSError:
            return False

    T = {}  # name -> dict(cmd, env, needs, ready, prio, train)

    def add(name, cmd, needs=(), env="", ready=None, prio=5, train=False):
        T[name] = dict(cmd=cmd, env=env, needs=list(needs), ready=ready, prio=prio, train=train, status="waiting", starts=0, node=None, proc=None, t0=0.0, log=None)

    for k in (0, 8, 16, 24, 32, 40):
        add(f"ply{k}", f"bash {N4}/ext_fox_scale4.sh ply {k} {X}/scale4/ply", ready=selected, prio=1)
    for k in (0, 8, 16, 24):
        add(f"sup{k}", f"bash {N4}/ext_fox_scale4.sh sup {k} {X}/scale4/supj", ready=selected, prio=1)
    ext = list(T)
    add("check", f"bash {N4}/ext_fox_scale4.sh check {X}/scale4/ply {X}/scale4/supj", needs=ext, prio=1)
    for fam in ("o36", "o37", "o38", "o39"):
        for g in (0, 1):
            add(f"{fam}_g{g}", f"bash {N4}/pid_exp_s4w_train.sh {fam} {g}", needs=["check"], env="WIN=9 MIN_FREE_GB=0", prio=2, train=True)
        add(f"vec_{fam}", f"bash {N4}/pid_exp_s4w_chain.sh {fam}", needs=[f"{fam}_g0", f"{fam}_g1"], env="FAST=1", prio=3)
    add("fin2", f"bash {N4}/pid_exp_s4w_chain.sh all", needs=[f"vec_{f}" for f in ("o36", "o37", "o38", "o39")], prio=3)
    for fam in ("o32", "o33", "o34", "o35"):
        add(f"vec_{fam}", f"bash {N4}/pid_exp_s3bw_chain.sh {fam}", env="FAST=1", ready=(lambda f=fam: written(ckpts(f))), prio=4)
    add("fin1", f"bash {N4}/pid_exp_s3bw_chain.sh all", needs=[f"vec_{f}" for f in ("o32", "o33", "o34", "o35")], prio=4)
    for n in a.skip:
        T[n]["status"] = "done"
    bad, serial = set(), 0

    def start(name, node):
        nonlocal serial
        t = T[name]
        serial += 1
        fake = f"{job}{serial:02d}"
        t.update(status="running", node=node, t0=time.time(), starts=t["starts"] + 1, log=f"logs/nano4/s4-{job}-{name}-{t['starts'] + 1}.out")
        inner = f"export SLURM_JOB_ID={fake} {t['env']}; exec {t['cmd']}"
        t["proc"] = subprocess.Popen(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=96", "--gres=gpu:8", f"--nodelist={node}",
                                      f"--output={t['log']}", f"--error={t['log']}", "bash", "-c", inner])
        log(f"start {name} on {node} (start {t['starts']}, id {fake}) -> {t['log']}")

    last = ""
    while True:
        now = time.time()
        for name, t in T.items():
            if t["status"] != "running":
                continue
            rc = t["proc"].poll()
            if rc is None and t["train"] and now - t["t0"] > a.load_limit * 60:
                try:
                    loaded = any("trainer " in line for line in open(t["log"]))
                except OSError:
                    loaded = False
                if not loaded:
                    log(f"{name} on {t['node']} has not finished loading after {a.load_limit:.0f} min: stopped")
                    t["proc"].terminate()
                    try:
                        t["proc"].wait(120)
                    except subprocess.TimeoutExpired:
                        t["proc"].kill()
                    rc = 1
                    bad.add(t["node"])
            if rc is None:
                continue
            mins = (now - t["t0"]) / 60
            if rc == 0:
                t["status"] = "done"
                log(f"done {name} on {t['node']} ({mins:.0f} min)")
            else:
                if mins < 10:
                    bad.add(t["node"])
                tail = subprocess.run(f"tail -n 4 {t['log']} | cut -c1-200", shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
                if t["starts"] >= 3:
                    t["status"] = "failed"
                    log(f"ALERT {name} failed {t['starts']} times (last on {t['node']}, {mins:.0f} min): {tail}")
                else:
                    t["status"] = "waiting"
                    log(f"{name} ended with an error on {t['node']} after {mins:.0f} min, will start again: {tail}")
        busy = {t["node"] for t in T.values() if t["status"] == "running"}
        free = [n for n in nodes if n not in busy and n not in bad]
        for name, t in sorted(T.items(), key=lambda kv: kv[1]["prio"]):
            if not free:
                break
            if t["status"] != "waiting" or not all(T[n]["status"] == "done" for n in t["needs"]):
                continue
            if t["ready"] is not None and not t["ready"]():
                continue
            node = next((n for n in free if n != t["node"]), free[0])  # not the node of its last start when another is free
            free.remove(node)
            start(name, node)
        summary = " ".join(f"{k}:{sum(1 for t in T.values() if t['status'] == k)}" for k in ("waiting", "running", "done", "failed"))
        summary += f" | bad nodes: {' '.join(sorted(bad)) or '-'}"
        if summary != last:
            log("tasks " + summary + " | running: " + " ".join(f"{n}@{t['node'][-3:]}" for n, t in T.items() if t["status"] == "running"))
            last = summary
        def dead(n, seen=()):  # can never run: failed, or waiting for something that can never run
            t_ = T[n]
            return t_["status"] == "failed" or (t_["status"] == "waiting" and any(dead(m) for m in t_["needs"]))
        if all(t["status"] == "done" or dead(n) for n, t in T.items()):
            log("all tasks have ended"); break
        time.sleep(20)
    failed = [n for n, t in T.items() if t["status"] != "done"]
    log("not done: " + (" ".join(failed) or "-"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
