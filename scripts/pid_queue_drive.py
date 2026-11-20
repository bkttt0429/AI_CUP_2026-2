#!/usr/bin/env python3
"""METHOD_NOTES 2.193: a task-queue driver for a multi-node job (scripts/nano4/run_multi.sbatch).

pid_s4_drive.py has its task graph built in; steps added later had to be started with srun from the login node and would
have died with the session. This driver runs on the first node of the job and takes its tasks from a directory, so tasks
can be added while the job runs (the login node only writes small files):

  <dir>/<name>.task     JSON {"cmd": "bash ...", "env": "A=1 B=2", "needs": ["other task", ...], "prio": 5, "starts": 1,
                              "node": "25a-hgpn088"}      only "cmd" is required; "starts" = how often it may be started
                        (after an error it goes to another node); "node" pins it to one node of the job
  <dir>/<name>.cancel   stop that task (a running step is terminated, a waiting one is dropped)
  <dir>/BAD_NODES       node names (white space separated) that get no further steps
  <dir>/STOP            no further starts; the driver ends when the running steps have ended
  <dir>/status.txt      written by the driver: one line per task

Every task is ONE srun step on ONE free node (8 GPUs, 96 cores) with its own job-id string (SLURM_JOB_ID = <job><serial>;
the scripts name their logs and temporary folders after it); its output goes to logs/nano4/q-<job>-<name>-<start>.out.
A task waits until all of its "needs" are done; when one of them failed or was cancelled it is marked blocked. Lower
"prio" starts first. The driver ends - and with it the job - when nothing has been running or waiting for --idle-exit
minutes (0 = never), so an empty queue does not hold the nodes.

  sbatch -p 128gpus -N <nodes> --time=<limit> scripts/nano4/run_multi.sbatch python3 -u <this file> --dir <dir>
  --dry-run: no srun; every command is run as given on this machine (for a check with trivial commands).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--idle-exit", type=float, default=30.0, help="minutes without a running or waiting task after which the driver ends (0: never)")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    Q = Path(a.dir)
    Q.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    if a.dry_run:
        nodes = ["n1", "n2", "n3"]
    else:
        nodes = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], capture_output=True, text=True).stdout.split()
    log(f"queue driver on {os.uname().nodename}; job {job}; {len(nodes)} nodes: {' '.join(nodes)}; queue {Q}")
    T, seen, serial, last, idle_since = {}, {}, 0, "", time.time()

    def scan():
        for f in sorted(Q.glob("*.task")):
            name, m = f.stem, f.stat().st_mtime
            if name in T or seen.get(name) == m:
                continue
            seen[name] = m
            try:
                d = json.loads(f.read_text())
                assert isinstance(d.get("cmd"), str) and d["cmd"].strip()
            except Exception as e:  # noqa: BLE001 - a half-written or wrong file: reported once, read again when it changes
                log(f"task file {f.name} not usable ({type(e).__name__}: {e})")
                continue
            if d.get("node") and d["node"] not in nodes:
                log(f"task {name}: node {d['node']} is not in this job - not accepted")
                continue
            T[name] = dict(cmd=d["cmd"], env=str(d.get("env", "")), needs=list(d.get("needs", [])), prio=float(d.get("prio", 5)), max_starts=int(d.get("starts", 1)),
                           pin=d.get("node"), status="waiting", starts=0, node=None, proc=None, t0=0.0, log=None, mins=0.0)
            log(f"new task {name}: prio {T[name]['prio']:g}, needs {' '.join(T[name]['needs']) or '-'}, starts {T[name]['max_starts']}: {d['cmd'][:160]}")

    def start(name, node):
        nonlocal serial
        t = T[name]
        serial += 1
        fake = f"{job}{serial:02d}"
        t.update(status="running", node=node, t0=time.time(), starts=t["starts"] + 1, log=f"logs/nano4/q-{job}-{name}-{t['starts'] + 1}.out")
        inner = f"export SLURM_JOB_ID={fake} {t['env']}; exec {t['cmd']}"
        if a.dry_run:
            t["proc"] = subprocess.Popen(["bash", "-c", t["cmd"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            t["proc"] = subprocess.Popen(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=96", "--gres=gpu:8", f"--nodelist={node}",
                                          f"--output={t['log']}", f"--error={t['log']}", "bash", "-c", inner])
        log(f"start {name} on {node} (start {t['starts']}, id {fake}) -> {t['log']}")

    def stop(t):
        t["proc"].terminate()
        try:
            t["proc"].wait(120)
        except subprocess.TimeoutExpired:
            t["proc"].kill()

    while True:
        now = time.time()
        scan()
        bad = set((Q / "BAD_NODES").read_text().split()) if (Q / "BAD_NODES").exists() else set()
        stopping = (Q / "STOP").exists()
        for name, t in T.items():
            if (Q / f"{name}.cancel").exists() and t["status"] in ("waiting", "running", "blocked"):
                if t["status"] == "running":
                    stop(t)
                    t["mins"] = (now - t["t0"]) / 60
                t["status"] = "cancelled"
                log(f"cancelled {name}")
                continue
            if t["status"] != "running":
                continue
            rc = t["proc"].poll()
            if rc is None:
                continue
            t["mins"] = (now - t["t0"]) / 60
            if rc == 0:
                t["status"] = "done"
                log(f"done {name} on {t['node']} ({t['mins']:.0f} min)")
            else:
                tail = "" if a.dry_run else subprocess.run(f"tail -n 4 {t['log']} | cut -c1-200", shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
                if t["starts"] >= t["max_starts"]:
                    t["status"] = "failed"
                    log(f"ALERT {name} failed (start {t['starts']} of {t['max_starts']}, on {t['node']}, {t['mins']:.0f} min, rc {rc}): {tail}")
                else:
                    t["status"] = "waiting"
                    log(f"{name} ended with an error on {t['node']} after {t['mins']:.0f} min (rc {rc}), will start again: {tail}")
        for name, t in T.items():  # a task whose need can never be done
            if t["status"] == "waiting" and any(T[n]["status"] in ("failed", "cancelled", "blocked") for n in t["needs"] if n in T):
                t["status"] = "blocked"
                log(f"ALERT {name} is blocked: needs {' '.join(n for n in t['needs'] if n in T and T[n]['status'] != 'done')}")
        busy = {t["node"] for t in T.values() if t["status"] == "running"}
        free = [n for n in nodes if n not in busy and n not in bad]
        if not stopping:
            for name, t in sorted(T.items(), key=lambda kv: (kv[1]["prio"], kv[0])):
                if not free:
                    break
                if t["status"] != "waiting" or not all(n in T and T[n]["status"] == "done" for n in t["needs"]):
                    continue
                if t["pin"]:
                    if t["pin"] not in free:
                        continue
                    node = t["pin"]
                else:
                    node = next((n for n in free if n != t["node"]), free[0])  # not the node of its last start when another is free
                free.remove(node)
                start(name, node)
        count = {k: sum(1 for t in T.values() if t["status"] == k) for k in ("waiting", "running", "done", "failed", "cancelled", "blocked")}
        summary = " ".join(f"{k}:{v}" for k, v in count.items() if v or k in ("waiting", "running", "done"))
        if summary != last:
            log("tasks " + summary + " | running: " + (" ".join(f"{n}@{t['node'][-3:]}" for n, t in T.items() if t["status"] == "running") or "-") + f" | free nodes: {len(free)}")
            last = summary
        lines = [f"{time.strftime('%F %T')} job {job}: {summary}; free nodes: {' '.join(n[-3:] for n in free) or '-'}{'; STOP' if stopping else ''}"]
        for name, t in T.items():
            mins = (now - t["t0"]) / 60 if t["status"] == "running" else t["mins"]
            lines.append(f"{name:28s} {t['status']:9s} {(t['node'] or '-'):12s} start {t['starts']}/{t['max_starts']} {mins:6.0f} min  {t['log'] or ''}")
        tmp = Q / "status.txt.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, Q / "status.txt")
        active = count["running"] + count["waiting"]
        if active:
            idle_since = now
        if stopping and not count["running"]:
            log("STOP: the running steps have ended"); break
        if a.idle_exit > 0 and not active and now - idle_since > a.idle_exit * 60:
            log(f"nothing running or waiting for {a.idle_exit:g} min: the driver ends"); break
        time.sleep(a.poll)
    notdone = [n for n, t in T.items() if t["status"] != "done"]
    log("not done: " + (" ".join(notdone) or "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
