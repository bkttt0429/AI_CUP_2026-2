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
import argparse  # 命令列參數
import json  # 讀取任務檔
import os  # 環境變數、主機名稱與原子性換名
import subprocess  # 啟動 srun 步驟與查詢節點
import sys  # 結束碼
import time  # 時間戳記、計時與輪詢間隔
from pathlib import Path  # 以路徑物件處理檔案


def log(msg):  # 印出帶時間的訊息
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)  # 立即輸出，方便即時追蹤


def main():  # 驅動程式本體
    ap = argparse.ArgumentParser()  # 命令列參數
    ap.add_argument("--dir", required=True)  # 任務佇列資料夾
    ap.add_argument("--idle-exit", type=float, default=30.0, help="minutes without a running or waiting task after which the driver ends (0: never)")  # 閒置多久後結束
    ap.add_argument("--poll", type=float, default=15.0)  # 輪詢間隔（秒）
    ap.add_argument("--dry-run", action="store_true")  # 不使用 srun 的測試模式
    a = ap.parse_args()  # 解析參數
    Q = Path(a.dir)  # 佇列資料夾
    Q.mkdir(parents=True, exist_ok=True)  # 不存在就建立
    job = os.environ.get("SLURM_JOB_ID", "local")  # 所在工作的編號
    if a.dry_run:  # 測試模式
        nodes = ["n1", "n2", "n3"]  # 假的節點名稱
    else:  # 正式執行
        nodes = subprocess.run(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], capture_output=True, text=True).stdout.split()  # 展開工作分配到的節點清單
    log(f"queue driver on {os.uname().nodename}; job {job}; {len(nodes)} nodes: {' '.join(nodes)}; queue {Q}")  # 回報驅動程式的位置與節點
    T, seen, serial, last, idle_since = {}, {}, 0, "", time.time()  # 任務表、看過的任務檔版本、步驟序號、上次的摘要、閒置起點

    def scan():  # 讀入新的任務檔
        for f in sorted(Q.glob("*.task")):  # 佇列中的每個任務檔
            name, m = f.stem, f.stat().st_mtime  # 任務名稱與修改時間
            if name in T or seen.get(name) == m:  # 已經接受，或先前看過、未被接受且沒有變動的檔
                continue  # 略過
            seen[name] = m  # 記下這次看過的版本
            try:  # 解析任務檔
                d = json.loads(f.read_text())  # 讀取 JSON
                assert isinstance(d.get("cmd"), str) and d["cmd"].strip()  # 必須有非空的指令
            except Exception as e:  # noqa: BLE001  寫到一半或內容錯誤的檔：只回報一次，檔案變動時再讀
                log(f"task file {f.name} not usable ({type(e).__name__}: {e})")  # 回報無法使用
                continue  # 先略過
            if d.get("node") and d["node"] not in nodes:  # 釘的節點不在這個工作中
                log(f"task {name}: node {d['node']} is not in this job - not accepted")  # 回報不接受
                continue  # 略過
            T[name] = dict(cmd=d["cmd"], env=str(d.get("env", "")), needs=list(d.get("needs", [])), prio=float(d.get("prio", 5)), max_starts=int(d.get("starts", 1)),  # 任務的設定
                           pin=d.get("node"), status="waiting", starts=0, node=None, proc=None, t0=0.0, log=None, mins=0.0)  # 與執行狀態
            log(f"new task {name}: prio {T[name]['prio']:g}, needs {' '.join(T[name]['needs']) or '-'}, starts {T[name]['max_starts']}: {d['cmd'][:160]}")  # 回報新任務

    def start(name, node):  # 在指定節點啟動一個任務
        nonlocal serial  # 遞增外層的步驟序號
        t = T[name]  # 任務資料
        serial += 1  # 新的步驟序號
        fake = f"{job}{serial:02d}"  # 給任務自己的工作編號字串
        t.update(status="running", node=node, t0=time.time(), starts=t["starts"] + 1, log=f"logs/nano4/q-{job}-{name}-{t['starts'] + 1}.out")  # 標成執行中並決定紀錄檔
        inner = f"export SLURM_JOB_ID={fake} {t['env']}; exec {t['cmd']}"  # 設定環境變數後執行指令
        if a.dry_run:  # 測試模式
            t["proc"] = subprocess.Popen(["bash", "-c", t["cmd"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # 直接在本機執行
        else:  # 正式執行
            t["proc"] = subprocess.Popen(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=96", "--gres=gpu:8", f"--nodelist={node}",  # 單一節點、8 張 GPU 的 srun 步驟
                                          f"--output={t['log']}", f"--error={t['log']}", "bash", "-c", inner])  # 輸出與錯誤都寫到任務的紀錄檔
        log(f"start {name} on {node} (start {t['starts']}, id {fake}) -> {t['log']}")  # 回報啟動

    def stop(t):  # 終止一個執行中的任務
        t["proc"].terminate()  # 先送終止訊號
        try:  # 等它自行結束
            t["proc"].wait(120)  # 最多等 120 秒
        except subprocess.TimeoutExpired:  # 逾時
            t["proc"].kill()  # 強制結束

    while True:  # 主迴圈
        now = time.time()  # 這一輪的時間
        scan()  # 接收新任務
        bad = set((Q / "BAD_NODES").read_text().split()) if (Q / "BAD_NODES").exists() else set()  # 不再分派的節點
        stopping = (Q / "STOP").exists()  # 是否要求停止
        for name, t in T.items():  # 更新每個任務的狀態
            if (Q / f"{name}.cancel").exists() and t["status"] in ("waiting", "running", "blocked"):  # 被要求取消
                if t["status"] == "running":  # 正在執行
                    stop(t)  # 終止步驟
                    t["mins"] = (now - t["t0"]) / 60  # 記錄執行分鐘數
                t["status"] = "cancelled"  # 標成已取消
                log(f"cancelled {name}")  # 回報
                continue  # 處理下一個
            if t["status"] != "running":  # 只檢查執行中的任務
                continue  # 其他略過
            rc = t["proc"].poll()  # 檢查步驟是否結束
            if rc is None:  # 還在執行
                continue  # 下一個
            t["mins"] = (now - t["t0"]) / 60  # 記錄執行分鐘數
            if rc == 0:  # 成功
                t["status"] = "done"  # 標成完成
                log(f"done {name} on {t['node']} ({t['mins']:.0f} min)")  # 回報
            else:  # 失敗
                tail = "" if a.dry_run else subprocess.run(f"tail -n 4 {t['log']} | cut -c1-200", shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")  # 取紀錄檔的最後 4 行
                if t["starts"] >= t["max_starts"]:  # 啟動次數已用完
                    t["status"] = "failed"  # 標成失敗
                    log(f"ALERT {name} failed (start {t['starts']} of {t['max_starts']}, on {t['node']}, {t['mins']:.0f} min, rc {rc}): {tail}")  # 發出警示
                else:  # 還能重試
                    t["status"] = "waiting"  # 回到等待，之後盡量換到其他節點再啟動
                    log(f"{name} ended with an error on {t['node']} after {t['mins']:.0f} min (rc {rc}), will start again: {tail}")  # 回報將重試
        for name, t in T.items():  # 前置任務永遠無法完成的任務
            if t["status"] == "waiting" and any(T[n]["status"] in ("failed", "cancelled", "blocked") for n in t["needs"] if n in T):  # 前置任務失敗、取消或被擋
                t["status"] = "blocked"  # 標成 blocked（不會自動解除）
                log(f"ALERT {name} is blocked: needs {' '.join(n for n in t['needs'] if n in T and T[n]['status'] != 'done')}")  # 列出未完成的前置任務
        busy = {t["node"] for t in T.values() if t["status"] == "running"}  # 正在執行任務的節點
        free = [n for n in nodes if n not in busy and n not in bad]  # 空閒且不是壞節點的節點
        if not stopping:  # 沒有要求停止時才啟動新任務
            for name, t in sorted(T.items(), key=lambda kv: (kv[1]["prio"], kv[0])):  # 依優先序（小的先）與名稱
                if not free:  # 沒有空閒節點
                    break  # 停止分派
                if t["status"] != "waiting" or not all(n in T and T[n]["status"] == "done" for n in t["needs"]):  # 不在等待，或前置任務未完成
                    continue  # 略過
                if t["pin"]:  # 釘在特定節點的任務
                    if t["pin"] not in free:  # 該節點不空閒
                        continue  # 下一輪再說
                    node = t["pin"]  # 用釘的節點
                else:  # 沒有釘節點
                    node = next((n for n in free if n != t["node"]), free[0])  # 有其他空閒節點時，不用上次啟動它的節點
                free.remove(node)  # 這個節點不再空閒
                start(name, node)  # 啟動任務
        count = {k: sum(1 for t in T.values() if t["status"] == k) for k in ("waiting", "running", "done", "failed", "cancelled", "blocked")}  # 各狀態的任務數
        summary = " ".join(f"{k}:{v}" for k, v in count.items() if v or k in ("waiting", "running", "done"))  # 狀態摘要
        if summary != last:  # 摘要有變化時才記錄
            log("tasks " + summary + " | running: " + (" ".join(f"{n}@{t['node'][-3:]}" for n, t in T.items() if t["status"] == "running") or "-") + f" | free nodes: {len(free)}")  # 印出摘要、執行中的任務與空閒節點數
            last = summary  # 記下這次的摘要
        lines = [f"{time.strftime('%F %T')} job {job}: {summary}; free nodes: {' '.join(n[-3:] for n in free) or '-'}{'; STOP' if stopping else ''}"]  # status.txt 的第一行：時間、摘要與空閒節點
        for name, t in T.items():  # 每個任務一行
            mins = (now - t["t0"]) / 60 if t["status"] == "running" else t["mins"]  # 執行中的任務以現在時間計算
            lines.append(f"{name:28s} {t['status']:9s} {(t['node'] or '-'):12s} start {t['starts']}/{t['max_starts']} {mins:6.0f} min  {t['log'] or ''}")  # 名稱、狀態、節點、啟動次數、分鐘數與紀錄檔
        tmp = Q / "status.txt.tmp"  # 先寫暫存檔
        tmp.write_text("\n".join(lines) + "\n")  # 寫入狀態
        os.replace(tmp, Q / "status.txt")  # 換名，讀的人不會看到寫到一半的檔
        active = count["running"] + count["waiting"]  # 執行中與等待中的任務數
        if active:  # 還有事要做
            idle_since = now  # 重設閒置起點
        if stopping and not count["running"]:  # 要求停止且已沒有執行中的步驟
            log("STOP: the running steps have ended"); break  # 回報後離開主迴圈
        if a.idle_exit > 0 and not active and now - idle_since > a.idle_exit * 60:  # 閒置超過時限
            log(f"nothing running or waiting for {a.idle_exit:g} min: the driver ends"); break  # 回報後離開主迴圈（工作隨之結束，不佔住節點）
        time.sleep(a.poll)  # 等待下一輪
    notdone = [n for n, t in T.items() if t["status"] != "done"]  # 沒有完成的任務
    log("not done: " + (" ".join(notdone) or "-"))  # 回報
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
