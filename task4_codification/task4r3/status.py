"""Round 3 progress at a glance, no Claude needed (plan §4 step 6).

    python status.py
"""
import csv
import json
import re
import subprocess
import time
from pathlib import Path

R3 = Path(__file__).resolve().parent
LOGS = R3 / "logs"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:                                  # noqa: BLE001
        return ""


def main():
    st = (LOGS / "stage.txt").read_text().strip().split("\n") if (LOGS / "stage.txt").exists() else []
    print("== stage:", st[-1] if st else "not started (start with: tmux new -d -s r3 ./go.sh)")
    planned = [l for l in sh(f"bash {R3}/run_all.sh list r3").split("\n") if l.strip()]
    res = R3 / "results" / "results.csv"
    done = {r["run_id"] for r in csv.DictReader(open(res))} if res.exists() else set()
    rid = lambda j: "{}_{}_s{}".format(*re.search(r"--arm (\S+) --fold (\S+) --seed (\d+)", j).groups())
    todo = [rid(j) for j in planned if rid(j) not in done]
    print(f"== runs: {len(done)}/{len(planned)} done, {len(todo)} to go")
    con = LOGS / "r3_console.log"
    if con.exists():
        txt = con.read_text()
        fails = re.findall(r"FAILED TWICE \d+: (.*)", txt)
        retry = len(re.findall(r"FAILED \d+ \(retrying once\)", txt))
        print(f"== failures: {retry} retried, {len(fails)} failed twice" + "".join(f"\n   {f}" for f in fails))
        started = {}
        for m in re.finditer(r"\[(\S+ \S+) gpu (\d+) slot (\d+)\] start \d+: (.*) --device", txt):
            started[rid(m.group(4))] = (m.group(1), m.group(2), m.group(4))
        running = [k for k in started if k not in done and k in todo]
        if running:
            print("== running:")
            for k in running:
                t0, g, cmd = started[k]
                log = LOGS / (cmd.replace(" ", "_")[:120] + ".log")
                ep = re.findall(r"epoch (\d+) done in (\d+)s; source-val image AUROC ([\d.]+)",
                                log.read_text()) if log.exists() else []
                last = f"epoch {int(ep[-1][0]) + 1}/20 done (val {ep[-1][2]}, {ep[-1][1]} s/epoch)" if ep else "epoch 1 running"
                print(f"   gpu {g}  {k:24s} since {t0[11:]}  {last}")
    proj = LOGS / "projection.json"
    if proj.exists() and con.exists() and st:
        p = json.load(open(proj))
        first = re.search(r"== \d+ jobs on \d+ slots (\S+ \S+)", con.read_text())
        if first:
            t0 = time.mktime(time.strptime(first.group(1), "%Y-%m-%d %H:%M:%S"))
            el = (time.time() - t0) / 3600
            left = sum(v for k, v in p["per_job_minutes"].items() if k not in done) / 12 / 60
            print(f"== time: {el:.1f} h elapsed; projection was {p['wall_hours']:.1f} h in total; "
                  f"≈ {left:.1f} h of queued work left (12 slots, rough)")
    gpu = sh("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader")
    print("== GPUs (index, used, total, load):\n   " + gpu.strip().replace("\n", "\n   "))
    if (R3 / "results" / "analysis.md").exists():
        print("== results/analysis.md is written: start Session B (task4r3/STATE.md)")


if __name__ == "__main__":
    main()
