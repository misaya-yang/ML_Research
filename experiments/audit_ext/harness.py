"""Frozen configs, parallel dispatch, resource guards and checkpoints for audit_ext.

The frozen config hashes the sources of both audit_lab and audit_ext. Workers are
single-threaded processes; at most `workers` run at once, and new tasks wait while
the 1-minute load average exceeds `load_ceiling` or free memory is below 6 GB.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from experiments.audit_lab.run_io import LOCAL_RUNS, ROOT, THREAD_ENV, cpu_time, make_run_dir

PACKAGES = ("audit_lab", "audit_ext")
MIN_FREE_BYTES = 6_000_000_000


def source_hashes():
    out = {}
    for pkg in PACKAGES:
        folder = ROOT / "experiments" / pkg
        for path in sorted(folder.glob("*.py")):
            out[f"{pkg}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def freeze(path, config):
    payload = dict(config, code_sha256=source_hashes())
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    for name, data in (("config.json", encoded), ("config.sha256", f"{digest}  config.json\n".encode())):
        output = path / name
        with output.open("xb") as stream:
            stream.write(data)
        output.chmod(0o444)
    return digest


def verify_frozen(path):
    encoded = (path / "config.json").read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if (path / "config.sha256").read_text().split() != [digest, "config.json"]:
        raise ValueError("Frozen config checksum mismatch")
    config = json.loads(encoded)
    if config["code_sha256"] != source_hashes():
        raise ValueError("Code changed since the run was frozen")
    return config, digest


def available_bytes():
    if platform.system() == "Darwin":
        total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        report = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        percent = int(re.search(r"free percentage: ([0-9]+)%", report)[1])
        return total * max(0, percent - 1) // 100
    with open("/proc/meminfo") as stream:
        info = dict(line.split(":", 1) for line in stream)
    return int(info["MemAvailable"].split()[0]) * 1024


class Guard:
    def __init__(self, load_ceiling):
        self.load_ceiling = load_ceiling
        self.min_available = available_bytes()
        self.wait_seconds = 0.
        self.max_load = os.getloadavg()[0]

    def ok(self):
        avail = available_bytes()
        load = os.getloadavg()[0]
        self.min_available = min(self.min_available, avail)
        self.max_load = max(self.max_load, load)
        return avail >= MIN_FREE_BYTES and load <= self.load_ceiling


def _init_worker():
    for name in THREAD_ENV:
        os.environ[name] = "1"
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass


def dispatch(tasks, function, *, workers, guard, checkpoint, log=print):
    """Run function(task) for tasks not yet in checkpoint (JSON lines keyed by task['key']).

    Returns {key: result}. Results are appended to checkpoint as they finish, so a
    relaunch on the same run directory resumes.
    """
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                done[item["key"]] = item["result"]
    pending = [task for task in tasks if task["key"] not in done]
    log(f"dispatch: {len(done)} cached, {len(pending)} to run, workers={workers}")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as pool, \
            checkpoint.open("a", buffering=1) as stream:
        running = {}
        queue = list(pending)
        while queue or running:
            while queue and len(running) < workers:
                if running and not guard.ok():
                    break
                if not running:
                    while not guard.ok():
                        guard.wait_seconds += 10
                        time.sleep(10)
                task = queue.pop(0)
                running[pool.submit(function, task)] = task
            finished, _ = wait(list(running), timeout=20, return_when=FIRST_COMPLETED)
            for future in finished:
                task = running.pop(future)
                result = future.result()
                done[task["key"]] = result
                stream.write(json.dumps({"key": task["key"], "result": result}, default=float) + "\n")
                log(f"done {task['key']} ({len(done)}/{len(tasks)}) load={os.getloadavg()[0]:.1f}")
    return done


class ExtRun:
    """Rows, costs and summary for one experiment directory."""

    def __init__(self, path, guard):
        self.path, self.guard = path, guard
        self.rows = []
        self.cpu_start, self.wall_start = cpu_time(), time.monotonic()

    def record(self, section, **fields):
        self.rows.append(dict(section=section, **fields))

    def cost(self):
        return {
            "cpu_hours": (cpu_time() - self.cpu_start) / 3600,
            "wall_hours": (time.monotonic() - self.wall_start) / 3600,
            "memory_wait_seconds": self.guard.wait_seconds,
            "min_observed_available_bytes": self.guard.min_available,
            "max_load_1min": self.guard.max_load,
            "gpu_hours": 0,
            **{key: int(sum(row.get(key, 0) or 0 for row in self.rows))
               for key in ("target_samples", "proposal_samples", "named_scores")},
        }
