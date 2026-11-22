"""Run directories, frozen configurations, costs and the local resource guard."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import secrets
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNS = ROOT / "results" / "local_runs"
MIN_FREE_BYTES = 6_000_000_000
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def make_run_dir(name, date=None):
    date = date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")
    LOCAL_RUNS.mkdir(exist_ok=True)
    while True:
        path = LOCAL_RUNS / f"{name}-{date}-{secrets.token_hex(2)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


@contextmanager
def local_job():
    """Hold a lock for the complete job, including any memory waits."""
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("Launch the local audit with nice -n 19")
    for lname in THREAD_ENV:
        if os.environ.get(lname) not in {"1", "2"}:
            raise RuntimeError(f"{lname} must be 1 or 2")
    LOCAL_RUNS.mkdir(exist_ok=True)
    fd = os.open(LOCAL_RUNS, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


class MemoryGuard:
    def __init__(self):
        self.total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        self.minimum = self.total
        self.wait_seconds = 0.
        self.last = -float("inf")

    def check(self, *, force=False):
        if not force and time.monotonic() - self.last < 2:
            return
        waiting_since = None
        while True:
            report = subprocess.check_output(["memory_pressure", "-Q"], text=True)
            percent = int(re.search(r"free percentage: ([0-9]+)%", report)[1])
            # The tool prints integral percentages. Use a lower bound.
            available = self.total * max(0, percent - 1) // 100
            if available >= MIN_FREE_BYTES:
                break
            if waiting_since is None:
                print(f"Pausing for memory reserve: macOS reports {percent}% free", flush=True)
                waiting_since = time.monotonic()
            time.sleep(10)
        if waiting_since is not None:
            self.wait_seconds += time.monotonic() - waiting_since
            print(f"Memory reserve restored: macOS reports {percent}% free", flush=True)
        self.minimum = min(self.minimum, available)
        self.last = time.monotonic()


def _source_hashes():
    folder = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.glob("*.py"))}


def source_digest():
    return hashlib.sha256(json.dumps(_source_hashes(), sort_keys=True).encode()).hexdigest()


def cpu_time():
    self_stats = resource.getrusage(resource.RUSAGE_SELF)
    child_stats = resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_stats.ru_utime + self_stats.ru_stime + child_stats.ru_utime + child_stats.ru_stime


def freeze(path, config):
    payload = dict(config, code_sha256=_source_hashes())
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
    if config["code_sha256"] != _source_hashes():
        raise ValueError("Code changed since the run was frozen")
    return config, digest


class Run:
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
            "min_observed_available_bytes": self.guard.minimum,
            "gpu_hours": 0,
            **{key: sum(row.get(key, 0) or 0 for row in self.rows)
               for key in ("target_samples", "proposal_samples", "named_scores")},
        }

    def save_raw(self):
        keys = list(dict.fromkeys(key for row in self.rows for key in row))
        with (self.path / "raw.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys or ["section"])
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self, digest, sections, cost=None):
        if cost is None:
            cost = self.cost()
        details = cost_lines(cost)
        lines = [f"# {self.path.name}", "", f"配置 SHA-256 `{digest}`", "",
                 *sections, "", "## 成本", "", *details, ""]
        (self.path / "summary.md").write_text("\n".join(lines))


@dataclass(frozen=True)
class Finding:
    status: str
    detail: str
    figures: tuple = ()

    def bullet(self, prediction, base=""):
        figures = " ".join(f"[{name}]({base}figs/{name})" for name in self.figures)
        return f"- **{prediction} {self.status}**。{self.detail} {figures}".rstrip()


def cost_lines(cost):
    return [
        f"- CPU {cost['cpu_hours']:.5f} 小时，墙钟 {cost['wall_hours']:.5f} 小时，"
        f"GPU {cost['gpu_hours']:g} 小时",
        f"- P 样本 {cost['target_samples']:,}，Q 样本 {cost['proposal_samples']:,}，"
        f"点名评分 {cost['named_scores']:,}",
        f"- macOS 可用内存下界 {cost['min_observed_available_bytes'] / 10**9:.3f} GB，"
        f"内存等待 {cost['memory_wait_seconds']:.1f} 秒",
    ]
