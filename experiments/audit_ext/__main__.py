"""Phase-1 remainder runner. From the repo root:

  nice -n 19 python -m experiments.audit_ext test
  nice -n 19 python -m experiments.audit_ext run X8 --tests results/local_runs/TESTEXT-.../test.log
  nice -n 19 python -m experiments.audit_ext run X8 --resume results/local_runs/X8-...   (resume)
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import resource
from pathlib import Path
import time
import traceback

from experiments.audit_lab.run_io import LOCAL_RUNS, ROOT, THREAD_ENV, cost_lines, cpu_time, make_run_dir
from . import analysis
from .configs import config_for
from .worker import work
from .harness import ExtRun, Guard, dispatch, freeze, source_hashes, verify_frozen

TASKS = dict(X8=analysis.x8_tasks, X9A=analysis.x9a_tasks, X9B=analysis.x9b_tasks,
             X10=analysis.x10_tasks, X1E=analysis.x1e_tasks, X11=analysis.x11_tasks)
ANALYZE = dict(X8=analysis.x8_analyze, X9A=analysis.x9a_analyze, X9B=analysis.x9b_analyze,
               X10=analysis.x10_analyze, X1E=analysis.x1e_analyze,
               X11=analysis.x11_analyze)
DIRNAME = dict(X8="X8", X9A="X9a", X9B="X9b", X10="X10", X1E="X1e", X11="X11")


def _digest():
    import hashlib
    return hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()


def _test():
    folder = make_run_dir("TESTEXT")
    print(folder, flush=True)
    with (folder / "test.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        wall, cpu = time.monotonic(), cpu_time()
        try:
            from experiments.audit_lab.test_suite import run as base_run
            from .tests import run as ext_run
            base_run(lambda **_: None)
            ext_run()
        except Exception:
            traceback.print_exc()
            return False
        print(f"PASS code_sha256={_digest()} cpu_seconds={cpu_time()-cpu:.4f} "
              f"wall_seconds={time.monotonic()-wall:.4f}")
    return True


def _run(name, tests, resume):
    if resume:
        folder = resume.resolve()
        config, digest = verify_frozen(folder)
    else:
        log = tests.read_text().splitlines()
        if not log or not log[-1].startswith(f"PASS code_sha256={_digest()} "):
            raise SystemExit("Run the extended acceptance tests against the current source first")
        folder = make_run_dir(DIRNAME[name])
        (folder / "figs").mkdir()
        config = config_for(name)
        config["acceptance_test"] = str(tests.resolve().relative_to(LOCAL_RUNS))
        digest = freeze(folder, config)
        config, _ = verify_frozen(folder)
    print(folder, flush=True)
    guard = Guard(config["load_ceiling"])
    run = ExtRun(folder, guard)
    with (folder / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(f"{'Resumed' if resume else 'Frozen'} config SHA-256 {digest}", flush=True)
        tasks = TASKS[name](config)
        for task in tasks:
            task["exp"] = name
        results = dispatch(tasks, work, workers=config["workers"], guard=guard,
                           checkpoint=folder / "checkpoint.jsonl")
        findings, rows, notes = ANALYZE[name](folder, config, results)
        analysis.save_rows(folder, rows)
        cost = run.cost()
        task_cpu = sum(res.get("cpu_seconds", 0) for res in results.values()) / 3600
        # Worker CPU is measured per task (survives resumes); add this process's own CPU.
        own = resource.getrusage(resource.RUSAGE_SELF)
        cost["cpu_hours"] = task_cpu + (own.ru_utime + own.ru_stime) / 3600
        cost["target_samples"] = sum(int(res.get("target_samples", 0)) for res in results.values()) \
            + cost["target_samples"]
        cost["proposal_samples"] = sum(int(res.get("proposal_samples", 0)) for res in results.values()) \
            + cost["proposal_samples"]
        lines = [f"# {folder.name}", "", f"配置 SHA-256 `{digest}`", ""]
        lines += [finding.bullet(pred, "") for pred, finding in sorted(findings.items())]
        lines += ["", "## 实施口径", "", *notes, "", "## 成本", "", *cost_lines(cost),
                  f"- 1 分钟负载峰值 {cost['max_load_1min']:.1f}，并行 worker ≤{config['workers']}（单线程）", ""]
        (folder / "summary.md").write_text("\n".join(lines))
        (folder / "findings.json").write_text(json.dumps(
            {pred: dict(status=f.status, detail=f.detail, figures=list(f.figures)) for pred, f in findings.items()},
            ensure_ascii=False, indent=1))
        (folder / "cost.json").write_text(json.dumps(cost, indent=1, default=float))
        print("Completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("test", "run"))
    parser.add_argument("name", nargs="?", choices=tuple(TASKS))
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("Launch from the repository root")
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        parser.error("Launch with nice -n 19")
    for env in THREAD_ENV:
        os.environ[env] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    if args.action == "test":
        raise SystemExit(0 if _test() else 1)
    if args.name is None or (args.tests is None and args.resume is None):
        parser.error("run needs NAME and --tests (or --resume)")
    _run(args.name, args.tests, args.resume)


if __name__ == "__main__":
    main()
