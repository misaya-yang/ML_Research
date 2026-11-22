"""Run the acceptance tests or the prespecified local phase from the repo root."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import traceback

from .run_io import (Finding, LOCAL_RUNS, ROOT, THREAD_ENV, MemoryGuard, Run,
                     cost_lines, cpu_time, freeze, local_job, make_run_dir, source_digest,
                     verify_frozen)


def _test(date):
    folder = make_run_dir("TEST", date)
    print(folder, flush=True)
    guard = MemoryGuard()
    with (folder / "test.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        wall, cpu = time.monotonic(), cpu_time()
        try:
            guard.check(force=True)
            from .test_suite import run

            run(guard.check)
            guard.check(force=True)
        except Exception:
            traceback.print_exc()
            return False
        print(f"PASS code_sha256={source_digest()} cpu_seconds={cpu_time()-cpu:.4f} "
              f"wall_seconds={time.monotonic()-wall:.4f} min_available_bytes={guard.minimum}")
    return True


def _execute(name, date, tests):
    folder = make_run_dir(name, date)
    print(folder, flush=True)
    (folder / "figs").mkdir()
    guard = MemoryGuard()
    run = Run(folder, guard)
    with (folder / "run.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        guard.check(force=True)
        from .configs import config_for

        config = config_for(name)
        config["acceptance_test"] = str(tests.relative_to(LOCAL_RUNS))
        digest = freeze(folder, config)
        config, _ = verify_frozen(folder)
        print(f"Frozen config SHA-256 {digest}", flush=True)
        try:
            module = importlib.import_module(f".{name.lower()}", __package__)
            findings, notes = module.run(run, config)
            guard.check(force=True)
        except Exception:
            traceback.print_exc()
            run.save_raw()
            run.finish(digest, ["运行中止。异常和已完成步骤见 [run.log](run.log)。"])
            raise
        run.save_raw()
        cost = run.cost()
        lines = [finding.bullet(prediction) for prediction, finding in sorted(findings.items())]
        run.finish(digest, [*lines, "", "## 实施口径", "", *notes], cost)
        print("Completed", flush=True)
    return folder, digest, cost, findings


def _status(runs, test_path, test_log):
    lines = ["# 冷启动 IMH 审计｜阶段 1", "",
             "本机 CPU 已按 X2 → X1 → X3 完成。停在阶段 1，等待确认后进入 Kaggle。", "",
             "## 单元测试", "",
             "第 2 节对照全部通过。覆盖稠密 K、100 万条独立链、cap equality 和分数 DP。"
             f" [测试日志]({test_path.relative_to(LOCAL_RUNS)})", "",
             "## 预测", ""]
    all_findings = {pred: (folder, result) for folder, _, _, findings in runs
                    for pred, result in findings.items()}
    for pred in sorted(all_findings):
        folder, finding = all_findings[pred]
        lines.append(finding.bullet(pred, f"{folder.name}/"))
    lines.append(Finding("未运行", "X6 在阶段 3 汇总。阶段 1 的 X1b 已保存不安全试验的签发数。").bullet("P8"))
    lines.extend(["", "## 配置、成本和口径", ""])
    for folder, digest, _, _ in runs:
        base = folder.name
        lines.append(
            f"- {base.split('-')[0]} · [配置]({base}/config.json) · [SHA-256]({base}/config.sha256) "
            f"· [逐点数据]({base}/raw.csv) · [汇总]({base}/summary.md) · [运行日志]({base}/run.log)"
        )
    total = {key: sum(cost[key] for _, _, cost, _ in runs) for key in runs[0][2]}
    total["min_observed_available_bytes"] = min(cost["min_observed_available_bytes"] for _, _, cost, _ in runs)
    test_cpu = float(test_log[-1].split("cpu_seconds=")[1].split()[0])
    lines.extend(["", f"以下是三个实验的合计。验收测试另用 CPU {test_cpu / 3600:.5f} 小时。", "",
                  *cost_lines(total), "",
                  "样本数是计数模拟代表的预算，包含 LR 每个预算点的两个世界。"
                  "X3 使用精确有限状态计算。各配置冻结后未更改。",
                  "P1 对截断点单列覆盖情况，P2 保留原网格，P6 使用全部正缺口。", ""])
    path = LOCAL_RUNS / "STATUS_zh.md"
    temporary = LOCAL_RUNS / ".STATUS_zh.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("test", "stage1"))
    parser.add_argument("--date", help="Run directory date, YYYYMMDD")
    parser.add_argument("--tests", type=Path, help="Passing test.log from a prior test command")
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("Launch from the repository root")
    if args.date is not None and (len(args.date) != 8 or not args.date.isdecimal()):
        parser.error("--date must have format YYYYMMDD")
    for env in THREAD_ENV:
        os.environ[env] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    with local_job():
        if args.action == "test":
            raise SystemExit(0 if _test(args.date) else 1)
        if args.tests is None:
            parser.error("stage1 requires --tests")
        tests = args.tests.resolve()
        if not tests.is_relative_to(LOCAL_RUNS):
            parser.error("--tests must be under results/local_runs")
        test_log = tests.read_text().splitlines()
        if not test_log or not test_log[-1].startswith(f"PASS code_sha256={source_digest()} "):
            parser.error("Run the acceptance tests against the current source before stage1")
        with TemporaryDirectory(prefix="imh-audit-mpl-") as cache:
            os.environ["MPLCONFIGDIR"] = cache
            runs = [_execute(name, args.date, tests) for name in ("X2", "X1", "X3")]
        _status(runs, tests, test_log)


if __name__ == "__main__":
    main()
