"""Submit and verify the versioned artifacts of a frozen private run."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

from ..run_io import LOCAL_RUNS, MemoryGuard, THREAD_ENV, local_job, verify_frozen
from .prepare import file_digest, write_json


def _private(metadata, config):
    return (str(metadata.get("is_private")).lower() == "true"
            and metadata.get("id") == config["kernel"])


def submit(folder):
    config, digest = verify_frozen(folder)
    submission = json.loads((folder / "submission.json").read_text())
    stage = Path(submission["stage_path"])
    metadata_path = stage / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (not _private(metadata, config) or not config["private"]
            or config["accelerator"] != metadata["machine_shape"]
            or submission["config_sha256"] != digest
            or file_digest(metadata_path) != submission["metadata_sha256"]
            or file_digest(stage / "run.py") != submission["remote_script_sha256"]):
        raise ValueError("Private submission metadata or frozen script did not verify")
    command = ["kaggle", "kernels", "push", "-p", str(stage),
               "--timeout", str(config["session_timeout_seconds"]),
               "--accelerator", config["accelerator"]]
    log = folder / "submission.log"
    with log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
    match = re.search(r"Kernel version (\d+) successfully pushed", log.read_text(), re.I)
    if not match:
        raise RuntimeError(f"Could not confirm Kaggle publication: {log}")
    version = int(match[1])
    receipt = {"version": version, "versioned_kernel": f"{config['kernel']}/{version}",
               "config_sha256": digest, "session_timeout_seconds": config["session_timeout_seconds"]}
    write_json(folder / "version.json", receipt)
    return receipt["versioned_kernel"]


def verify_bank(raw, row, config):
    import numpy as np

    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"{row['key']} sample-bank hash mismatch")
    types = {"r": "float32", "logq": "float64", "text_hash": "uint64",
             "length": "uint8", "h": "uint8"}
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != set(types):
            raise ValueError("Bank column mismatch")
        columns = {name: data[name] for name in types}
    n = config["bank_rows"]
    if n != row["rows"] or any(col.shape != (n,) or col.dtype != types[name]
                               for name, col in columns.items()):
        raise ValueError("Bank shape or dtype mismatch")
    r, logq, length, h = (columns[name] for name in ("r", "logq", "length", "h"))
    if (not np.isfinite(r).all() or not np.isfinite(logq).all()
            or (abs(r) > 4).any() or (logq > 0).any()
            or (length < 1).any() or (length > config["generation"]["max_new_tokens"]).any()
            or (h > 7).any() or int(length.sum()) != row["generated_tokens"]
            or not np.array_equal((h & 2) != 0,
                                  r.astype(float) > config["labels"]["positive_logit_threshold"])):
        raise ValueError("Bank value or accounting mismatch")
    for i in range(3):
        if np.count_nonzero(h & (1 << i)) != row[f"h{i+1}_ones"]:
            raise ValueError(f"Bank bit {i} count mismatch")


def verify_summary(summary, config, digest, checkpoint):
    if (summary["experiment"] != config["experiment"] or summary["smoke"] != config["smoke"]
            or summary["config_sha256"] != digest or checkpoint["config_sha256"] != digest
            or checkpoint["status"] != summary["status"]):
        raise ValueError("Remote summary and checkpoint disagree with the frozen config")
    rows = summary["prompts"]
    frozen_keys = [row["key"] for row in config["prompts"]]
    keys = [row["key"] for row in rows]
    if (len(set(keys)) != len(keys) or any(key not in frozen_keys for key in keys)
            or (summary["status"] == "complete" and keys != frozen_keys)):
        raise ValueError("Remote prompt ledger has missing or unexpected keys")
    if summary["status"] == "complete" and summary["proposal_samples"] != sum(row["rows"] for row in rows):
        raise ValueError("Inconsistent sample tally")
    if summary["status"] == "complete":
        devices = summary["runtime"]["devices"]
        if len(devices) != config["expected_gpus"] or any("T4" not in device for device in devices):
            raise ValueError("Kaggle runtime does not match the configured accelerators")
    if config["experiment"] == "X0":
        screen = config["screen"]
        passing = []
        for row in rows:
            passed = row["dup_rate"] < screen["max_dup_rate"] and all(
                row[f"head_Pmass_beta_{beta:g}"] < screen["max_head_Pmass"] for beta in config["betas"])
            if passed != row["passed"] or row["rows"] != screen["samples"]:
                raise ValueError("X0 ledger does not match the frozen screen")
            if passed:
                passing.append(row["key"])
        if summary["passing_prompts"] != passing:
            raise ValueError("X0 passing-prompt ledger mismatch")
        if summary["status"] == "complete" and not config["smoke"]:
            if summary["passed"] != (len(passing) >= screen["min_pass"]):
                raise ValueError("X0 gate mismatch")


def verify_archive(folder, archive_path):
    config, digest = verify_frozen(folder)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.read("config.json") != (folder / "config.json").read_bytes():
            raise ValueError("Remote frozen config differs byte-for-byte")
        checkpoint = json.loads(archive.read("checkpoint.json"))
        summary = json.loads(archive.read("summary.json"))
        verify_summary(summary, config, digest, checkpoint)
        extracted = ["checkpoint.json", "summary.json"]
        history = []
        for member in sorted(archive.namelist()):
            if re.fullmatch(r"checkpoints/ckpt_[0-9]{4}\.json", member):
                history.append(json.loads(archive.read(member)))
                extracted.append(member)
        if (not history or history[-1] != checkpoint
                or any(item["config_sha256"] != digest for item in history)):
            raise ValueError("Checkpoint history mismatch")
        gaps = [second["elapsed_s"] - first["elapsed_s"] for first, second in zip(history, history[1:])]
        if max(gaps, default=0) > config["checkpoint_seconds"] + 5:
            raise ValueError("Checkpoint cadence exceeded")
        if "results/raw.csv" in archive.namelist():
            extracted.append("results/raw.csv")
        for row in summary["prompts"]:
            extracted.append(f"results/{row['key']}.json")
            if config["experiment"] == "X4":
                bank = f"results/bank-{row['key']}.npz"
                verify_bank(archive.read(bank), row, config)
                extracted.append(bank)
        for member in extracted:
            path = folder / member
            if member == "results/raw.csv":
                path = folder / "raw.csv"
            path.parent.mkdir(exist_ok=True)
            content = archive.read(member)
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError(f"Previously collected artifact differs: {path.name}")
            else:
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
    return summary


def write_run_summary(folder, config, summary, version):
    rows, elapsed = summary["prompts"], summary["elapsed_s"] / 3600
    description = "SMOKE" if config["smoke"] else "全量"
    lines = [f"# {folder.name}｜{config['experiment']} {description}", "",
             f"私有 Kaggle 作业 #{version.rsplit('/', 1)[1]}，运行状态 {summary['status']}。"
             f" [Notebook](https://www.kaggle.com/code/{config['kernel']})",
             "", f"配置 SHA-256 {summary['config_sha256']}。", ""]
    if config["experiment"] == "X0" and rows:
        screen = config["screen"]
        n_pass = len(summary["passing_prompts"])
        max_head = max(row[f"head_Pmass_beta_{beta:g}"] for row in rows for beta in config["betas"])
        max_dup = max(row["dup_rate"] for row in rows)
        lines.extend([
            f"- 已筛查 {len(rows)}/{len(config['prompts'])} 个前缀。每个前缀使用 {screen['beams']} 束和 "
            f"{screen['samples']:,} 条独立 Q 样本。",
            f"- {n_pass}/{len(rows)} 个前缀满足两条筛查阈值。最大加权头质量 {max_head:.8g}，"
            f"最大重复率 {max_dup:.4%}。 [逐前缀数据](raw.csv)",
            "- review ID 为 IMDb 原始相对文件名。按其 SHA-256 排序后冻结 GPT-2 前 8 个 token。"
            "头部按 token 序列去重，Q 样本重复率按完整解码文本哈希计算。",
        ])
        if summary["passed"] is not None:
            passed = "通过" if summary["passed"] else "未通过"
            lines.append(f"- X0 {passed}。进入 X4 要求至少 {screen['min_pass']} 个前缀通过。")
    if config["experiment"] == "X4" and rows:
        lines.extend([
            f"- 完成 {len(rows)}/{len(config['prompts'])} 个样本库，每库 {config['bank_rows']:,} 行。"
            " [样本库清单](raw.csv)",
            "- 每行有 float32 奖励、log Q、uint64 文本哈希、生成长度和 3 位标签。"
            "标签来自去除 EOS 的完整解码文本。log Q 和生成长度计入首次 EOS。",
        ])
    if summary["error"]:
        lines.extend(["", f"运行错误 {summary['error']}。 [运行日志](run.log)"])
    lines.extend([
        "", "## 成本", "",
        f"- 作业内计时 {elapsed:.5f} 小时，按 T4 计 {elapsed * len(summary['runtime'].get('devices', [])):.5f} 设备小时。"
        f"服务器超时上限 {config['session_timeout_seconds'] / 3600:.5f} 小时。",
        f"- Q 采样 {summary['proposal_samples']:,} 条，点名评分 {summary['named_scores']:,} 次，"
        f"Q 采样的新增 token {summary['generated_tokens']:,} 个。P 样本与拒绝采样尝试均为 0。",
        "- 下载后核验配置、checkpoint 历史和逐前缀数据。 [运行输出](results_final.zip)", "",
    ])
    path = folder / "summary.md"
    content = "\n".join(lines)
    if path.exists() and path.read_text() != content:
        raise ValueError("Previously written summary differs")
    if not path.exists():
        path.write_text(content)


def collect(folder):
    config, _ = verify_frozen(folder)
    version = json.loads((folder / "version.json").read_text())["versioned_kernel"]
    scratch = Path(json.loads((folder / "submission.json").read_text())["stage_path"])
    # Read back Kaggle's own metadata before accepting any output.
    checked = scratch / "server_metadata"
    checked.mkdir(exist_ok=True)
    commands = (
        ["kaggle", "kernels", "pull", version, "-p", str(checked), "-m"],
        ["kaggle", "kernels", "output", version, "-p", str(folder), "-q",
         "--file-pattern", r"^results_final\.zip$"],
    )
    with (folder / "fetch.log").open("a") as log:
        subprocess.run(commands[0], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        metadata = json.loads((checked / "kernel-metadata.json").read_text())
        if not _private(metadata, config):
            raise ValueError("Kaggle metadata does not confirm this notebook is private")
        archive = folder / "results_final.zip"
        if not archive.is_file():
            subprocess.run(commands[1], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    summary = verify_archive(folder, archive)
    write_json(folder / "verification.json", {"version": version, "private": True,
                                               "archive_sha256": file_digest(archive)})
    write_run_summary(folder, config, summary, version)
    return summary


def main():
    for var in THREAD_ENV:
        os.environ[var] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "collect"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if not folder.is_relative_to(LOCAL_RUNS):
        parser.error("Run folder must be in results/local_runs")
    with local_job():
        if args.action == "submit":
            print(submit(folder))
        else:
            MemoryGuard().check(force=True)
            summary = collect(folder)
            print(f"{summary['experiment']} {summary['status']} · {len(summary['prompts'])} prompts")


if __name__ == "__main__":
    main()
