"""Freeze a private Kaggle job and render its single-file script."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

from ..run_io import ROOT, MemoryGuard, freeze, local_job, make_run_dir, verify_frozen


TEMPLATE = Path.home() / "kaggle_t4_template"
IMDB_SOURCE = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
MODELS = {
    "proposal": {"id": "lvwerra/gpt2-imdb", "revision": "f1bfd819c6bee6c18fa5f95bfe88d9198839a435"},
    "reward": {"id": "lvwerra/distilbert-imdb", "revision": "0fc02cd68445b599a9cb2da2368050e7fb31d29a",
               "logit_index": {"negative": 0, "positive": 1}},
}
LIMITS = {("X0", True): 600, ("X0", False): 3000, ("X4", True): 600, ("X4", False): 16800}
NEGATIONS = ["no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "cannot"]


def prompts_from_imdb(archive, cache):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS["proposal"]["id"],
                                               revision=MODELS["proposal"]["revision"],
                                               cache_dir=cache, use_fast=True)
    review_pattern = re.compile(r"aclImdb/(test/(?:neg|pos)/[0-9]+_[0-9]+\.txt)")
    with tarfile.open(archive, mode="r:gz") as tar:
        reviews = {}
        for member in tar:
            match = review_pattern.fullmatch(member.name)
            if match is not None and member.isfile():
                review_id = match[1]
                reviews[review_id] = member
        if len(reviews) != 25000:
            raise ValueError(f"Expected 25,000 IMDb test reviews, found {len(reviews)}")
        rows = []
        ranked = sorted(reviews, key=lambda key: hashlib.sha256(key.encode()).hexdigest())[:12]
        for review_id in ranked:
            raw = tar.extractfile(reviews[review_id]).read().decode("utf-8")
            ids = tokenizer.encode(raw, add_special_tokens=False)[:8]
            if len(ids) != 8 or tokenizer.eos_token_id in ids:
                raise ValueError(f"Invalid prefix for {review_id}")
            digest = hashlib.sha256(review_id.encode()).hexdigest()
            rows.append({"key": digest[:16], "review_id": review_id, "review_id_sha256": digest,
                         "tokens": ids})
    return {
        "archive_source": IMDB_SOURCE,
        "archive_sha256": file_digest(archive),
        "review_id_rule": "UTF-8 original relative path test/{neg,pos}/<id>_<rating>.txt",
        "review_order": "ascending SHA-256 of review_id",
        "prefix_tokens": 8,
        "prompts": rows,
    }


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def _base_config(name, smoke, prompts, source):
    return {
        "phase": 2,
        "experiment": name,
        "smoke": smoke,
        "expected_gpus": 2,
        "accelerator": "NvidiaTeslaT4",
        "private": True,
        "session_timeout_seconds": LIMITS[name, smoke],
        "soft_limit_seconds": LIMITS[name, smoke] - 45,
        "checkpoint_seconds": 600,
        "phase2_max_session_seconds": sum(LIMITS.values()),
        "phase2_max_device_hours": 2 * sum(LIMITS.values()) / 3600,
        "seed": {("X0", True): 20261122001, ("X0", False): 20261122002,
                 ("X4", True): 20261122041, ("X4", False): 20261122042}[name, smoke],
        "models": MODELS,
        "prompts": prompts,
        "prompt_source": source,
        "generation": {"max_new_tokens": 32, "top_k": 0, "top_p": 1, "temperature": 1,
                       "sample_batch": 256, "score_batch": 32, "reward_batch": 128,
                       "chunk_rows": 4096, "eos_rule": "include first EOS in length and log Q"},
        "labels": {"scope": "decoded prompt plus completion, excluding EOS",
                   "negations": NEGATIONS, "negation_contraction": "n't or n’t",
                   "positive_logit_threshold": math.log(19),
                   "h_bit_order": ["negation", "sigmoid(r)>0.95", "exclamation"]},
        "text_hash": "SHA-256 first 8 bytes as big-endian uint64",
    }


def config_for(name, smoke, prompt_spec, passed_x0):
    all_prompts = prompt_spec["prompts"]
    source = {key: val for key, val in prompt_spec.items() if key != "prompts"}
    if name == "X0":
        config = _base_config(name, smoke, all_prompts[:2] if smoke else all_prompts, source)
        config["betas"] = [.5, 1, 2, 4]
        config["screen"] = {"beams": 8 if smoke else 512, "samples": 64 if smoke else 8192,
                            "length_penalty": 0, "max_head_Pmass": .5, "max_dup_rate": .05,
                            "duplication_key": "decoded text hash", "min_pass": 8}
    else:
        x0_config, x0_digest = verify_frozen(passed_x0)
        evidence = json.loads((passed_x0 / "summary.json").read_text())
        if (x0_config["experiment"] != "X0" or x0_config["smoke"] or evidence["smoke"]
                or evidence["status"] != "complete" or evidence["passed"] is not True
                or evidence["config_sha256"] != x0_digest or x0_config["prompts"] != all_prompts):
            raise ValueError("X4 needs verified passing full X0 with the same frozen prompts")
        passing = set(evidence["passing_prompts"])
        prompts = [prompt for prompt in all_prompts if prompt["key"] in passing]
        if len(prompts) < x0_config["screen"]["min_pass"]:
            raise ValueError("X0 passed prompt threshold not met")
        config = _base_config(name, smoke, prompts[:2] if smoke else prompts, source)
        config.update(bank_rows=64 if smoke else 2**18, x0_run=passed_x0.name,
                      x0_config_sha256=x0_digest, x0_summary_sha256=file_digest(passed_x0 / "summary.json"))
    return config


def prepare_job(name, smoke, scratch, date, passed_x0, smoke_run):
    prompt_spec = json.loads((scratch / "prompts.json").read_text())
    config = config_for(name, smoke, prompt_spec, passed_x0)
    source = Path(__file__).with_name("workload.py")
    config["workload_sha256"] = file_digest(source)
    if not smoke:
        tested, tested_digest = verify_frozen(smoke_run)
        report = json.loads((smoke_run / "summary.json").read_text())
        verified = json.loads((smoke_run / "verification.json").read_text())
        if (tested["experiment"] != name or not tested["smoke"] or not report["smoke"]
                or report["status"] != "complete" or report["config_sha256"] != tested_digest
                or verified["private"] is not True or tested["models"] != config["models"]
                or tested["prompts"] != config["prompts"][:len(tested["prompts"])]
                or tested["workload_sha256"] != config["workload_sha256"]):
            raise ValueError("Full run requires the verified SMOKE for this workload and prompt prefix")
        config.update(smoke_run=smoke_run.name, smoke_config_sha256=tested_digest)
    run_dir = make_run_dir(name, date)
    slug = f"imh-audit-{name.lower()}-{'smoke' if smoke else 'full'}-{run_dir.name.split('-', 1)[1]}"
    job = scratch / slug
    job.mkdir(mode=0o700)
    metadata = json.loads((TEMPLATE / "kernel-metadata.json").read_text())
    metadata.update(id=f"mingjieyang1/{slug}", title=slug, is_private=True, enable_gpu=True,
                    enable_internet=True, machine_shape="NvidiaTeslaT4")
    if any(metadata[field] for field in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources")):
        raise ValueError("Unexpected data or notebook attachments in Kaggle template")
    config["kernel"] = metadata["id"]
    digest = freeze(run_dir, config)
    frozen_json = (run_dir / "config.json").read_text()
    script = source.read_text() + f"\n\nif __name__ == '__main__':\n    run({frozen_json!r})\n"
    compile(script, "run.py", "exec")
    write_json(job / "kernel-metadata.json", metadata)
    (job / "run.py").write_text(script)
    write_json(run_dir / "submission.json", {
        "stage_path": str(job), "kernel": metadata["id"], "config_sha256": digest,
        "remote_script_sha256": file_digest(job / "run.py"),
        "metadata_sha256": file_digest(job / "kernel-metadata.json"),
    })
    return run_dir


def main():
    import os

    from ..run_io import THREAD_ENV

    for var in THREAD_ENV:
        os.environ[var] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prompts", "X0", "X4"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--passed-x0", type=Path)
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error("Launch from the experiment repository root")
    if args.action == "X4" and not args.passed_x0:
        parser.error("X4 requires --passed-x0")
    if args.action != "prompts" and not args.smoke and not args.smoke_run:
        parser.error("A full run requires --smoke-run")
    with local_job():
        if args.action == "prompts":
            MemoryGuard().check(force=True)
            payload = prompts_from_imdb(args.scratch / "aclImdb_v1.tar.gz", args.scratch / "hf-cache")
            with (args.scratch / "prompts.json").open("x") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
            print(f"Froze {len(payload['prompts'])} IMDb prefixes")
        else:
            print(prepare_job(args.action, args.smoke, args.scratch, args.date, args.passed_x0, args.smoke_run))


if __name__ == "__main__":
    main()
