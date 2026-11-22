"""Standalone Kaggle workload. The submitter appends its frozen config."""

import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


class Checkpoints:
    def __init__(self, root, config_hash, seconds):
        self.root = root
        self.directory = root / "checkpoints"
        self.directory.mkdir()
        self.seconds = seconds
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.counter = 0
        self.started = time.monotonic()
        self.state = {"config_sha256": config_hash, "started_utc": utc_now(),
                      "status": "starting", "prompts": {}}
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)

    def _write(self):
        self.counter += 1
        payload = dict(self.state, sequence=self.counter, saved_utc=utc_now(),
                       elapsed_s=round(time.monotonic() - self.started, 3))
        atomic_json(self.directory / f"ckpt_{self.counter:04d}.json", payload)
        atomic_json(self.root / "checkpoint.json", payload)

    def _heartbeat(self):
        while not self.stop.wait(self.seconds):
            with self.lock:
                self._write()

    def start(self):
        self._write()
        self.worker.start()

    def update(self, key, *, force=False, **values):
        with self.lock:
            entry = self.state["prompts"].setdefault(key, {})
            entry.update(values)
            if force:
                self._write()

    def finish(self, status):
        self.stop.set()
        self.worker.join()
        with self.lock:
            self.state["status"] = status
            self._write()


def seed_for(seed, review_id, batch):
    payload = f"{seed}/{review_id}/{batch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**63


def text_hash(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def sequence_lengths(tokens, eos_id):
    import numpy as np

    eos = tokens == eos_id
    return np.where(eos.any(axis=1), eos.argmax(axis=1) + 1, tokens.shape[1])


class Engine:
    def __init__(self, device, config, torch):
        from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                                  AutoTokenizer)
        import inspect
        import re

        self.torch, self.device, self.config = torch, torch.device(f"cuda:{device}"), config
        proposal, reward = (config["models"][key] for key in ("proposal", "reward"))
        self.tokenizer = AutoTokenizer.from_pretrained(proposal["id"], revision=proposal["revision"])
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward["id"], revision=reward["revision"])
        self.model = AutoModelForCausalLM.from_pretrained(
            proposal["id"], revision=proposal["revision"], torch_dtype=torch.float16
        ).to(self.device).eval()
        self.reward = AutoModelForSequenceClassification.from_pretrained(
            reward["id"], revision=reward["revision"], torch_dtype=torch.float16
        ).to(self.device).eval()
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None or self.model.config.eos_token_id != self.eos:
            raise ValueError("GPT-2 model and tokenizer disagree on EOS")
        parameters = inspect.signature(self.model.forward).parameters
        self.logits_to_keep = next((name for name in ("logits_to_keep", "num_logits_to_keep")
                                    if name in parameters), None)
        terms = "|".join(re.escape(term) for term in config["labels"]["negations"])
        self.negation = re.compile(r"\b(?:" + terms + r")\b|n['’]t\b", re.IGNORECASE)

    def _last_logits(self, inputs, past=None, keep=1, use_cache=True):
        kwargs = {self.logits_to_keep: keep} if self.logits_to_keep else {}
        return self.model(input_ids=inputs, past_key_values=past, use_cache=use_cache, **kwargs)

    def draw(self, prefix, count, seed):
        """Unfiltered categorical draws with log Q recorded during decoding."""
        torch = self.torch
        length = self.config["generation"]["max_new_tokens"]
        rng = torch.Generator(device=self.device).manual_seed(seed)
        inputs = torch.tensor(prefix, device=self.device).expand(count, -1).contiguous()
        tokens = torch.full((count, length), self.eos, dtype=torch.long, device=self.device)
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        logq = torch.zeros(count, dtype=torch.float64, device=self.device)
        past = None
        for step in range(length):
            output = self._last_logits(inputs, past)
            past = output.past_key_values
            log_prob = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
            drawn = torch.multinomial(log_prob.exp(), 1, generator=rng).squeeze(-1)
            increment = log_prob.gather(-1, drawn[:, None]).squeeze(-1)
            logq += torch.where(alive, increment, 0.)
            tokens[:, step] = torch.where(alive, drawn, self.eos)
            alive &= drawn != self.eos
            inputs = tokens[:, step:step + 1].contiguous()
        return tokens.cpu().numpy(), logq.cpu().numpy()

    def _sequence_scores(self, prefix, tokens):
        import numpy as np

        torch = self.torch
        q = []
        length = tokens.shape[1]
        for start in range(0, len(tokens), self.config["generation"]["score_batch"]):
            block = tokens[start:start + self.config["generation"]["score_batch"]]
            generated = torch.as_tensor(block, device=self.device)
            context = torch.tensor(prefix, device=self.device).expand(len(block), -1)
            full = torch.cat((context, generated), dim=1)
            logits = self._last_logits(full[:, :-1], keep=length, use_cache=False).logits
            logits = logits[:, -length:, :].float()
            chosen = logits.gather(-1, generated[..., None]).squeeze(-1)
            step_logq = chosen - torch.logsumexp(logits, dim=-1)
            lengths = torch.as_tensor(sequence_lengths(block, self.eos), device=self.device)
            mask = torch.arange(length, device=self.device)[None, :] < lengths[:, None]
            q.append(torch.where(mask, step_logq, 0.).sum(dim=1, dtype=torch.float64).cpu().numpy())
        return np.concatenate(q)

    def beam_head(self, prefix):
        import numpy as np

        n = self.config["screen"]["beams"]
        limit = self.config["generation"]["max_new_tokens"]
        context = self.torch.tensor([prefix], device=self.device)
        sequences = self.model.generate(
            context, attention_mask=self.torch.ones_like(context), max_new_tokens=limit,
            num_beams=n, num_return_sequences=n, do_sample=False, length_penalty=0.,
            early_stopping="never", eos_token_id=self.eos, pad_token_id=self.eos,
        ).cpu().numpy()[:, len(prefix):]
        if sequences.shape[1] < limit:
            sequences = np.pad(sequences, ((0, 0), (0, limit - sequences.shape[1])),
                               constant_values=self.eos)
        lengths = sequence_lengths(sequences, self.eos)
        unique = {}
        for row, length in zip(sequences, lengths):
            unique.setdefault(tuple(row[:length]), row)
        tokens = np.stack(list(unique.values()))
        return tokens, self._sequence_scores(prefix, tokens)

    def annotate(self, prefix, tokens, logq):
        import numpy as np

        lengths = sequence_lengths(tokens, self.eos)
        texts = self.tokenizer.batch_decode(
            [prefix + row[:length].tolist() for row, length in zip(tokens, lengths)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        rewards = []
        batch = self.config["generation"]["reward_batch"]
        labels = self.config["models"]["reward"]["logit_index"]
        for start in range(0, len(texts), batch):
            encoded = self.reward_tokenizer(
                texts[start:start + batch], return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)
            logits = self.reward(**encoded).logits.float()
            reward = (logits[:, labels["positive"]] - logits[:, labels["negative"]]).clamp(-4, 4)
            rewards.append(reward.cpu().numpy())
        r = np.concatenate(rewards).astype(np.float32, copy=False)
        h = np.asarray([bool(self.negation.search(text)) | (4 if "!" in text else 0)
                        for text in texts], dtype=np.uint8)
        h |= (r.astype(np.float64) > self.config["labels"]["positive_logit_threshold"]).astype(np.uint8) << 1
        return {"r": r, "logq": logq, "text_hash": np.asarray(list(map(text_hash, texts)), dtype=np.uint64),
                "length": lengths.astype(np.uint8), "h": h}


def save_bank(path, arrays):
    import numpy as np

    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def collect_bank(engine, prompt, count, results, partial, check, deadline):
    import numpy as np

    prefix, review_id, key = (prompt[k] for k in ("tokens", "review_id", "key"))
    config = engine.config
    batch_size = config["generation"]["sample_batch"]
    chunk_size = config["generation"]["chunk_rows"]
    completed = 0
    generated = 0
    paths = []
    parts = []
    for start in range(0, count, batch_size):
        if time.monotonic() >= deadline:
            break
        size = min(batch_size, count - start)
        tokens, logq = engine.draw(prefix, size, seed_for(config["seed"], review_id, start))
        arrays = engine.annotate(prefix, tokens, logq)
        parts.append(arrays)
        completed += size
        generated += int(arrays["length"].sum())
        check.update(key, proposal_samples=completed, generated_tokens=generated)
        if completed % chunk_size == 0 or completed == count:
            path = partial / f"{key}-{completed:07d}.npz"
            save_bank(path, {name: np.concatenate([row[name] for row in parts]) for name in arrays})
            paths.append(path)
            parts.clear()
            check.update(key, rows=completed, last_part=path.relative_to(check.root).as_posix())
    if completed < count:
        if parts:
            path = partial / f"{key}-{completed:07d}.npz"
            save_bank(path, {name: np.concatenate([row[name] for row in parts]) for name in parts[0]})
            check.update(key, rows=completed, last_part=path.relative_to(check.root).as_posix())
        return None
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            chunks.append({name: data[name] for name in data.files})
    bank = {name: np.concatenate([part[name] for part in chunks]) for name in chunks[0]}
    if config["experiment"] == "X4":
        path = results / f"bank-{key}.npz"
        save_bank(path, bank)
        return {
            "key": key, "review_id": review_id, "rows": len(bank["r"]),
            "generated_tokens": int(bank["length"].sum()),
            "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **{f"h{i+1}_ones": int(np.count_nonzero(bank["h"] & (1 << i))) for i in range(3)},
        }
    return bank


def screen_prompt(engine, prompt, results, partial, check, deadline):
    import numpy as np

    config = engine.config
    screen = config["screen"]
    head, logq = engine.beam_head(prompt["tokens"])
    rewards = engine.annotate(prompt["tokens"], head, logq)["r"].astype(np.float64)
    check.update(prompt["key"], named_scores=len(head))
    bank = collect_bank(engine, prompt, screen["samples"], results, partial, check, deadline)
    if bank is None:
        return None
    beta = np.asarray(config["betas"], dtype=np.float64)
    zhat = np.exp(bank["r"].astype(np.float64)[:, None] / beta).mean(axis=0)
    pmass = np.exp(logq[:, None] + rewards[:, None] / beta).sum(axis=0) / zhat
    duplicate = 1 - len(np.unique(bank["text_hash"])) / len(bank["r"])
    row = {
        "key": prompt["key"], "review_id": prompt["review_id"], "rows": len(bank["r"]),
        "generated_tokens": int(bank["length"].sum()),
        "head_sequences": len(head), "named_scores": len(head),
        "head_Qmass": float(np.exp(logq).sum()), "dup_rate": float(duplicate),
        "passed": bool(np.all(pmass < screen["max_head_Pmass"]) and duplicate < screen["max_dup_rate"]),
    }
    for b, mass, z in zip(config["betas"], pmass, zhat):
        row[f"head_Pmass_beta_{b:g}"] = float(mass)
        row[f"Zhat_beta_{b:g}"] = float(z)
    return row


def run_gpu(config, root, results, check, deadline, runtime):
    import torch
    import transformers

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    runtime.update(torch=torch.__version__, transformers=transformers.__version__, devices=devices)
    if len(devices) != config["expected_gpus"] or any("T4" not in name for name in devices):
        raise RuntimeError(f"Requested {config['expected_gpus']} T4s, received {devices}")
    print(f"CUDA devices {devices}", flush=True)
    prompts = config["prompts"]
    partial = root / "partial"
    partial.mkdir()
    with check.lock:
        check.state["status"] = "running"
    engines = [Engine(device, config, torch) for device in range(min(len(prompts), len(devices)))]

    def work(index, engine):
        with torch.cuda.device(engine.device), torch.inference_mode():
            for prompt in prompts[index::len(engines)]:
                if time.monotonic() >= deadline:
                    return
                key = prompt["key"]
                start = time.monotonic()
                check.update(key, rows=0, status="running")
                if config["experiment"] == "X0":
                    row = screen_prompt(engine, prompt, results, partial, check, deadline)
                else:
                    row = collect_bank(engine, prompt, config["bank_rows"], results, partial, check, deadline)
                if row is None:
                    check.update(key, force=True, status="time_guard")
                    return
                row.update(device=index, elapsed_s=round(time.monotonic() - start, 3))
                atomic_json(results / f"{key}.json", row)
                check.update(key, force=True, rows=row["rows"], status="completed")
                print(f"{key} completed, {row['rows']} rows in {row['elapsed_s']:.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = [pool.submit(work, index, engine) for index, engine in enumerate(engines)]
        for future in futures:
            future.result()


def run(frozen_json):
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = json.loads(frozen_json)
    root = Path("/kaggle/working")
    digest = hashlib.sha256(frozen_json.encode()).hexdigest()
    (root / "config.json").write_text(frozen_json)
    (root / "config.sha256").write_text(f"{digest}  config.json\n")
    results = root / "results"
    results.mkdir()
    check = Checkpoints(root, digest, config["checkpoint_seconds"])
    check.start()
    runtime, error = {}, None
    try:
        run_gpu(config, root, results, check,
                check.started + config["soft_limit_seconds"], runtime)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    rows = [json.loads(path.read_text()) for prompt in config["prompts"]
            if (path := results / f"{prompt['key']}.json").is_file()]
    status = ("failed" if error else
              "complete" if len(rows) == len(config["prompts"]) else "time_guard")
    check.finish(status)
    if rows:
        with (results / "raw.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    passing = [row["key"] for row in rows if row.get("passed")] if config["experiment"] == "X0" else []
    summary = {
        "experiment": config["experiment"], "smoke": config["smoke"], "status": status,
        "error": error, "config_sha256": digest, "started_utc": check.state["started_utc"],
        "finished_utc": utc_now(), "elapsed_s": round(time.monotonic() - check.started, 3),
        "runtime": runtime, "prompts": rows, "passing_prompts": passing,
        "passed": (len(passing) >= config["screen"]["min_pass"]
                   if config["experiment"] == "X0" and status == "complete" and not config["smoke"]
                   else None),
        **{key: sum(row.get(key, 0) for row in check.state["prompts"].values())
           for key in ("proposal_samples", "named_scores", "generated_tokens")},
        "partial_progress": check.state["prompts"],
    }
    atomic_json(root / "summary.json", summary)
    with zipfile.ZipFile(root / "results_final.zip", "w", zipfile.ZIP_STORED) as archive:
        for name in ("summary.json", "checkpoint.json", "config.json", "config.sha256"):
            archive.write(root / name, name)
        for folder in (results, check.directory):
            for path in sorted(folder.iterdir()):
                archive.write(path, path.relative_to(root))
    print(f"{config['experiment']} {'SMOKE' if config['smoke'] else 'full'} {status}", flush=True)
