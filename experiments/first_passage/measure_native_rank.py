"""Measure lazy first verification on a frozen local four-particle MBPP run.

Supply the existing model directory, feedback-policy archive, and a new output
path. The public and hidden verdicts come from exact program matches in the
archive. An unavailable public verdict makes the case unresolved. The timed
arms share the SMC release protocol, single-row terminal kernel and prefix
cache. The separately timed archived compact run provides a batching reference.
"""

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from lazy_rank import Candidate, Result, first_verified


class MissingVerdict(LookupError):
    pass


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class ArchiveVerifier:
    def __init__(self, verdicts, hidden_tests):
        self.verdicts = verdicts
        self.hidden_tests = hidden_tests
        self.lookup_s = 0.0
        self.archived_public_s = 0.0
        self.checked = []

    def __call__(self, payload):
        start = time.perf_counter()
        result = self.verdicts.get((payload, 0))
        self.lookup_s += time.perf_counter() - start
        self.checked.append(payload)
        if result is None:
            raise MissingVerdict(payload)
        status, elapsed = result
        self.archived_public_s += elapsed
        return status == "pass"

    def outcome(self, result):
        if not result:
            return "missing_public_verdict"
        if result.answer is None:
            return "abstain"
        statuses = [self.verdicts.get((result.answer, i)) for i in self.hidden_tests]
        if any(status is None for status in statuses):
            return "missing_hidden_verdict"
        return "correct" if all(status[0] == "pass" for status in statuses) else "wrong"


def read_archive(archive):
    source = archive / "mbpp_source.json"
    ledger = archive / "power_race_triple_4_compact_budget4.jsonl"
    grading = archive / "population_extraction.json"
    rows = json.loads(source.read_text())["rows"]
    states = [json.loads(line) for line in ledger.read_text().splitlines()]
    params = next(record["params"] for record in states if record["stage"] == "loaded")
    references = {record["task"]: record for record in states if record["stage"] == "paired"}
    expected = [item["row"]["task_id"] for item in rows]
    if set(expected) != references.keys():
        raise ValueError("Task manifest differs from the completed four-particle ledger")
    grades = next(record for record in json.loads(grading.read_text()) if record["params"] == params)
    verdicts = {task: {} for task in expected}
    for key, status, _, elapsed in grades["grades"]:
        task, index, test = (int(part) for part in key.split("/"))
        payload = references[task]["full"]["population_codes"][index]
        prior = verdicts[task].setdefault((payload, test), (status, elapsed))
        if prior[0] != status:
            raise ValueError(f"Conflicting archived verdicts for task {task}")
    return rows, references, verdicts, params, (source, ledger, grading)


def prepare_terminal(engine, race):
    """Complete all potentially state-changing resampling before the race."""
    cfg, slots, horizon = engine.cfg, engine.slots, engine.cap
    all_slots = list(range(engine.n))
    last_check = (horizon - 1) // cfg.block * cfg.block
    ancestry = []
    for step in range(1, last_check + 1):
        engine.advance(all_slots, step - 1, step)
        if step % cfg.block == 0:
            probabilities = race.weights([slot.w for slot in slots])
            if 1.0 / sum(p * p for p in probabilities) < cfg.ess * engine.n:
                parents = race.ancestors(probabilities, engine.offset[step - 1])
                engine.slots = slots = [replace(slots[i], w=0.0) for i in parents]
                ancestry.append((step, parents))
        if step == last_check:
            return step, ancestry
        if any(slot.done for slot in slots):
            clear = race.release_envelope(
                [slot.w for slot in slots], [slot.lp for slot in slots],
                [slot.done for slot in slots], step, horizon, cfg.block, cfg.ess,
                lambda t: race.exponent(t, cfg), check_at_cap=False,
            )
            if clear is not None:
                return step, ancestry
    return 0, ancestry


class TerminalDecoder:
    def __init__(self, engine, release, race, wave):
        self.engine = engine
        self.race = race
        self.wave = wave
        self.steps = [release] * engine.n
        self.prefix_cache = {}
        self.cached_positions = 0
        self.bound_s = 0.0
        self.stop_by_prefix = {}

    def candidate(self, i):
        engine = self.engine
        slot = engine.slots[i]
        step = self.steps[i]
        start = time.perf_counter()
        bound = self.race.final_bound(slot.w, slot.lp, step, engine.cap, engine.cfg)
        self.bound_s += time.perf_counter() - start
        if not slot.done and step < engine.cap:
            return Candidate(bound)
        text = engine.tok.decode(list(slot.tape), skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
        return Candidate(bound, engine.nat.clean(text))

    def advance(self, i):
        engine = self.engine
        slot = engine.slots[i]
        step = self.steps[i] + 1
        cfg, stats = engine.cfg, engine.stats
        context = slot.tape
        alpha = self.race.exponent(step, cfg)
        slot.w += (alpha - self.race.exponent(step - 1, cfg)) * slot.lp
        if not context:
            logits, store = engine.prompt_logits, slot.store
        elif context in self.prefix_cache:
            logits, store = self.prefix_cache[context]
            self.cached_positions += 1
        else:
            start = time.perf_counter()
            torch = engine.torch
            # Match this row shape on every call, whichever arm requests it.
            cache = self.wave.gather_cache([(slot.store, slot.row)], torch, engine.device)
            ids = torch.tensor([[context[-1]]], device=engine.device, dtype=torch.long)
            engine.sync()
            stats["cache_reorder_s"] += time.perf_counter() - start
            start = time.perf_counter()
            output = engine.model.model(input_ids=ids, past_key_values=cache, use_cache=True)
            logits = engine.head(output.last_hidden_state[:, -1, :]).float()
            store = self.wave.Store(output.past_key_values, 1)
            engine.sync()
            stats["decode_forward_s"] += time.perf_counter() - start
            stats["decode_calls"] += 1
            stats["executed_decode_positions"] += 1
            self.prefix_cache[context] = logits, store

        start = time.perf_counter()
        sample = self.race.draw_rows_device(logits, [engine.u[step - 1][i]], alpha, engine.torch)
        token, base_logprob, weight_increment = sample[0]
        stats["proposal_s"] += time.perf_counter() - start
        stats["sampled_tokens"] += 1
        stats["executed_proposal_rows"] += 1
        stats["live_decode_positions"] += bool(context)
        slot.lp += base_logprob
        slot.w += weight_increment
        slot.tape += (int(token),)
        slot.store, slot.row = store, 0
        self.steps[i] = step

        start = time.perf_counter()
        if token in engine.nat.stops:
            slot.done, slot.reason = True, "eos"
        else:
            if slot.tape not in self.stop_by_prefix:
                text = engine.tok.decode(list(slot.tape), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
                self.stop_by_prefix[slot.tape] = text.count("```") >= 2
                stats["stop_text_decodes"] += 1
            if self.stop_by_prefix[slot.tape]:
                slot.done, slot.reason = True, "fence"
        stats["stop_s"] += time.perf_counter() - start
        return self.candidate(i)


def eager(candidates, keys, budget, advance, verify):
    for i, candidate in enumerate(candidates):
        while candidate.payload is None:
            candidate = advance(i)
        candidates[i] = candidate
    order = sorted(range(len(keys)), key=lambda i: (-(candidates[i].log_upper + keys[i]), i))
    seen = set()
    checked = []
    for i in order:
        if len(checked) == budget:
            break
        payload = candidates[i].payload
        if payload in seen:
            continue
        seen.add(payload)
        checked.append(payload)
        if verify(payload):
            return Result(payload, i, tuple(checked), ())
    return Result(None, None, tuple(checked), ())


def measure(nat, race, wave, cfg, row, seed, verdicts, budget, lazy):
    start = time.perf_counter()
    verifier = ArchiveVerifier(verdicts, range(1, len(row["test_list"])))
    with nat.torch.inference_mode():
        engine = wave.Engine(nat, row, cfg, seed)
        release, ancestry = prepare_terminal(engine, race)
        common = dict(engine.stats)
        decoder = TerminalDecoder(engine, release, race, wave)
        candidates = [decoder.candidate(i) for i in range(engine.n)]
        select = first_verified if lazy else eager
        try:
            result = select(candidates, engine.g, budget, decoder.advance, verifier)
        except MissingVerdict:
            result = None
        engine.sync()
    elapsed = time.perf_counter() - start
    summarize = lambda payload: hashlib.sha256(payload.encode()).hexdigest()
    return {
        "release": release, "ancestry": ancestry,
        "common": common, "all": engine.stats, "elapsed_s": elapsed,
        "terminal_slots": [
            {"token_count": decoder.steps[i], "complete": slot.done or decoder.steps[i] == engine.cap}
            for i, slot in enumerate(engine.slots)
        ],
        "cached_terminal_positions": decoder.cached_positions,
        "cached_terminal_prefixes": len(decoder.prefix_cache),
        "bound_s": decoder.bound_s, "lookups_s": verifier.lookup_s,
        "archived_public_tests_s": verifier.archived_public_s, "outcome": verifier.outcome(result),
        "checked": [summarize(payload) for payload in verifier.checked],
        "slot": result.slot if result else None,
        "answer_hash": summarize(result.answer) if result and result.answer is not None else None,
        "answer_tokens": list(engine.slots[result.slot].tape) if result and result.slot is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("archive_root", type=Path, help="Path to feedback_policy_v2")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--budget", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--inputs-only", action="store_true", help="Validate archive without loading a model")
    args = parser.parse_args()
    start = time.perf_counter()
    rows, references, verdicts, params, files = read_archive(args.archive_root / "race_heldout")
    archive_setup = time.perf_counter() - start
    if not 1 <= args.cases <= len(rows):
        parser.error("--cases must be between 1 and the number of frozen tasks")
    root = args.archive_root.resolve()
    dependencies = [root / name for name in (
        "power_race_v6.py", "power_race_v5.py", "ess_certificate.py", "resample_certificate.py", "powersmc_mbpp.py",
    )]
    native_code = root.parent / "independent_mh_screen" / "native_sampling.py"
    config = SimpleNamespace(particles=params["n"], alpha=params["alpha"], ramp=params["ramp"],
                             block=params["block"], ess=params["ess"], sampler="device", capture=False)
    metadata = {"stage": "setup", "source": params, "budget": args.budget,
                "ordered_task_ids": [item["row"]["task_id"] for item in rows[:args.cases]],
                "sha256": {str(path): sha256(path) for path in (*files, *dependencies, native_code,
                             Path(__file__), Path(__file__).with_name("lazy_rank.py"))}}
    if args.inputs_only:
        print(json.dumps(metadata, indent=2))
        return
    if args.output.exists():
        parser.error("The output file must be new")
    metadata["model_file_sha256"] = {path.name: sha256(path) for path in sorted(args.model.glob("*.json"))}
    weights = args.model / "model.safetensors"
    metadata["model_weights_sha256"] = sha256(weights)
    sys.path[:0] = [str(root), str(native_code.parent)]
    race = importlib.import_module("power_race_v5")
    wave = importlib.import_module("power_race_v6")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], str(args.model), str(root / "race_heldout"), "measure", "0", str(args.cases)]
        nat = importlib.import_module("native_sampling")
    finally:
        sys.argv = old_argv
    if nat.cap != params["cap"]:
        raise ValueError("Loaded generation horizon differs from the archive")
    metadata.update(model_load_s=nat.load_time, archive_setup_s=archive_setup, torch_version=nat.torch.__version__)
    with args.output.open("x", buffering=1) as stream:
        print(json.dumps(metadata), file=stream)
        for ordinal, item in enumerate(rows[:args.cases]):
            row = item["row"]
            task = row["task_id"]
            reference = references[task]["full"]
            order = ("full", "lazy") if ordinal % 2 == 0 else ("lazy", "full")
            arms = {name: measure(nat, race, wave, config, row, reference["seed"], verdicts[task], args.budget,
                                  lazy=name == "lazy") for name in order}
            full, lazy = arms["full"], arms["lazy"]
            observable = ("checked", "answer_hash", "answer_tokens", "slot", "outcome")
            match = None if any(arm["outcome"] == "missing_public_verdict" for arm in arms.values()) else all(
                full[key] == lazy[key] for key in observable)
            archive = {key: reference[key] for key in ("seconds", "decode_calls", "executed_decode_positions", "prefill_positions")}
            print(json.dumps({"stage": "paired", "task": task, "order": order, "match": match, "arms": arms,
                              "archived_compact_full": archive}), file=stream)
            print(task, match, full["outcome"], lazy["outcome"],
                  full["all"]["executed_decode_positions"], lazy["all"]["executed_decode_positions"], flush=True)


if __name__ == "__main__":
    main()
