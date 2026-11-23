"""Audit the archived 24-task MBPP populations without model or verifier calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


RACE_PHASES = {4: "budget4", 64: "published64"}
MAX_DISTINCT_PROBES = 4


def read(path, provenance, *, lines=False):
    data = path.read_bytes()
    provenance[str(path)] = hashlib.sha256(data).hexdigest()
    if lines:
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    return json.loads(data)


def ranked_probes(population, budget, *, fixed=False):
    indices = population["population_indices"]
    codes = population["population_codes"]
    rank = range(len(indices)) if fixed else population["population_rank"]
    if set(rank) != set(range(len(indices))) or len(rank) != len(indices):
        raise ValueError("Population rank must be a permutation of the slots")
    if not fixed and rank[0] != population["winner"]:
        raise ValueError("The ranked population must start with the recorded winner")
    if codes[indices[population["winner"]]] != population["code"]:
        raise ValueError("The winning slot has an inconsistent program")
    seen = set()
    probes = []
    for slot in rank:
        index = indices[slot]
        if codes[index] in seen:
            continue
        seen.add(codes[index])
        probes.append((slot, index))
        if len(probes) == budget:
            break
    return probes


def stop(task, tests, probes, grades):
    worker_seconds = 0.0
    tested_slots = []
    for slot, index in probes:
        tested_slots.append(slot)
        public, seconds = grades[task, index, 0]
        worker_seconds += seconds
        if public == "pass":
            correct = all(grades[task, index, test][0] == "pass" for test in range(1, tests))
            return {"outcome": "correct" if correct else "wrong", "selected_slot": slot,
                    "tested_slots": tested_slots, "public_worker_seconds": worker_seconds}
    return {"outcome": "abstain", "selected_slot": None, "tested_slots": tested_slots,
            "public_worker_seconds": worker_seconds}


def tail_hindsight(population, cutoff, tested_slots):
    """Bound the work to continue only tested slots after a resampling-free checkpoint.

    The full reference tape gives physical groups and live masks, but does not give the
    partition of live slots into physical groups at each step. Final token tape aliases
    provide additional same-prefix equalities in this window. Within each step, for g
    full groups, a needed classes and b other classes the retained group count is at
    least max(1, g-b) and at most min(g, a) when a>0. This is a hindsight bound with
    future ranks, oracle verifier outcomes, ideal cache reuse, and zero scheduling cost.
    """
    resampling = population["resampling"]
    if any(event["t"] > cutoff for event in resampling):
        raise ValueError("Slot identities can change after the selected checkpoint")
    indices = population["population_indices"]
    needed = {indices[slot] for slot in tested_slots}
    full_positions = minimum = maximum = full_calls = needed_calls = 0
    for step, physical, active_mask in population["decode_trace"]:
        if step <= cutoff:
            continue
        live = {index for slot, index in enumerate(indices) if active_mask & (1 << slot)}
        if not 1 <= physical <= len(live):
            raise ValueError("Identical final token tapes disagree with physical group count")
        full_calls += 1
        full_positions += physical
        active_needed = live & needed
        if not active_needed:
            continue
        lower = max(1, physical - len(live - needed))
        upper = min(physical, len(active_needed))
        if lower > upper:
            raise ValueError("Inconsistent physical-group bounds")
        minimum += lower
        maximum += upper
        needed_calls += 1
    return {"cutoff": cutoff, "full_positions": full_positions, "full_calls": full_calls,
            "minimum_retained_positions": minimum, "maximum_retained_positions": maximum,
            "minimum_saved_positions": full_positions - maximum,
            "maximum_saved_positions": full_positions - minimum,
            "saved_depths": full_calls - needed_calls}


def ancestor_windows(full, pruned):
    """Count measured physical position/call differences only in certified parent windows."""
    previous = 0
    saved_positions = saved_calls = 0
    windows = pruned["certified_checkpoints"]
    full_resamples = {item["t"] for item in full["resampling"]}
    for window in windows:
        start, end = window["start"], window["to"]
        if not previous <= start < end <= pruned["release_step"] or end not in full_resamples:
            raise ValueError("A parent certificate overlaps an uncertified or terminal window")
        previous = end
        full_cost = [cost for step, cost, _ in full["decode_trace"] if start < step <= end]
        pruned_cost = [cost for step, cost, _ in pruned["decode_trace"] if start < step <= end]
        saved_positions += sum(full_cost) - sum(pruned_cost)
        saved_calls += len(full_cost) - len(pruned_cost)
    return {"windows": len(windows), "saved_positions": saved_positions, "saved_calls": saved_calls}


def result_summary(items, keys):
    return {key: sum(item[key] for item in items) for key in keys}


def compare(rows, comparator):
    transitions = Counter((row["weighted"]["outcome"], comparator[row["task"]]) for row in rows)
    weighted_correct = {row["task"] for row in rows if row["weighted"]["outcome"] == "correct"}
    comparator_correct = {task for task, outcome in comparator.items() if outcome == "correct"}
    return {"weighted_wins": sorted(weighted_correct - comparator_correct),
            "weighted_losses": sorted(comparator_correct - weighted_correct),
            "weighted_vs_comparator": {f"{left}/{right}": n for (left, right), n in transitions.items()}}


def correct_by_budget(rows, portfolio, direct_policies):
    """Compare every integer budget at which a delivery count can change.

    The weighted arm gets its cheapest possible continuation under the studied
    ancestor windows and terminal cutoff. The direct arms reuse the prompt on
    retries. A point on the direct envelope is one fixed policy for all tasks.
    """
    projected = {name: [] for name in direct_policies}
    optimistic = []
    for row in rows:
        task = str(row["task"])
        full = row["measured_single_output_costs"]["full"]
        prefill = full["prefill_positions"]
        saved = (row["measured_ancestor_windows"]["saved_positions"]
                 + row["early_release_hindsight"]["maximum_saved_positions"])
        optimistic.append((prefill + full["executed_decode_positions"] - saved,
                           row["weighted"]["outcome"]))
        for name in direct_policies:
            direct = portfolio[name][task]
            total = prefill + direct["cost"]["executed_decode_positions"]
            projected[name].append((total, direct["outcome"]))
    budgets = {0}
    for arm in [optimistic, *projected.values()]:
        budgets.update(cost for cost, _ in arm)

    def delivered(arm, limit):
        return sum(cost <= limit and outcome == "correct" for cost, outcome in arm)

    curve = []
    for limit in sorted(budgets):
        counts = {name: delivered(arm, limit) for name, arm in projected.items()}
        correct = delivered(optimistic, limit)
        curve.append({"budget": limit, "weighted_correct_upper": correct, "direct_correct": counts,
                      "gap_to_best_direct": correct - max(counts.values())})
    return {"axis": "per-question prefill plus actual decode input positions with stated projections",
            "breakpoints": curve, "max_correct_gap": max(point["gap_to_best_direct"] for point in curve)}


def analyze(archive, particles=4):
    provenance = {}
    race = archive / "race_heldout"
    holdout = archive / "holdout"
    race_source = read(race / "mbpp_source.json", provenance)["rows"]
    holdout_source = read(holdout / "mbpp_source.json", provenance)["rows"]
    protocol = read(race / "timing_protocol.json", provenance)
    if race_source != [holdout_source[i] for i in protocol["selection"]]:
        raise ValueError("The race population must be the prespecified subset of the holdout")
    sources = {item["row"]["task_id"]: item["row"] for item in race_source}
    phase = RACE_PHASES[particles]
    ledger = read(race / f"power_race_triple_{particles}_compact_{phase}.jsonl", provenance, lines=True)
    configs = [record["params"] for record in ledger if record["stage"] == "loaded"]
    params = configs[0]
    if any(config != params for config in configs):
        raise ValueError(f"The {particles}-particle run changed configuration")
    recorded_rows = [record for record in ledger if record["stage"] == "paired"]
    recorded = {record["task"]: record for record in recorded_rows}
    if (len(recorded_rows) != len(sources) or set(recorded) != set(sources)
            or len(sources) != protocol["n"] or params["n"] != particles or params["phase"] != phase):
        raise ValueError(f"Incomplete, duplicate, or mismatched {particles}-particle receipt")
    extraction = read(race / "population_extraction.json", provenance)
    extraction, = [entry for entry in extraction if entry["params"] == params]
    extracted_rows = {record["task"]: record for record in extraction["rows"]}
    if len(extraction["rows"]) != len(sources) or set(extracted_rows) != set(sources):
        raise ValueError("The grades must cover the identical complete set of tasks")
    grades = {}
    for label, status, _, seconds in extraction["grades"]:
        key = tuple(map(int, label.split("/")))
        if status == "infra_error" or key in grades:
            raise ValueError(f"Unusable or duplicate result: {key}")
        grades[key] = status, seconds
    measures = ("decode_calls", "executed_decode_positions", "prefill_positions",
                "terminal_decode_calls", "terminal_executed_positions", "sampled_tokens",
                "seconds", "decode_forward_s", "public_verifier_s", "end_to_end_s")
    rows = []
    for task, record in recorded.items():
        full = record["full"]
        tests = len(sources[task]["test_list"])
        if full["last_boundary"] != params["cap"] - params["block"]:
            raise ValueError("Unexpected terminal checkpoint")
        if len(full["population_indices"]) != params["n"]:
            raise ValueError("Incorrect slot count")
        weighted_order = ranked_probes(full, MAX_DISTINCT_PROBES)
        fixed_order = ranked_probes(full, MAX_DISTINCT_PROBES, fixed=True)
        distinct = len(set(full["population_codes"]))
        if distinct != extracted_rows[task]["unvalidated_unique"]:
            raise ValueError(f"The distinct program count changed on {task}")
        for _, index in set(weighted_order + fixed_order):
            if any((task, index, test) not in grades for test in range(tests)):
                raise ValueError(f"The bounded probe grade is missing on task {task}, program {index}")
        weighted = stop(task, tests, weighted_order, grades)
        fixed = stop(task, tests, fixed_order, grades)
        for label, result in (("weighted", weighted), ("fixed", fixed)):
            archived = extracted_rows[task]
            if result["outcome"] != archived[label] or len(result["tested_slots"]) != archived[f"{label}_probes"]:
                raise ValueError(f"Stored {label} extraction disagrees with the replay on {task}")
        winner_only = stop(task, tests, weighted_order[:1], grades)
        if winner_only["outcome"] != record["outcome"]["full"]:
            raise ValueError(f"Winner-only outcome disagrees with the selected program on {task}")
        winner = full["winner"]
        winner_index = full["population_indices"][winner]
        for result in (weighted, fixed, winner_only):
            slot = result["selected_slot"]
            result["program_sha256"] = None
            if slot is not None:
                code_index = full["population_indices"][slot]
                code = full["population_codes"][code_index]
                result["program_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        for label, status, _, _ in record["gradings"]:
            source, test_str = label.split("/")
            if source == "full" and status != grades[task, winner_index, int(test_str)][0]:
                raise ValueError(f"The independent grade receipts disagree on the winner for {task}")
        costs = {}
        for arm in ("full", "terminal", "pruned"):
            cost = {**record[arm], "public_verifier_s": record["public_verifier_wall_s"][arm],
                    "end_to_end_s": record["end_to_end_s"][arm]}
            costs[arm] = {key: cost[key] for key in measures}
        rows.append({"task": task, "weighted": weighted, "fixed": fixed, "winner_only": winner_only,
                     "winner_matches": record["match_by_arm"], "measured_single_output_costs": costs,
                     "native_terminal_hindsight": tail_hindsight(
                         full, full["last_boundary"], weighted["tested_slots"]),
                     "early_release_hindsight": tail_hindsight(
                         full, record["pruned"]["release_step"], weighted["tested_slots"]),
                     "measured_ancestor_windows": ancestor_windows(full, record["pruned"])})

    summary = {"tasks": len(rows), "params": params, "distinct_probe_budget": MAX_DISTINCT_PROBES}
    for label in ("weighted", "fixed", "winner_only"):
        summary[label] = {"outcomes": dict(Counter(row[label]["outcome"] for row in rows)),
                          "public_probes": sum(len(row[label]["tested_slots"]) for row in rows),
                          "public_worker_seconds": sum(row[label]["public_worker_seconds"] for row in rows)}
    summary["observed_weighted_vs_fixed"] = compare(rows, {row["task"]: row["fixed"]["outcome"] for row in rows})
    summary["weighted_vs_fixed_different_program_tasks"] = [
        row["task"] for row in rows
        if row["weighted"]["program_sha256"] != row["fixed"]["program_sha256"]]
    summary["observed_weighted_vs_winner_only"] = compare(
        rows, {row["task"]: row["winner_only"]["outcome"] for row in rows})
    summary["measured_single_output_costs"] = {
        arm: result_summary([row["measured_single_output_costs"][arm] for row in rows], measures)
        for arm in ("full", "terminal", "pruned")}
    summary["winner_identity_and_tokens_match"] = {
        arm: sum(row["winner_matches"][arm] for row in rows) for arm in ("terminal", "pruned")}
    for field in ("native_terminal_hindsight", "early_release_hindsight", "measured_ancestor_windows"):
        summed = [r[field] for r in rows]
        summary[field] = result_summary(summed, [k for k in summed[0] if k != "cutoff"])
    summary["ancestor_and_early_release_hindsight_ceiling_positions"] = (
        summary["measured_ancestor_windows"]["saved_positions"]
        + summary["early_release_hindsight"]["maximum_saved_positions"])

    portfolio = read(holdout / "portfolio_report.json", provenance)["per_task"]
    summary["cheap_same_tasks"] = {}
    for policy, report in portfolio.items():
        matched = {task: report[str(task)] for task in sources}
        summary["cheap_same_tasks"][policy] = {
            "outcomes": dict(Counter(value["outcome"] for value in matched.values())),
            "reconstructed_costs": result_summary([value["cost"] for value in matched.values()], (
                "executed_decode_positions", "prefill_positions", "model_s", "verify_s",
                "executed_target_input_positions", "public_test_cases_executed")),
            **compare(rows, {task: value["outcome"] for task, value in matched.items()})}
    direct_policies = ("greedy", "greedy_iid1", "greedy_iid2", "greedy_iid3", "greedy_iid2_struct")
    for policy in direct_policies:
        # The archived direct arms use the same prompt for all attempts on one task.
        # Reuse the prefill state and logits across retries, charging the full decode work.
        costs = portfolio[policy]
        first = portfolio["greedy"]
        for task in map(str, sources):
            prompt = first[task]["cost"]["prefill_positions"]
            if costs[task]["cost"]["prefill_positions"] != prompt * costs[task]["cost"]["prefill_calls"]:
                raise ValueError(f"Inconsistent prefill cost across {policy} attempts on {task}")
        prefill = summary["measured_single_output_costs"]["full"]["prefill_positions"]
        if prefill != summary["cheap_same_tasks"]["greedy"]["reconstructed_costs"]["prefill_positions"]:
            raise ValueError("The first-direct-prompt cost changed across sources")
        decode = summary["cheap_same_tasks"][policy]["reconstructed_costs"]["executed_decode_positions"]
        summary["cheap_same_tasks"][policy]["shared_prompt_projected_input_positions"] = prefill + decode
    summary["correct_by_input_position_budget"] = correct_by_budget(rows, portfolio, direct_policies)
    extraction_sha = hashlib.sha256(json.dumps(extraction, sort_keys=True).encode()).hexdigest()
    return {"summary": summary, "per_task": rows, "sha256": provenance,
            "extraction_record_sha256": extraction_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the existing feedback_policy_v2 archive")
    parser.add_argument("output", type=Path, help="New JSON report path")
    parser.add_argument("--particles", type=int, choices=tuple(RACE_PHASES), default=4)
    args = parser.parse_args()
    results = analyze(args.archive.resolve(), args.particles)
    with args.output.open("x") as outfile:
        json.dump(results, outfile, indent=2)
        outfile.write("\n")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
