"""Analyze frozen Lean verdicts and repair traces without running a prover."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import re


FENCE_PATTERNS = (
    re.compile(r"```lean4\n(.*?)\n```", re.DOTALL),
    re.compile(r"```lean4\n(.*?)```", re.DOTALL),
    re.compile(r"```lean\n(.*?)```", re.DOTALL),
)


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def is_length_cutoff(row):
    return row["diagnostics"].get("cause") == "finish_reason=length"


def analyze_baseline(directory):
    rows = read_jsonl(directory / "ver_minif2f_c16384.jsonl")
    manifest = json.loads((directory / "generation_manifest.json").read_text())
    (session,) = manifest["sessions"]
    assert session["samples_written"] == len(rows)
    cap = session["args"]["cap_tokens"]
    width = session["args"]["n"]
    by_problem = defaultdict(list)
    by_code = defaultdict(list)
    for row in rows:
        assert row["round"] == 0 and row["cap_tokens"] == cap
        by_problem[row["problem_id"]].append(row)
        if row["stage"] != "skipped":
            by_code[row["problem_id"], row["code_sha256"]].append(row)
        if is_length_cutoff(row):
            assert not row["accepted"] and row["verify_s"] == 0

    first_hits = Counter()
    candidates_through_first_hit = []
    candidates_after_first_hit = []
    cutoffs_in_unsolved_problems = 0
    for problem in by_problem.values():
        problem.sort(key=lambda row: row["sample_idx"])
        assert [row["sample_idx"] for row in problem] == list(range(width))
        first = next((index + 1 for index, row in enumerate(problem) if row["accepted"]), None)
        first_hits["failed" if first is None else str(first)] += 1
        stop = width if first is None else first
        if first is None:
            cutoffs_in_unsolved_problems += sum(map(is_length_cutoff, problem))
        candidates_through_first_hit.extend(problem[:stop])
        candidates_after_first_hit.extend(problem[stop:])

    caps_before = sum(map(is_length_cutoff, candidates_through_first_hit))
    caps_after = sum(map(is_length_cutoff, candidates_after_first_hit))
    assert caps_before + caps_after == session["finish_length"]
    for copies in by_code.values():
        copies.sort(key=lambda row: row["sample_idx"])
    repeated = [row for copies in by_code.values() for row in copies[1:]]

    return {
        "problems": len(by_problem),
        "candidates": len(rows),
        "candidate_cap_tokens": cap,
        "accepted_candidates": sum(row["accepted"] for row in rows),
        "first_accepted_candidate_index_1_based": first_hits,
        "decode_tokens_observed_full_batch": session["tokens_generated"],
        "generation_wall_seconds_observed_full_batch": session["generate_wall_s"],
        "engine_load_seconds_observed_full_batch": session["engine_load_s"],
        "candidates_through_first_hit": len(candidates_through_first_hit),
        "candidates_after_first_hit": len(candidates_after_first_hit),
        "verifier_seconds_through_first_hit": round(sum(
            row["verify_s"] for row in candidates_through_first_hit
        ), 4),
        "verifier_seconds_after_first_hit": round(sum(
            row["verify_s"] for row in candidates_after_first_hit
        ), 4),
        "length_cutoffs_through_first_hit": caps_before,
        "length_cutoffs_in_unsolved_problems": cutoffs_in_unsolved_problems,
        "length_cutoffs_after_first_hit": caps_after,
        "length_cutoff_decode_tokens_through_first_hit": caps_before * cap,
        "length_cutoff_decode_tokens_after_first_hit": caps_after * cap,
        "repeated_verified_code_hash_candidates": len(repeated),
        "repeated_verified_code_hash_verifier_seconds": round(sum(
            row["verify_s"] for row in repeated
        ), 4),
        "repeated_verified_code_hash_groups_with_conflicting_acceptance": sum(
            len({row["accepted"] for row in copies}) > 1 for copies in by_code.values()
        ),
    }


def fence_counts(generated):
    counts = Counter()
    for row in generated:
        finish_reason = row["finish_reason"]
        counts[f"{finish_reason}_candidates"] += 1
        for pattern in FENCE_PATTERNS:
            matches = pattern.findall(row["completion_text"])
            if matches:
                counts[f"{finish_reason}_with_selected_fence"] += 1
                if matches[0] != matches[-1]:
                    counts[f"{finish_reason}_with_different_first_and_last_fence"] += 1
                break
        if row["lean_code"] is not None:
            counts[f"{finish_reason}_with_extracted_code"] += 1
    return counts


def analyze_repair_pilot(directory):
    results = {}
    initial_verdicts = {}
    for batch in ("r0", "r1", "r1_n1"):
        generated = read_jsonl(directory / f"pilot_gen_{batch}.jsonl")
        verdicts = read_jsonl(directory / f"pilot_ver_{batch}.jsonl")
        generated_by_attempt = {
            (row["problem_id"], row["round"], row["sample_idx"]): row for row in generated
        }
        verified_by_attempt = {
            (row["problem_id"], row["round"], row["sample_idx"]): row for row in verdicts
        }
        assert len(generated_by_attempt) == len(generated) == len(verdicts)
        assert generated_by_attempt.keys() == verified_by_attempt.keys()
        parents = defaultdict(set)
        if batch == "r0":
            initial_verdicts = verified_by_attempt
        else:
            for row in generated:
                parent = row["parent"]
                key = (parent["problem_id"], parent["round"], parent["sample_idx"])
                assert parent["problem_id"] == row["problem_id"]
                assert not initial_verdicts[key]["accepted"]
                parents[key].add(tuple(row["prompt_ids"]))
            assert all(len(prompt_variants) == 1 for prompt_variants in parents.values())
        results[batch] = {
            "problems": len({row["problem_id"] for row in generated}),
            "candidates": len(generated),
            "accepted_candidates": sum(row["accepted"] for row in verdicts),
            "decode_tokens": sum(len(row["completion_ids"]) for row in generated),
            "failed_parent_attempts": len(parents),
            "parser_diagnostics": fence_counts(generated),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    result = {
        "miniF2F_test_independent_baseline": analyze_baseline(arguments.directory),
        "selected_Goedel_Pset_repair_pilot": analyze_repair_pilot(arguments.directory),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
