#!/usr/bin/env python3
"""Recompute the reader-independent MBPP56 work-accounting bounds.

Standard library only. This does not import the decoder, run a model, or replay
missing weight intervals. The bound is conditional on the archived certificate
events being valid; it checks the frozen source identity and receipt accounting.

Example:
  python audit_reader_independent_certificates.py --output /tmp/reader_check.json

Use --receipt and --source-dir when the original files have a different layout.
An optional --compare-audit checks the numeric claims of the accompanying audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_SOURCES = {
    "fixed_verify_k.py": "f7d46ffc65a773f0d48c782eb924c9d35a808050e8af1f591ad1deeb3adeba90",
    "power_race_v6.py": "b951f5270b7e30e10299b0ebe23878bdd5acf900b06ff84b57ab7ccbd8b435cd",
    "power_race_v5.py": "116b1f9593d25d624e645333400b8c12babc8e7437258563cb17a9216d4e497c",
    "checkpoint_replay.py": "5c487d6e8ebd3386eafb514459f35e1a0282fb530c98617d9b6d77cdade7f8df",
    "ess_certificate.py": "1eb2635c9d505f246d107ba8eb7588a79d353c41a2ddd87817c781d01ae00c60",
    "resample_certificate.py": "a98eaae266cefe7a954d4c1be6c0777f75c4870ef5bd0fc4db3bee6739c0f9d0",
}
FIELDS = (
    "prefill_positions", "executed_decode_positions", "decode_calls",
    "sampled_tokens", "end_to_end_s", "decode_forward_s", "certificate_s",
    "public_verify_s",
)


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    return {
        arm: {
            key: (math.fsum if key.endswith("_s") else sum)(
                row["methods"][arm][key] for row in rows
            )
            for key in FIELDS
        }
        for arm in ("full", "terminal", "across")
    }


def classify(events):
    if any("all_parents" in event for event in events):
        return "whole_parent_map"
    if any("child" in event for event in events):
        return "individual_or_code_equivalence"
    if any(event.get("no_further_resampling") for event in events):
        return "no_future_resampling"
    return "no_event"


def audit(receipt, source_dir):
    source_identity = {}
    for name, expected in EXPECTED_SOURCES.items():
        path = source_dir / name
        actual = digest(path)
        require(actual == expected, "Frozen source mismatch: " + str(path))
        source_identity[name] = actual

    parsed = [
        (number, json.loads(line))
        for number, line in enumerate(receipt.read_text().splitlines(), 1)
        if line.strip()
    ]
    loaded = [row for _, row in parsed if row.get("stage") == "loaded"]
    require(bool(loaded), "Missing loaded contract")
    expected_params = {
        "particles": 64, "k": 4, "alpha": 4.0, "ramp": 100, "block": 64,
        "ess": 0.5, "cap": 192, "capture": False, "layout": "compact",
        "tail_width": 1, "sampler": "device",
    }
    for row in loaded:
        require(all(row["params"].get(k) == v for k, v in expected_params.items()),
                "Receipt is outside the frozen MBPP56 contract")
    records = [(line, row) for line, row in parsed if row.get("stage") == "paired"]
    require(len(records) == 56, "Expected exactly 56 paired records")
    require(len({row["task"] for _, row in records}) == 56, "Duplicate task")
    require(all(row["same_probe_path"] for _, row in records),
            "Observed probe-path mismatch; do not silently exclude it")
    require(all(row["methods"][a]["prefill_calls"] == 1
                for _, row in records for a in ("full", "terminal", "across")),
            "Prefill denominator is no longer one call per request")

    per_task = []
    buckets = {name: [] for name in (
        "whole_parent_map", "individual_or_code_equivalence",
        "no_future_resampling", "no_event",
    )}
    for line, row in records:
        methods = row["methods"]
        full, terminal, across = [methods[a] for a in ("full", "terminal", "across")]
        events = across["events"]
        kind = classify(events)
        buckets[kind].append(row)
        item = {
            "task": row["task"], "receipt_line": line, "category": kind,
            "full_decode_positions": full["executed_decode_positions"],
            "terminal_decode_positions": terminal["executed_decode_positions"],
            "across_decode_positions": across["executed_decode_positions"],
            "full_decode_calls": full["decode_calls"],
            "terminal_decode_calls": terminal["decode_calls"],
            "across_decode_calls": across["decode_calls"],
            "observed_decode_position_difference": full["executed_decode_positions"] - across["executed_decode_positions"],
            "observed_decode_call_difference": full["decode_calls"] - across["decode_calls"],
            "independent_lower_bound_decode_positions": 0,
            "independent_lower_bound_decode_calls": 0,
        }
        if kind == "whole_parent_map":
            require(len(events) == 1, "Mixed events require a new causal attribution")
            event = events[0]
            checks = {
                "checkpoint_64": event["s"] == 64,
                "before_checkpoint": 1 <= event["t"] < event["s"],
                "complete_map": len(event["all_parents"]) == 64,
                "valid_parent_indices": all(0 <= p < 64 for p in event["all_parents"]),
                "remaining_count": event["remaining"] == len(set(event["all_parents"])),
                "reference_63_decode_calls": full["decode_calls"] == 63,
                "no_call_after_certificate": across["decode_calls"] == event["t"] - 1,
                "terminal_equals_full_positions": terminal["executed_decode_positions"] == full["executed_decode_positions"],
                "terminal_equals_full_calls": terminal["decode_calls"] == full["decode_calls"],
                "terminal_equals_full_sampled_tokens": terminal["sampled_tokens"] == full["sampled_tokens"],
            }
            require(all(checks.values()), "Whole-map accounting check failed: task " + str(row["task"]))
            item.update(
                certificate_at=event["t"], checkpoint=event["s"],
                distinct_parents=event["remaining"], checks=checks,
                independent_lower_bound_decode_positions=item["observed_decode_position_difference"],
                independent_lower_bound_decode_calls=item["observed_decode_call_difference"],
            )
        per_task.append(item)

    totals = aggregate([row for _, row in records])
    full, across = totals["full"], totals["across"]
    selected = [r for r in per_task if r["category"] == "whole_parent_map"]
    saved_positions = sum(r["independent_lower_bound_decode_positions"] for r in selected)
    saved_calls = sum(r["independent_lower_bound_decode_calls"] for r in selected)
    bound = {
        "n": len(selected), "task_ids": [r["task"] for r in selected],
        "decode_positions_saved": saved_positions, "decode_calls_saved": saved_calls,
        "decode_positions_denominator_full": full["executed_decode_positions"],
        "decode_calls_denominator_full": full["decode_calls"],
        "total_model_calls_denominator_full_including_prefill": full["decode_calls"] + len(records),
        "observed_total_decode_position_saving": full["executed_decode_positions"] - across["executed_decode_positions"],
        "observed_total_decode_call_saving": full["decode_calls"] - across["decode_calls"],
        "fraction_of_full_decode_positions": saved_positions / full["executed_decode_positions"],
        "fraction_of_observed_across_decode_position_reduction": saved_positions / (full["executed_decode_positions"] - across["executed_decode_positions"]),
        "fraction_of_full_decode_calls": saved_calls / full["decode_calls"],
        "fraction_of_full_model_calls_including_prefill": saved_calls / (full["decode_calls"] + len(records)),
        "fraction_of_observed_across_decode_call_reduction": saved_calls / (full["decode_calls"] - across["decode_calls"]),
        "observed_subset_totals": aggregate(buckets["whole_parent_map"]),
    }

    # A separate, explicitly named extension: all 64 exact child certificates
    # jointly form a complete map at one unchanged state. No code equivalence.
    extra_row = next(row for _, row in records if row["task"] == 468)
    extra_methods = extra_row["methods"]
    events = extra_methods["across"]["events"]
    extra_checks = {
        "all_children_in_order": [e.get("child") for e in events] == list(range(64)),
        "all_exact_parent": all("parent" in e and "possible_done" not in e for e in events),
        "same_clock": all(e["t"] == 119 and e["s"] == 128 for e in events),
        "forced_resampling": all(e["resampling"] and not e["identity"] for e in events),
        "unchanged_unknown": all(e["unknown_before"] == 59 for e in events),
        "five_known_parents": len({e["parent"] for e in events}) == 5,
        "no_later_call": extra_methods["across"]["decode_calls"] == 118,
        "same_full_terminal_work": extra_methods["full"]["executed_decode_positions"] == extra_methods["terminal"]["executed_decode_positions"],
    }
    require(all(extra_checks.values()), "Task468 complete-map extension no longer supported")
    extra_positions = extra_methods["full"]["executed_decode_positions"] - extra_methods["across"]["executed_decode_positions"]
    extra_calls = extra_methods["full"]["decode_calls"] - extra_methods["across"]["decode_calls"]
    return {
        "audit_kind": "receipt_based_reader_independent_accounting_v1",
        "proof_limit": "Frozen source plus aggregate work accounting; original intervals and full numeric states are absent. This is not a new model run or independent numeric-certificate replay.",
        "receipt": str(receipt), "receipt_sha256": digest(receipt),
        "source_sha256": source_identity, "receipt_cases": len(records),
        "totals": totals, "conservative_whole_map_lower_bound": bound,
        "category_summaries": {
            kind: {
                "n": len(rows), "tasks": [r["task"] for r in rows],
                "observed_decode_positions_difference": sum(r["methods"]["full"]["executed_decode_positions"] - r["methods"]["across"]["executed_decode_positions"] for r in rows),
                "observed_decode_calls_difference": sum(r["methods"]["full"]["decode_calls"] - r["methods"]["across"]["decode_calls"] for r in rows),
                "observed_end_to_end_difference_s": math.fsum(r["methods"]["full"]["end_to_end_s"] - r["methods"]["across"]["end_to_end_s"] for r in rows),
            }
            for kind, rows in buckets.items()
        },
        "additional_complete_slot_map_certificate": {
            "task": 468, "checks": extra_checks,
            "decode_positions_saved": extra_positions, "decode_calls_saved": extra_calls,
            "combined_supported_decode_position_lower_bound": saved_positions + extra_positions,
            "combined_supported_decode_call_lower_bound": saved_calls + extra_calls,
            "full_end_to_end_s": extra_methods["full"]["end_to_end_s"],
            "across_end_to_end_s": extra_methods["across"]["end_to_end_s"],
        },
        "weighted_accuracy": None, "weighted_latency_s": None,
        "per_task": per_task,
    }


def compare_claims(result, existing):
    require(result["receipt_cases"] == existing["receipt_cases"], "Case count differs")
    for arm, values in result["totals"].items():
        for key, value in values.items():
            target = existing["totals"][arm][key]
            require(math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-10),
                    "Total differs: " + arm + "." + key)
    for key, value in result["conservative_whole_map_lower_bound"].items():
        if key in existing["conservative_whole_map_lower_bound"]:
            target = existing["conservative_whole_map_lower_bound"][key]
            require(value == target, "Primary bound differs: " + key)
    for key in ("decode_positions_saved", "decode_calls_saved",
                "combined_supported_decode_position_lower_bound",
                "combined_supported_decode_call_lower_bound"):
        require(result["additional_complete_slot_map_certificate"][key] ==
                existing["additional_complete_slot_map_certificate"][key],
                "Task468 extension differs: " + key)
    previous = {r["task"]: r for r in existing["per_task"]}
    for row in result["per_task"]:
        for key in ("category", "observed_decode_position_difference",
                    "observed_decode_call_difference", "independent_lower_bound_decode_positions",
                    "independent_lower_bound_decode_calls"):
            require(row[key] == previous[row["task"]][key],
                    "Per-task claim differs: " + str(row["task"]) + "." + key)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=here / "raw/fixed_verify_22005f2708c3.jsonl")
    parser.add_argument("--source-dir", type=Path, default=here / "code")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-audit", type=Path)
    args = parser.parse_args()
    require(args.output.resolve() != args.receipt.resolve(), "Output must not overwrite source receipt")
    if args.compare_audit:
        require(args.output.resolve() != args.compare_audit.resolve(), "Comparison audit must stay read-only")
    result = audit(args.receipt, args.source_dir)
    if args.compare_audit:
        compare_claims(result, json.loads(args.compare_audit.read_text()))
        result["existing_audit_numeric_claims_match"] = True
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    primary = {k: v for k, v in result["conservative_whole_map_lower_bound"].items()
               if k not in ("task_ids", "observed_subset_totals")}
    print(json.dumps({"output": str(args.output), "primary": primary,
                      "additional_task468": result["additional_complete_slot_map_certificate"],
                      "comparison_match": result.get("existing_audit_numeric_claims_match")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
