"""Compare completed 56-task serving ledgers at common warm deadlines and position caps."""

import argparse
import hashlib
import json
from pathlib import Path


def load_arms(folder):
    fixed_path = folder / "fixed_verify_22005f2708c3.jsonl"
    serial_path = folder / "cached_iid_report.json"
    wave_path = folder / "wave_iid_report.json"
    rows = [json.loads(line) for line in fixed_path.read_text().splitlines()]
    records = [r for r in rows if r["stage"] == "paired"]
    if (len(records) != 56 or len({r["task"] for r in records}) != 56
            or not all(r["same_probe_path"] is True and len(set(r["outcomes"].values())) == 1 for r in records)):
        raise ValueError("Expected 56 complete fixed-policy comparisons with identical probe paths")
    arms = {}
    for name in ("full", "terminal", "across"):
        arms[name] = {}
        for record in records:
            m = record["methods"][name]
            arms[name][str(record["task"])] = (m["end_to_end_s"],
                m["prefill_positions"] + m["executed_decode_positions"], record["outcomes"][name])
    serial = json.loads(serial_path.read_text())["per_task"]
    wave = json.loads(wave_path.read_text())["per_task"]
    for name, data in [(f"serial_{k}", {i: v[str(k)] for i, v in serial.items()})
                       for k in range(9)] + [("parallel_8", wave)]:
        if data.keys() != arms["full"].keys():
            raise ValueError(f"Incomplete comparator {name}")
        arms[name] = {i: (r["wall_s"], r["target_input_positions"], r["outcome"]) for i, r in data.items()}
    if any(row[2] not in {"correct", "wrong", "abstain"} for arm in arms.values() for row in arm.values()):
        raise ValueError("Missing or ambiguous truth grades")
    return arms, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (fixed_path, serial_path, wave_path)}


def compare(arms, axis, presets):
    def sets(arm, limit):
        correct = {i for i, row in arm.items() if row[axis] <= limit and row[2] == "correct"}
        wrong = {i for i, row in arm.items() if row[axis] <= limit and row[2] == "wrong"}
        return correct, wrong

    direct = [k for k in arms if k.startswith(("serial_", "parallel_"))]
    thresholds = sorted({0, *presets, *(row[axis] for arm in arms.values() for row in arm.values())})
    grid, intervals = {}, []
    max_gap = 0
    for n, limit in enumerate(thresholds):
        results = {k: sets(v, limit) for k, v in arms.items()}
        counts = {k: (len(c), len(w), len(arms[k])-len(c)-len(w)) for k, (c, w) in results.items()}
        best = max(direct, key=lambda k: (counts[k][0], -counts[k][1]))
        new = counts["across"][0]
        gap = new - max(counts[k][0] for k in ("full", "terminal", best))
        max_gap = max(max_gap, gap)
        if gap > 0:
            end = thresholds[n+1] if n+1 < len(thresholds) else None
            if intervals and intervals[-1]["end_exclusive"] == limit:
                intervals[-1]["end_exclusive"] = end
                intervals[-1]["max_advantage"] = max(intervals[-1]["max_advantage"], gap)
            else:
                intervals.append({"start_inclusive": limit, "end_exclusive": end, "max_advantage": gap})
        if limit in presets:
            ac, ref = results["across"][0], results[best][0]
            grid[limit] = {"counts_correct_wrong_abstain": counts, "best_direct": best,
                           "correct_only_across": sorted(ac-ref, key=int),
                           "correct_only_best_direct": sorted(ref-ac, key=int),
                           "gap_vs_both_exact_and_direct": gap}
    return {"breakpoints_including_grid": len(thresholds), "max_joint_correct_advantage": max_gap,
            "strictly_ahead_of_both_intervals": intervals, "preset": grid}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    arms, hashes = load_arms(args.source)
    result = {"sha256": hashes, "tasks": len(arms["full"]),
              "wall_seconds": compare(arms, 0, (1, 2, 4, 8, 16, 32, 64)),
              "target_input_positions": compare(arms, 1, ())}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    for axis in ("wall_seconds", "target_input_positions"):
        row = result[axis]
        print(axis, row["max_joint_correct_advantage"], row["strictly_ahead_of_both_intervals"])


if __name__ == "__main__":
    main()
