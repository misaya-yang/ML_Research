"""Summarize the MBPP Instruct PowerMH pilot under its frozen config.

Reads arms.jsonl (PowerMH eager and early-rejection runs, low-temperature
sample, independent-resampling attempts, verdicts), the Power-SMC logs written
by powersmc_mbpp.py, and replay_summary.json from the R0 replay. Grades the
Power-SMC selected programs with the same grader. Applies the frozen budget
rule to the resampling arms. Writes summary.json and summary.md.

Three pruning quantities are kept apart and never mixed:
  cached    kept-prefix positions reused from the KV cache (both PowerMH runs);
  replay    decode positions the R0 replay says early rejection could skip;
  measured  positions and seconds of the actual early-rejection run.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results/local_runs/native-firstpass-1qr7b1va/feedback_policy_v2"))


def iid_at(atts, verdicts, budget):
    spent, included = 0, []
    for att in atts:
        if spent + att["positions"] > budget:
            break
        spent += att["positions"]
        included.append((att, spent))
    if not included:
        return {"solved": False, "returned": None, "positions": 0, "sec": 0.0, "attempts": 0}
    sec = 0.0
    for att, cum in included:
        sec += att["sec"]
        if verdicts[att["code_hash"]]["public"] == "pass":
            return {"solved": verdicts[att["code_hash"]]["all"], "returned": att["a"], "positions": cum,
                    "sec": sec, "attempts": att["a"]}
    first = included[0][0]
    return {"solved": verdicts[first["code_hash"]]["all"], "returned": first["a"], "positions": included[-1][1],
            "sec": sec, "attempts": len(included), "fallback": True}


def main():
    run = Path(sys.argv[1])
    cfg = json.loads((run / "config.json").read_text())
    recs = [json.loads(l) for l in (run / "arms.jsonl").read_text().splitlines()]
    rows = {r["row"]["task_id"]: r["row"] for r in json.loads((run / "mbpp_source.json").read_text())["rows"]}
    from grade_batch import grade

    smc = {}
    for p in sorted(run.glob("powersmc_*.jsonl")):
        lines = [json.loads(l) for l in p.read_text().splitlines()]
        n = next(l["params"]["particles"] for l in lines if l.get("stage") == "load")
        scored = {l["task"]: l for l in lines if l.get("stage") == "scored"}
        jobs = []
        for tid, l in scored.items():
            for j, test in enumerate(rows[tid]["test_list"]):
                jobs.append([f"{tid}/{j}", l["code"], rows[tid].get("test_setup_code", ""), test])
        verdict = {r[0]: r[1] for r in grade(jobs)[0]} if jobs else {}
        smc[n] = {tid: {"solved": all(verdict[f"{tid}/{j}"] == "pass" for j in range(len(rows[tid]["test_list"]))),
                        "public": verdict.get(f"{tid}/0"),
                        "positions": l["executed_decode_positions"] + 1, "sec": l["seconds"]}
                  for tid, l in scored.items()}

    table, only, losses = [], [], []
    tot = {k: 0 for k in ("pmh_pos", "er_pos", "pmh_sec", "er_sec", "kept", "identical", "stopped")}
    for r in recs:
        v = r["verdicts"]
        tid = r["task"]
        pmh = r["powermh"]
        er = r["powermh_early_rejection"]
        B = pmh["positions"]
        row = {"task": tid,
               "low_temp": v[r["low_temp"]["code_hash"]]["all"],
               "powermh": v[pmh["code_hash"]]["all"],
               "powermh_public": v[pmh["code_hash"]]["public"],
               "powermh_changed_program": pmh["code_hash"] != r["low_temp"]["code_hash"],
               "accepted_moves": pmh["accepted"],
               "pmh_positions": B, "pmh_sec": round(pmh["sec"], 2),
               "er_positions": er["positions"], "er_sec": round(er["sec"], 2),
               "identical": r["pathwise_identical_states"] and r["final_identical"],
               "max_abs_logz_diff": r["max_abs_logz_diff"]}
        for name, atts in r["iid"].items():
            for tag, factor in (("1x", 1), ("2x", 2), ("all", 10**9)):
                res = iid_at(atts, v, factor * B if factor < 10**9 else 10**12)
                row[f"{name}@{tag}"] = res["solved"]
                if tag == "1x":
                    row[f"{name}@1x_positions"] = res["positions"]
                    row[f"{name}@1x_sec"] = round(res["sec"], 2)
        for n, d in smc.items():
            if tid in d:
                row[f"power_smc_{n}"] = d[tid]["solved"]
                row[f"power_smc_{n}_positions"] = d[tid]["positions"]
                row[f"power_smc_{n}_sec"] = round(d[tid]["sec"], 2)
        baselines = ["low_temp", "restart_q@1x", "greedy_then_t1@1x"] + [f"power_smc_{n}" for n in smc]
        if row["powermh"] and not any(row.get(b, False) for b in baselines):
            only.append(tid)
        if not row["powermh"] and any(row.get(b, False) for b in baselines):
            losses.append((tid, [b for b in baselines if row.get(b, False)]))
        table.append(row)
        tot["pmh_pos"] += B
        tot["er_pos"] += er["positions"]
        tot["pmh_sec"] += pmh["sec"]
        tot["er_sec"] += er["sec"]
        tot["kept"] += pmh["kept_cached_positions"]
        tot["identical"] += row["identical"]
        tot["stopped"] += er["stopped_early"]

    arms = ["low_temp", "powermh"] + [f"{n}@{t}" for n in cfg["iid_arms"] for t in ("1x", "2x", "all")] + \
           [f"power_smc_{n}" for n in sorted(smc)]
    solved = {a: sum(bool(t.get(a)) for t in table) for a in arms}
    replay = json.loads((run / "replay_summary.json").read_text()) if (run / "replay_summary.json").exists() else None
    summary = {
        "run": run.name, "config_sha256": (run / "config.sha256").read_text().strip(), "problems": len(table),
        "solved_counts": solved,
        "only_powermh": only,
        "powermh_misses_solved_by_a_baseline": losses,
        "powermh_changed_program_vs_low_temp": sum(t["powermh_changed_program"] for t in table),
        "pruning": {
            "pathwise_identical_problems": f"{tot['identical']}/{len(table)}",
            "cached_kept_prefix_positions": tot["kept"],
            "replay_estimate": None if replay is None else {
                k: replay[k] for k in ("decode_positions", "headroom_raw", "headroom_cached", "decision_mismatches",
                                       "final_state_rebuilt", "rejected", "accepted", "proposals",
                                       "rejection_decided_fraction_q0_25_50_75_100",
                                       "numerics_max_abs_logz_at_cut")},
            "measured_early_rejection_run": {
                "positions_eager": tot["pmh_pos"], "positions_early_rejection": tot["er_pos"],
                "positions_saved_fraction": 1 - tot["er_pos"] / tot["pmh_pos"] if tot["pmh_pos"] else None,
                "seconds_eager": round(tot["pmh_sec"], 2), "seconds_early_rejection": round(tot["er_sec"], 2),
                "seconds_saved_fraction": 1 - tot["er_sec"] / tot["pmh_sec"] if tot["pmh_sec"] else None,
                "proposals_stopped_early": tot["stopped"]},
        },
        "per_problem": table,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    cols = ["task", "low_temp", "powermh", "restart_q@1x", "greedy_then_t1@1x", "restart_q@all",
            "greedy_then_t1@all"] + [f"power_smc_{n}" for n in sorted(smc)] + \
           ["accepted_moves", "powermh_changed_program", "pmh_positions", "er_positions", "pmh_sec", "er_sec",
            "restart_q@1x_positions", "greedy_then_t1@1x_positions"] + \
           [f"power_smc_{n}_positions" for n in sorted(smc)] + ["identical"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        md.append("| " + " | ".join(("✓" if t.get(c) is True else "·" if t.get(c) is False else str(t.get(c, "")))
                                     for c in cols) + " |")
    (run / "summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_problem"}, indent=1))


if __name__ == "__main__":
    main()
