"""Analyze an arms.jsonl run: first-answer accuracy, paired wins/regressions,
compute, plan implementation, rewording deltas, and the frozen decision rule.

Plans are exported for blind annotation (task text and plan only, no code or
outcome). If plan_labels.csv (columns task,variant,plan_correct in
{yes,no,unclear}) exists in the run directory, plan-implementation counts are
added: a correct plan whose code fails is an implementation gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def binom_two_sided(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyze(run, cfg):
    recs = [json.loads(x) for x in (run / "arms.jsonl").read_text().splitlines()]
    arms = cfg["arms"]
    prop = cfg["decision"]["proposed"]
    out = {"run": str(run), "variants": {}}
    for variant in cfg["variants"]:
        rs = [r for r in recs if r["variant"] == variant]
        ok = {a: {r["task"]: r["labels"][r["arms"][a]["code_hash"]]["all"] for r in rs} for a in arms}
        comp = {a: [r["arms"][a]["ledger"]["positions"] for r in rs] for a in arms}
        v = {"items": len(rs), "accuracy": {a: sum(ok[a].values()) for a in arms},
             "positions_mean": {a: round(mean(comp[a]), 1) if comp[a] else None for a in arms},
             "positions_median": {a: median(comp[a]) if comp[a] else None for a in arms},
             "exec_calls_mean": {a: round(mean(r["arms"][a]["ledger"]["exec_calls"] for r in rs), 2) if rs else None
                                 for a in arms},
             "wall_s_mean": {a: round(mean(r["arms"][a]["wall_s"] for r in rs), 2) if rs else None for a in arms},
             "paired": {}}
        for b in arms:
            if b == prop:
                continue
            w = [t for t in ok[prop] if ok[prop][t] and not ok[b][t]]
            l = [t for t in ok[prop] if ok[b][t] and not ok[prop][t]]
            v["paired"][b] = {"wins": w, "regressions": l, "net": len(w) - len(l),
                              "exact_two_sided_p": round(binom_two_sided(len(w), len(l)), 4)}
        if prop in arms:
            changed = [r["task"] for r in rs if r["arms"][prop].get("decision") == "fork"]
            v["proposed_changed_answer"] = changed
            v["proposed_forks"] = {r["task"]: r["arms"][prop].get("fork") for r in rs
                                   if r["arms"][prop].get("decision") == "fork"}
        if "plan" in arms:
            v["plan"] = plan_table(run, rs, variant, ok)
        crit = cfg["decision"]["net_wins_min"]
        v["decision_checks"] = {b: {"net": v["paired"][b]["net"], "min": m, "pass": v["paired"][b]["net"] >= m}
                                for b, m in crit.items() if b in v["paired"]}
        tol = cfg["decision"].get("compute_tolerance_vs")
        if tol:
            ref, frac = tol["arm"], tol["max_ratio"]
            ratio = v["positions_mean"][prop] / v["positions_mean"][ref] if v["positions_mean"][ref] else None
            v["decision_checks"]["compute"] = {"ratio_vs_" + ref: round(ratio, 3) if ratio else None, "max": frac,
                                               "pass": ratio is not None and ratio <= frac}
        v["positive"] = all(c["pass"] for c in v["decision_checks"].values())
        out["variants"][variant] = v
    if {"base", "reworded"} <= set(out["variants"]):
        b, r = out["variants"]["base"], out["variants"]["reworded"]
        out["rewording_control"] = {
            "accuracy_delta": {a: r["accuracy"][a] - b["accuracy"][a] for a in arms},
            "proposed_net_base_vs_reworded": {k: (b["paired"][k]["net"], r["paired"][k]["net"]) for k in b["paired"]},
            "survives": b["positive"] and r["positive"]}
    return out


def plan_table(run, rs, variant, ok):
    export = run / f"plans_for_annotation_{variant}.csv"
    with open(export, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "variant", "task_text_in_prompt_is_identical_across_arms", "plan"])
        for r in rs:
            w.writerow([r["task"], variant, "yes", r["arms"]["plan"].get("plan", "")])
    labels = run / "plan_labels.csv"
    res = {"exported": str(export), "annotated": False}
    if labels.exists():
        lab = {(row["task"], row["variant"]): row["plan_correct"] for row in csv.DictReader(open(labels))}
        cells = {"plan_correct_code_correct": 0, "plan_correct_code_wrong": 0, "plan_wrong_code_correct": 0,
                 "plan_wrong_code_wrong": 0, "unclear_or_missing": 0}
        gap = []
        for r in rs:
            pc = lab.get((r["task"], variant))
            cc = ok["plan"][r["task"]]
            if pc not in ("yes", "no"):
                cells["unclear_or_missing"] += 1
                continue
            cells[f"plan_{'correct' if pc == 'yes' else 'wrong'}_code_{'correct' if cc else 'wrong'}"] += 1
            if pc == "yes" and not cc:
                gap.append(r["task"])
        res |= {"annotated": True, "cells": cells, "implementation_gap_tasks": gap}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = json.loads(Path(a.config).read_text())
    res = analyze(run, cfg)
    (run / "analysis.json").write_text(json.dumps(res, indent=1))
    lines = []
    for variant, v in res["variants"].items():
        lines.append(f"## {variant} ({v['items']} items)\n")
        lines.append("| arm | correct | mean positions | mean visible-test calls | mean wall s |")
        lines.append("|---|---|---|---|---|")
        for arm in cfg["arms"]:
            lines.append(f"| {arm} | {v['accuracy'][arm]} | {v['positions_mean'][arm]} | {v['exec_calls_mean'][arm]} | "
                         f"{v['wall_s_mean'][arm]} |")
        lines.append("")
        lines.append(f"Paired, {cfg['decision']['proposed']} vs each arm: " + "; ".join(
            f"{b}: +{len(p['wins'])}/-{len(p['regressions'])} (p={p['exact_two_sided_p']})" for b, p in v["paired"].items()))
        lines.append(f"\nPositive under the frozen rule: {v['positive']}\n")
    if "rewording_control" in res:
        lines.append(f"Rewording control survives: {res['rewording_control']['survives']}")
    (run / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
