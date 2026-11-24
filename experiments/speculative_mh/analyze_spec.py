"""Analysis of the native speculative-PowerMH run.

Writes paired.csv (one row per task x arm: ledger, first-hit and terminal costs), direct.csv,
summary.json and summary.md into the run directory. Costs are kept apart:
  ticks   lockstep advances until a decision is known (logical critical path)
  calls   model forward calls (physical critical path)
  live    model input positions issued for real cursors (paid work); dummy = filler rows (fixed shape)
  wall    measured seconds (prefill, forwards, cache packing, float64 scoring and sampling, host logic)
  checks  public-test executions, charged at the measured single-check latency
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "first_answer"))
from common import chash, load_panel  # noqa: E402

ARMS = [("serial_eager", "compact"), ("serial_lazy", "compact")] + [
    (m, s) for s in ("compact", "fixed") for m in ("tree_eager", "tree_lazy", "tree_safe", "tree_coin")]


def first_hit(rec, lab, codes_of):
    """tau = first committed state whose program passes the public assert; checks counted once per new state."""
    seen, checks = set(), 0
    for idx, s in enumerate(rec["steps"]):
        h = s.get("state_hash")
        if h is None:
            continue
        if h not in seen:
            seen.add(h)
            checks += 1
            g = lab[chash(codes_of[h])]
            if g["visible"]:
                return dict(tau=s["k"], step_index=idx, delivered=True, correct=bool(g["all"]), checks=checks,
                            answer=chash(codes_of[h]))
    return dict(tau=None, step_index=len(rec["steps"]) - 1, delivered=False, correct=False, checks=checks, answer=None)


def waste(rec, upto_calls=None):
    """Classify live rows by the cursor labels they served (trace order = call order)."""
    dead = {}
    for s in rec["steps"]:
        e = s.get("eliminated_child")
        if e:
            dead[e["label"]] = "reject_child_of_accepted_parent" if e["label"].endswith("_reject") else \
                "accept_child_of_rejected_parent"
    out = Counter()
    for i, t in enumerate(rec["batch_trace"]):
        if upto_calls is not None and i >= upto_calls:
            break
        for g in t["real_row_consumers"]:
            kinds = {dead.get(lbl) for lbl in g}
            out[kinds.pop() if len(kinds) == 1 and None not in kinds else "used"] += 1
    return out


def coin_saving_identity(safe, coin):
    """Rows the safe arm issued for children that the coin arm cancelled, after the cancel tick."""
    total = 0
    for s in coin["steps"]:
        if "coin_cancel_tick" not in s:
            continue
        label = f"pair{s['k']}_{'reject' if s['pair_type'] == 'shared' else 'accept'}"
        for t in safe["batch_trace"]:
            if t["tick"] > s["coin_cancel_tick"]:
                total += sum(label in g for g in t["real_row_consumers"])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/fresh40.json"))
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = Path(a.run_dir) if a.run_dir else (HERE / cfg["run_dir"]).resolve()
    _, tasks = load_panel(HERE / cfg["panel"])
    grades = json.loads((rd / "grades.json").read_text())
    lat = json.loads((rd / "check_latency.json").read_text())["median"]
    parity = [json.loads(l) for l in open(rd / "parity.jsonl")]
    rows, drows, recs = [], [], {}
    widths = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        lab = grades.get(t.task_id)
        if lab is None:
            continue
        ref_hit = None
        for mode, shape in ARMS:
            p = rd / f"task{t.task_id}_{mode}_{shape}.json"
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            recs[t.task_id, mode, shape] = rec
            codes_of = {h: v["code"] for h, v in rec["states"].items()}
            fh = first_hit(rec, lab, codes_of)
            if ref_hit is None:
                ref_hit = fh
            s = rec["steps"][fh["step_index"]]
            w_all, w_fh = waste(rec), waste(rec, s["commit_calls"])
            for tr in rec["batch_trace"]:
                widths[shape][tr["width"]].append(tr["forward_s"])
            rows.append(dict(
                task_id=t.task_id, mode=mode, shape=shape, same_ledger_as_serial_eager=fh == ref_hit,
                tau=fh["tau"], delivered=fh["delivered"], correct=fh["correct"], checks_to_answer=fh["checks"],
                prompt_len=rec["prompt_len"],
                fh_ticks=s["commit_tick"], fh_calls=s["commit_calls"], fh_live=s["commit_live_inputs"],
                fh_dummy=s["commit_dummy_inputs"], fh_wall_s=s["commit_wall_s"],
                fh_speculative_waste=w_fh["reject_child_of_accepted_parent"] + w_fh["accept_child_of_rejected_parent"],
                fh_wall_with_checks_s=s["commit_wall_s"] + fh["checks"] * lat,
                ticks=rec["ticks"], calls=rec["decode_calls"], live=rec["live_unique_decode_inputs"],
                dummy=rec["dummy_decode_inputs"], cache_hits=rec["cached_prefix_hits"], wall_s=rec["total_wall_s"],
                decode_s=rec["decode_s"], prefill_s=rec["prefill_s"], packing_s=rec["cache_packing_s"],
                scoring_sampling_s=rec["cdf_and_bookkeeping_s"],
                host_other_s=rec["total_wall_s"] - rec["decode_s"] - rec["prefill_s"] - rec["cache_packing_s"]
                - rec["cdf_and_bookkeeping_s"],
                waste_reject_child_of_accepted_parent=w_all["reject_child_of_accepted_parent"],
                waste_accept_child_of_rejected_parent=w_all["accept_child_of_rejected_parent"],
                used_rows=w_all["used"],
                accepted_steps=sum(1 for x in rec["steps"][1:] if x["accepted"] and not x.get("pad")),
                rejected_steps=sum(1 for x in rec["steps"][1:] if x["accepted"] is False),
                pad_steps=sum(1 for x in rec["steps"] if x.get("pad")),
                **{f"ctl_{k}": v for k, v in rec["control"].items()}))
        for pol in cfg["direct_policies"]:
            p = rd / f"direct_task{t.task_id}_{pol['name']}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            wall, pos = d["prefill_s"], d["prompt_len"]
            hit = None
            for at in d["attempts"]:
                wall += at["wall_s"] + lat
                pos += at["decode_inputs"]
                g = lab[chash(at["code"])]
                if g["visible"]:
                    hit = at
                    break
            drows.append(dict(task_id=t.task_id, policy=pol["name"], delivered=hit is not None,
                              correct=bool(hit is not None and lab[chash(hit["code"])]["all"]),
                              attempts_used=(hit["attempt"] + 1) if hit else len(d["attempts"]),
                              cost_wall_s=wall, cost_positions=pos,
                              any_correct_in_16=any(lab[chash(x["code"])]["all"] for x in d["attempts"])))
    for name, rr in (("paired.csv", rows), ("direct.csv", drows)):
        with open(rd / name, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rr[0]))
            wcsv.writeheader()
            wcsv.writerows(rr)

    by = defaultdict(dict)
    for r in rows:
        by[r["mode"], r["shape"]][r["task_id"]] = r
    common_tasks = sorted(set.intersection(*[set(v) for v in by.values()]), key=int)

    def tot(arm, key, tids=common_tasks):
        return sum(by[arm][t][key] for t in tids)

    def paired(a1, a2, key):
        d = [by[a2][t][key] - by[a1][t][key] for t in common_tasks]
        return dict(a=tot(a1, key), b=tot(a2, key), delta=sum(d), rel=(sum(d) / tot(a1, key)) if tot(a1, key) else None,
                    tasks_lower=sum(x < 0 for x in d), tasks_higher=sum(x > 0 for x in d), tasks_equal=sum(x == 0 for x in d))

    keys = ["ticks", "calls", "live", "dummy", "wall_s", "fh_ticks", "fh_calls", "fh_live", "fh_wall_s"]
    comparisons = {
        "certified thinning, serial (serial_eager -> serial_lazy)": (("serial_eager", "compact"), ("serial_lazy", "compact")),
        "speculation itself (serial_lazy -> tree_safe compact)": (("serial_lazy", "compact"), ("tree_safe", "compact")),
        "shortened on-path rejections, constant width (tree_eager -> tree_safe, fixed)": (("tree_eager", "fixed"), ("tree_safe", "fixed")),
        "same, compact (adds contention changes)": (("tree_eager", "compact"), ("tree_safe", "compact")),
        "MECHANISM, constant width (tree_safe -> tree_coin, fixed)": (("tree_safe", "fixed"), ("tree_coin", "fixed")),
        "MECHANISM, compact (tree_safe -> tree_coin, compact)": (("tree_safe", "compact"), ("tree_coin", "compact")),
        "risky pause heuristic (tree_safe -> tree_lazy, compact)": (("tree_safe", "compact"), ("tree_lazy", "compact")),
        "completed-score baseline vs mechanism (tree_eager -> tree_coin, compact)": (("tree_eager", "compact"), ("tree_coin", "compact")),
    }
    comp = {name: {k: paired(x, y, k) for k in keys} for name, (x, y) in comparisons.items()}
    c1 = all([s["commit_tick"] for s in recs[t, "tree_safe", sh]["steps"]] ==
             [s["commit_tick"] for s in recs[t, "tree_coin", sh]["steps"]] for t in common_tasks for sh in ("compact", "fixed"))
    ident = coin_actual = 0
    for t in common_tasks:
        ident += coin_saving_identity(recs[t, "tree_safe", "compact"], recs[t, "tree_coin", "compact"])
        coin_actual += by["tree_safe", "compact"][t]["live"] - by["tree_coin", "compact"][t]["live"]
    coin_counts = {k: tot(("tree_coin", "compact"), k) for k in
                   ("ctl_coin_cancelled_accept_children", "ctl_coin_cancelled_shared_children", "ctl_accept_children_launched",
                    "ctl_shared_pairs", "ctl_distinct_pairs", "ctl_coin_checks_on_realized_branch", "ctl_coin_cancel_at_launch")}
    waste_tbl = {f"{m}/{s}": dict(reject_child_of_accepted_parent=tot((m, s), "waste_reject_child_of_accepted_parent"),
                                  accept_child_of_rejected_parent=tot((m, s), "waste_accept_child_of_rejected_parent"),
                                  used=tot((m, s), "used_rows"), live=tot((m, s), "live"))
                 for m, s in ARMS if m.startswith("tree")}
    contention = {sh: {w: dict(n=len(v), median_ms=1e3 * statistics.median(v)) for w, v in sorted(ws.items())}
                  for sh, ws in widths.items()}
    ledger = dict(
        parity_identical=sum(p["identical"] for p in parity), parity_total=len(parity),
        first_hit_same_as_serial=sum(r["same_ledger_as_serial_eager"] for r in rows), rows=len(rows),
        delivered=sum(by["serial_eager", "compact"][t]["delivered"] for t in common_tasks),
        correct=sum(by["serial_eager", "compact"][t]["correct"] for t in common_tasks),
        accepted=tot(("serial_eager", "compact"), "accepted_steps"), rejected=tot(("serial_eager", "compact"), "rejected_steps"),
        pad=tot(("serial_eager", "compact"), "pad_steps"), tasks=len(common_tasks))
    dsum = {}
    for pol in cfg["direct_policies"]:
        rr = [r for r in drows if r["policy"] == pol["name"] and r["task_id"] in common_tasks]
        dsum[pol["name"]] = dict(tasks=len(rr), delivered=sum(r["delivered"] for r in rr), correct=sum(r["correct"] for r in rr),
                                 any_correct_in_16=sum(r["any_correct_in_16"] for r in rr),
                                 median_cost_wall_s=statistics.median(r["cost_wall_s"] for r in rr) if rr else None,
                                 total_positions=sum(r["cost_positions"] for r in rr))
    summary = dict(config=cfg["name"], check_latency_s=lat, ledger=ledger, comparisons=comp,
                   corollary_C1_identical_commit_ticks_safe_vs_coin=c1,
                   coin=dict(counts=coin_counts, live_saved_compact=coin_actual, safe_rows_after_cancel_tick=ident),
                   speculative_waste=waste_tbl, contention_forward_ms_by_width=contention, direct=dsum,
                   tasks=common_tasks)
    (rd / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(ledger=ledger, C1=c1, coin=summary["coin"], direct=dsum), indent=1))
    for name, v in comp.items():
        print(name, {k: (v[k]["a"], v[k]["b"], None if v[k]["rel"] is None else round(100 * v[k]["rel"], 2)) for k in
                     ("ticks", "calls", "live", "wall_s", "fh_ticks", "fh_live", "fh_wall_s")})
    print("waste", json.dumps(waste_tbl))
    print("contention", json.dumps(contention))


if __name__ == "__main__":
    main()
