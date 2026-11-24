"""Analysis of the native MATH run (run_spec_math.py, run_direct_math.py).

Writes paired_math.csv (one row per problem x arm), summary_math.json, frontier_math.png and
frontier_math_table.csv into the run directory. Cost columns are kept apart: ticks (logical critical
path), calls (model forwards), live/dummy rows (paid work / filler), wall seconds by component.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_spec import coin_saving_identity, waste  # noqa: E402


def pair_stats(rec):
    """Per pair (two MH steps resolved together): realized outcomes and the commit tick of each step."""
    out = []
    steps = rec["steps"]
    for a, b in zip(steps, steps[1:]):
        if a.get("pair_type") and "parent_accepted" in b:
            out.append(dict(k=a["k"], pair_type=a["pair_type"], first=a["accepted"], second=b["accepted"],
                            first_tick=a["commit_tick"], second_tick=b["commit_tick"],
                            coin_cancel=("coin_cancel_tick" in a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/math10.json"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = (HERE / cfg["run_dir"]).resolve()
    arms = [tuple(x) for x in cfg["arms"]]
    lo, hi = map(int, cfg["rows"].split(":"))
    recs, rows = {}, []
    widths = defaultdict(lambda: defaultdict(list))
    for i in range(lo, hi):
        for mode, shape in arms:
            p = rd / f"row{i:04d}_{mode}_{shape}.json"
            if p.exists():
                recs[i, mode, shape] = json.loads(p.read_text())
    done = sorted({i for i in range(lo, hi) if all((i, m, s) in recs for m, s in arms)})
    for i in done:
        for mode, shape in arms:
            r = recs[i, mode, shape]
            w = waste(r)
            for tr in r["batch_trace"]:
                widths[shape][tr["width"]].append(tr["forward_s"])
            st = r["steps"][1:]
            rows.append(dict(row=i, mode=mode, shape=shape, correct=r["correct"], answer=r["answer"],
                             ticks=r["ticks"], calls=r["decode_calls"], live=r["live_unique_decode_inputs"],
                             dummy=r["dummy_decode_inputs"], cache_hits=r["cached_prefix_hits"], wall_s=r["total_wall_s"],
                             prefill_s=r["prefill_s"], decode_s=r["decode_s"], packing_s=r["cache_packing_s"],
                             scoring_sampling_s=r["cdf_and_bookkeeping_s"],
                             host_other_s=r["total_wall_s"] - r["prefill_s"] - r["decode_s"] - r["cache_packing_s"]
                             - r["cdf_and_bookkeeping_s"],
                             accepted=sum(1 for s in st if s["accepted"] and not s.get("pad")),
                             rejected=sum(1 for s in st if s["accepted"] is False), pad=sum(1 for s in st if s.get("pad")),
                             waste_reject_child_of_accepted_parent=w["reject_child_of_accepted_parent"],
                             waste_accept_child_of_rejected_parent=w["accept_child_of_rejected_parent"],
                             used_rows=w["used"], **{f"ctl_{k}": v for k, v in r["control"].items()}))
    with open(rd / "paired_math.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)
    by = defaultdict(dict)
    for r in rows:
        by[r["mode"], r["shape"]][r["row"]] = r

    def tot(arm, key):
        return sum(by[arm][i][key] for i in done)

    def paired(a1, a2, key):
        d = [by[a2][i][key] - by[a1][i][key] for i in done]
        base = tot(a1, key)
        return dict(a=round(base, 3), b=round(tot(a2, key), 3), rel_pct=round(100 * sum(d) / base, 2) if base else None,
                    lower=sum(x < 0 for x in d), higher=sum(x > 0 for x in d), equal=sum(x == 0 for x in d))

    keys = ("ticks", "calls", "live", "dummy", "wall_s", "decode_s")
    C = {"speculation itself (serial_lazy -> tree_safe compact)": (("serial_lazy", "compact"), ("tree_safe", "compact")),
         "certificate-time decisions vs completed-score cancellation (tree_eager -> tree_safe, compact)":
             (("tree_eager", "compact"), ("tree_safe", "compact")),
         "MECHANISM at constant width (tree_safe -> tree_coin, fixed)": (("tree_safe", "fixed"), ("tree_coin", "fixed")),
         "MECHANISM compact (tree_safe -> tree_coin, compact)": (("tree_safe", "compact"), ("tree_coin", "compact")),
         "contention: compact vs fixed width (tree_safe)": (("tree_safe", "fixed"), ("tree_safe", "compact")),
         "baseline -> mechanism (tree_eager -> tree_coin, compact)": (("tree_eager", "compact"), ("tree_coin", "compact"))}
    comp = {n: {k: paired(x, y, k) for k in keys} for n, (x, y) in C.items()}
    parity = [json.loads(l) for l in open(rd / "parity.jsonl")]
    c1 = all([s["commit_tick"] for s in recs[i, "tree_safe", sh]["steps"]] ==
             [s["commit_tick"] for s in recs[i, "tree_coin", sh]["steps"]] for i in done for sh in ("compact", "fixed"))
    ident = sum(coin_saving_identity(recs[i, "tree_safe", "compact"], recs[i, "tree_coin", "compact"]) for i in done)
    saved = tot(("tree_safe", "compact"), "live") - tot(("tree_coin", "compact"), "live")
    # Where do tick savings of certificate-time decisions come from? (Corollary C4)
    tick_src = Counter()
    for i in done:
        e, s = pair_stats(recs[i, "tree_eager", "compact"]), pair_stats(recs[i, "tree_safe", "compact"])
        for pe, ps in zip(e, s):
            key = f"first_{'acc' if pe['first'] else 'rej'}_second_{'acc' if pe['second'] else 'rej'}"
            tick_src[key + "_pairs"] += 1
            tick_src[key + "_first_commit_ticks_saved"] += pe["first_tick"] - ps["first_tick"]
    serial_steps = Counter()
    for i in done:
        for s in recs[i, "serial_lazy", "compact"]["steps"][1:]:
            serial_steps["pad" if s.get("pad") else ("accepted" if s["accepted"] else "rejected")] += 1
    waste_tbl = {f"{m}/{s}": dict(reject_child_of_accepted_parent=tot((m, s), "waste_reject_child_of_accepted_parent"),
                                  accept_child_of_rejected_parent=tot((m, s), "waste_accept_child_of_rejected_parent"),
                                  used=tot((m, s), "used_rows"), live=tot((m, s), "live"))
                 for m, s in arms if m.startswith("tree")}
    coin = {k: tot(("tree_coin", "compact"), k) for k in
            ("ctl_coin_cancelled_accept_children", "ctl_coin_cancelled_shared_children", "ctl_accept_children_launched",
             "ctl_shared_pairs", "ctl_distinct_pairs", "ctl_coin_checks_on_realized_branch", "ctl_coin_cancel_at_launch")}
    contention = {sh: {w: dict(n=len(v), median_ms=round(1e3 * statistics.median(v), 2)) for w, v in sorted(ws.items())}
                  for sh, ws in widths.items()}
    # Task-level comparison on the same rows: PowerMH (identical answers in every arm) vs q_alpha samples.
    task = {}
    direct = {i: json.loads((rd / f"direct_row{i:04d}_qalpha.json").read_text()) for i in done
              if (rd / f"direct_row{i:04d}_qalpha.json").exists()}
    if direct:
        def vote(samples):
            c = Counter(s["norm"] for s in samples if s["norm"] is not None)
            if not c:
                return None
            best = max(c.values())
            return next(s["norm"] for s in samples if s["norm"] is not None and c[s["norm"]] == best)
        from math_contract import norm
        curves = {}
        for name, arm in (("PowerMH K=16, serial early rejection", ("serial_lazy", "compact")),
                          ("PowerMH K=16, tree + coin cap", ("tree_coin", "compact"))):
            curves[name] = [(by[arm][i]["wall_s"], by[arm][i]["correct"]) for i in done]
        for k in (1, 2, 4, 8):
            pts = []
            for i, d in direct.items():
                ss = d["samples"][:k]
                wall = d["prefill_s"] + sum(s["wall_s"] for s in ss)
                pts.append((wall, vote(ss) == norm(d["gold"])))
            curves[f"q_alpha {'sample' if k == 1 else f'majority vote of {k}'}"] = pts
        task = {n: dict(correct=sum(c for _, c in v), tasks=len(v), total_wall_s=round(sum(w for w, _ in v), 1),
                        median_wall_s=round(statistics.median(w for w, _ in v), 1)) for n, v in curves.items()}
        plot(rd, curves, len(done))
    summary = dict(config=cfg["name"], problems=done, parity_identical=sum(p["identical"] for p in parity),
                   parity_total=len(parity), serial_step_outcomes=dict(serial_steps), comparisons=comp,
                   corollary_C1_identical_commit_ticks=c1,
                   coin=dict(counts=coin, live_rows_saved_compact=saved, safe_rows_after_cancel_tick=ident),
                   commit_tick_sources_eager_to_safe=dict(tick_src), speculative_waste=waste_tbl,
                   contention_forward_ms_by_width=contention, task_level=task,
                   correct_powermh=tot(("serial_lazy", "compact"), "correct"))
    (rd / "summary_math.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}, indent=1))
    for n, v in comp.items():
        print(n, {k: (v[k]["a"], v[k]["b"], v[k]["rel_pct"]) for k in ("ticks", "calls", "live", "wall_s")})


def plot(rd, curves, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    style = {"PowerMH K=16, serial early rejection": ("#2a78d6", "-"), "PowerMH K=16, tree + coin cap": ("#eb6834", "-"),
             "q_alpha sample": ("#1baf7a", "--"), "q_alpha majority vote of 8": ("#eda100", "--")}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "figure.facecolor": surface, "axes.facecolor": surface, "axes.labelcolor": ink})
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    g = np.geomspace(5, 2000, 500)
    rows = []
    for name, (color, ls) in style.items():
        costs = np.sort([w for w, c in curves[name] if c])
        y = np.searchsorted(costs, g, side="right")
        ax.step(g, y, where="post", color=color, lw=2, ls=ls, label=name)
        ax.annotate(str(len(costs)), (g[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=ink, va="center")
        for b in (15, 30, 60, 120, 240, 480, 960):
            rows.append(dict(budget_s=b, policy=name, correct=int(np.searchsorted(costs, b, side="right")), tasks=n))
    ax.set_xscale("log")
    ax.set_xlabel("per-problem budget: wall-clock seconds (fp32, M4 MPS, one model job)")
    ax.set_ylabel("problems answered correctly")
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"MATH-500 rows 0-{n - 1}, Qwen2.5-Math-1.5B: correct answers vs actual cost", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(rd / "frontier_math.png", dpi=160)
    with open(rd / "frontier_math_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
