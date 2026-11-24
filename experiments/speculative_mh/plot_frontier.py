"""Correct answers delivered versus actual per-task cost (two panels: wall-clock with public checks, and
model input positions). Reads paired.csv and direct.csv from the run directory; writes frontier.png
and frontier_table.csv (the table view of the same curves)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
SERIES = [  # fixed categorical order (validated: scripts/validate_palette.js, light mode)
    ("PowerMH, serial eager", "#2a78d6", "-"),
    ("PowerMH, tree + coin cap", "#eb6834", "-"),
    ("Retries: greedy, then T=1", "#1baf7a", "--"),
    ("Retries: q_alpha (T=1/4)", "#eda100", "--"),
]


def curves(rd):
    rows = list(csv.DictReader(open(rd / "paired.csv")))
    drows = list(csv.DictReader(open(rd / "direct.csv")))
    out = {}
    for name, (mode, shape) in zip([s[0] for s in SERIES[:2]], [("serial_eager", "compact"), ("tree_coin", "compact")]):
        rr = [r for r in rows if r["mode"] == mode and r["shape"] == shape]
        out[name] = dict(n=len(rr),
                         wall=[float(r["fh_wall_with_checks_s"]) for r in rr if r["correct"] == "True"],
                         pos=[int(r["prompt_len"]) + int(r["fh_live"]) + int(r["fh_dummy"]) for r in rr if r["correct"] == "True"])
    for name, pol in zip([s[0] for s in SERIES[2:]], ["greedy_then_t1", "q_alpha"]):
        rr = [r for r in drows if r["policy"] == pol]
        out[name] = dict(n=len(rr), wall=[float(r["cost_wall_s"]) for r in rr if r["correct"] == "True"],
                         pos=[int(r["cost_positions"]) for r in rr if r["correct"] == "True"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/fresh40.json"))
    ap.add_argument("--run-dir")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = Path(a.run_dir) if a.run_dir else (HERE / cfg["run_dir"]).resolve()
    cv = curves(rd)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    grids = {"wall": np.geomspace(0.5, 200, 400), "pos": np.geomspace(100, 30000, 400)}
    table = []
    for ax, key, xlabel in ((axes[0], "wall", "per-task budget: wall-clock seconds, including public checks"),
                            (axes[1], "pos", "per-task budget: model input positions (prefill + decode)")):
        for name, color, ls in SERIES:
            costs = np.sort(cv[name][key])
            g = grids[key]
            y = np.searchsorted(costs, g, side="right")
            ax.step(g, y, where="post", color=color, lw=2, ls=ls, label=name)
            ax.annotate(f"{len(costs)}", (g[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")
            for b in (2, 4, 8, 16, 32, 64) if key == "wall" else (500, 1000, 2000, 4000, 8000, 16000):
                table.append(dict(axis=key, budget=b, policy=name, correct=int(np.searchsorted(costs, b, side="right")),
                                  tasks=cv[name]["n"]))
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("tasks with a correct answer delivered")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    n = cv[SERIES[0][0]]["n"]
    fig.suptitle(f"Correct first verified answers vs actual cost, MBPP+ fresh panel ({n} tasks), Qwen2.5-1.5B-Instruct, M4 MPS",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(rd / "frontier.png", dpi=160)
    with open(rd / "frontier_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print("wrote", rd / "frontier.png")


if __name__ == "__main__":
    main()
