"""Stage 0: same-prefix branch replay on development items (diagnostic).

For each development item: greedy decoding with top-k log-probs at every
position; a teacher-forced second stream on the masked-output prompt (the
visible test with its expected value replaced by ...); every eligible fork
(same rule as the policies, up to max_forks_replay per item, smallest margin
first); and the greedy completion of every plausible alternative from the
shared prefix. All distinct programs are graded on the visible test and,
offline, on every hidden test.

Because each alternative shares greedy's prefix and is completed by the same
deterministic rule the policies use, the replay reproduces exactly what flec
and the likelihood control would submit, with their model-position cost.

Gate (frozen in the config, decided on development items only):
  G0 oracle headroom   items where greedy is hidden-wrong and some single-fork
                       completion is hidden-correct >= gate.oracle_items_min
  G1 task-grounded     simulated flec minus simulated likelihood control, in
     choice            hidden-correct items, >= gate.flec_minus_likelihood_min
  G2 cheap signal      at decisive forks (candidates with different hidden
     (descriptive)     labels) the contrast rule's net correct choices vs greedy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DockerGrader, Engine, HFBackend, Ledger, VisibleChecker, chash, clean, file_sha256,  # noqa: E402
                    grade_hidden, load_panel)
from policies import branch, greedy_trace, list_forks  # noqa: E402


def replay_item(eng, task, cfg, grader):
    led = Ledger()
    eng.bind(led, task)
    ids = eng.encode(task.messages())
    tr = greedy_trace(eng, ids)
    greedy_positions = led.positions
    forks = list_forks(eng, tr, cfg["fork_ratio"], cfg["k_alt"])[: cfg["max_forks_replay"]]
    # teacher-forced masked-output stream along greedy, for the contrast signal
    masked = task.user_content(variant="masked_output")
    lpm_at = {}
    if masked != task.user_content():
        need = {f["pos"]: [f["top"]] + [a for a, _ in f["alts"]] for f in forks}
        ids_m = eng.encode([{"role": "user", "content": masked}])
        lpm, cache_m = eng.prefill(ids_m, aux=True)
        for pos, v in enumerate(tr["gen"]):
            if pos in need:
                lpm_at[pos] = {t: float(lpm[t]) for t in need[pos]}
            if pos == max(need, default=-1):
                break
            lpm, cache_m = eng.step(cache_m, v, aux=True)
    aux_positions = led.aux_prefill + led.aux_decode
    codes = {chash(clean(eng.text(tr["gen"]))): clean(eng.text(tr["gen"]))}
    greedy_hash = next(iter(codes))
    out_forks = []
    for f in forks:
        cands = [{"tok": f["top"], "str": f["strs"][0], "lp": f["top_lp"], "code_hash": greedy_hash,
                  "branch_positions": 0, "mean_lp": float(np.mean(tr["seq_lp"])), "is_greedy": True}]
        for (v, lpv), s in zip(f["alts"], f["strs"][1:]):
            before = led.positions
            b = branch(eng, tr, f["pos"], v)
            code = clean(eng.text(b["gen"]))
            h = chash(code)
            codes.setdefault(h, code)
            lps = tr["seq_lp"][: f["pos"]] + [lpv] + b["seq_lp"]
            cands.append({"tok": v, "str": s, "lp": lpv, "code_hash": h, "branch_positions": led.positions - before,
                          "mean_lp": float(np.mean(lps)), "is_greedy": False, "reason": b["reason"]})
        for c in cands:
            m = lpm_at.get(f["pos"], {}).get(c["tok"])
            c["lp_masked"] = m
            c["contrast"] = None if m is None else c["lp"] + cfg["contrast_lambda"] * (c["lp"] - m)
        out_forks.append({"pos": f["pos"], "margin": f["margin"],
                          "line": eng.text(tr["gen"][:f["pos"]]).rsplit("\n", 1)[-1], "candidates": cands})
    vis = VisibleChecker(grader, task, led)
    visible = {h: vis(code) for h, code in codes.items()}
    labels, grade_s = grade_hidden(grader, task, codes)
    for f in out_forks:
        for c in f["candidates"]:
            c["visible"] = visible[c["code_hash"]]
            c["correct"] = labels[c["code_hash"]]["all"]
    return {"task": task.task_id, "greedy_code_hash": greedy_hash, "greedy_code": codes[greedy_hash],
            "greedy_visible": visible[greedy_hash], "greedy_correct": labels[greedy_hash]["all"],
            "greedy_len": len(tr["gen"]), "greedy_positions": greedy_positions, "aux_positions": aux_positions,
            "eligible_forks_total": len(list_forks(eng, tr, cfg["fork_ratio"], cfg["k_alt"])),
            "forks": out_forks, "codes": codes, "labels": labels, "ledger": led.as_dict(), "grade_s": grade_s}


def simulate(item, cfg):
    """Exact replay of flec and the likelihood control from the recorded branches."""
    g_ok, g_vis = item["greedy_correct"], item["greedy_visible"]
    base = item["greedy_positions"]
    if g_vis is not False:
        return {"flec": (g_ok, base, "greedy"), "likelihood": (g_ok, base, "greedy")}
    budget = base + int(round(cfg["extra_budget_factor"] * item["greedy_len"]))
    spent, flec = base, None
    forks = sorted(item["forks"], key=lambda f: (f["margin"], f["pos"]))[: cfg["max_forks"]]
    greedy_mean = forks[0]["candidates"][0]["mean_lp"] if forks else 0.0
    lik = [(greedy_mean, 0, g_ok)]
    order = 0
    for f in forks:
        for c in f["candidates"][1:]:
            if spent + c["branch_positions"] > budget:
                flec = flec or (g_ok, spent, "fallback_budget")
                best = max(lik, key=lambda x: (x[0], -x[1]))
                return {"flec": flec, "likelihood": (best[2], spent, "budget")}
            spent += c["branch_positions"]
            order += 1
            lik.append((c["mean_lp"], order, c["correct"]))
            if flec is None and c["visible"]:
                flec = (c["correct"], spent, f"fork@{f['pos']}")
    best = max(lik, key=lambda x: (x[0], -x[1]))
    return {"flec": flec or (g_ok, spent, "fallback"), "likelihood": (best[2], spent, "max_mean_lp")}


def summarize(items, cfg):
    gate = cfg["gate"]
    oracle = [it["task"] for it in items if not it["greedy_correct"]
              and any(c["correct"] for f in it["forks"] for c in f["candidates"][1:])]
    sims = {it["task"]: simulate(it, cfg) for it in items}
    flec_ok = sum(bool(s["flec"][0]) for s in sims.values())
    lik_ok = sum(bool(s["likelihood"][0]) for s in sims.values())
    greedy_ok = sum(bool(it["greedy_correct"]) for it in items)
    decisive, contrast_net, margin_dec, margin_non = [], 0, [], []
    which = {"greedy_token": 0, "mean_lp": 0, "contrast": 0, "visible_exec": 0}
    for it in items:
        for f in it["forks"]:
            labs = [c["correct"] for c in f["candidates"]]
            if len(set(labs)) < 2:
                margin_non.append(f["margin"])
                continue
            margin_dec.append(f["margin"])
            decisive.append((it["task"], f["pos"]))
            cs = f["candidates"]
            which["greedy_token"] += cs[0]["correct"]
            which["mean_lp"] += max(cs, key=lambda c: c["mean_lp"])["correct"]
            if all(c["contrast"] is not None for c in cs):
                pick = max(cs, key=lambda c: (c["contrast"], c["is_greedy"]))
                which["contrast"] += pick["correct"]
                contrast_net += int(pick["correct"]) - int(cs[0]["correct"])
            vis_pass = [c for c in cs if c["visible"]]
            which["visible_exec"] += (vis_pass[0]["correct"] if vis_pass else cs[0]["correct"])
    flec_pos = sum(s["flec"][1] for s in sims.values())
    lik_pos = sum(s["likelihood"][1] for s in sims.values())
    g_pos = sum(it["greedy_positions"] for it in items)
    res = {
        "items": len(items), "greedy_correct": greedy_ok,
        "oracle_single_fork_items": oracle,
        "simulated": {"flec_correct": flec_ok, "likelihood_correct": lik_ok,
                      "positions": {"greedy": g_pos, "flec": flec_pos, "likelihood": lik_pos},
                      "per_item": {t: {k: list(v) for k, v in s.items()} for t, s in sims.items()}},
        "decisive_forks": len(decisive), "which_correct_at_decisive_forks": which,
        "contrast_net_vs_greedy_at_decisive_forks": contrast_net,
        "margin_quartiles": {"decisive": np.quantile(margin_dec, [0, .25, .5, .75, 1]).round(3).tolist() if margin_dec else None,
                             "non_decisive": np.quantile(margin_non, [0, .25, .5, .75, 1]).round(3).tolist() if margin_non else None},
        "eligible_forks_per_item_median": float(np.median([it["eligible_forks_total"] for it in items])) if items else None,
    }
    res["gate"] = {
        "G0_oracle_headroom": {"value": len(oracle), "min": gate["oracle_items_min"], "pass": len(oracle) >= gate["oracle_items_min"]},
        "G1_task_grounded_choice": {"value": flec_ok - lik_ok, "min": gate["flec_minus_likelihood_min"],
                                    "pass": flec_ok - lik_ok >= gate["flec_minus_likelihood_min"]},
        "G2_cheap_signal_descriptive": {"contrast_net_vs_greedy": contrast_net},
    }
    res["decision"] = ("run Stage 1 + Stage 2 (needs approval)" if res["gate"]["G0_oracle_headroom"]["pass"]
                       and res["gate"]["G1_task_grounded_choice"]["pass"] else
                       "stop flec: oracle headroom small; switch mechanism or workload" if not res["gate"]["G0_oracle_headroom"]["pass"]
                       else "stop flec: task-grounded choice does not beat confidence on development forks")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="hf", choices=["hf", "fake"])
    ap.add_argument("--grader", default="docker", choices=["docker", "fake"])
    ap.add_argument("--run-dir", default=None, help="override (dry runs must not write into frozen run dirs)")
    ap.add_argument("--approve", action="store_true", help="required for a real-model run")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.backend == "hf" and not a.approve:
        raise SystemExit("real-model run refused without --approve")
    run = Path(a.run_dir or cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    _, tasks = load_panel(Path(cfg["panel"]))
    tasks = tasks[: a.limit] if a.limit else tasks
    if a.backend == "fake":
        from fake import FakeBackend
        backend = FakeBackend.from_tasks(tasks)
    else:
        backend = HFBackend(cfg["model_dir"])
    if a.grader == "fake":
        from fake import FakeGrader
        grader = FakeGrader()
    else:
        grader = DockerGrader()
    eng = Engine(backend, cap=cfg["cap"])
    (run / "provenance.json").write_text(json.dumps({
        "config_sha256": file_sha256(a.config), "panel_sha256": file_sha256(cfg["panel"]),
        "code_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "backend": a.backend, "limit": a.limit, "started": time.strftime("%F %T")}, indent=1))
    items = []
    with open(run / "forks.jsonl", "w") as fh:
        for t in tasks:
            t0 = time.perf_counter()
            it = replay_item(eng, t, cfg, grader)
            it["sec"] = time.perf_counter() - t0
            fh.write(json.dumps(it) + "\n")
            fh.flush()
            items.append(it)
            print(json.dumps({"task": t.task_id, "greedy_ok": it["greedy_correct"], "greedy_vis": it["greedy_visible"],
                              "forks": len(it["forks"]), "positions": it["ledger"]["positions"],
                              "fixes": sum(c["correct"] for f in it["forks"] for c in f["candidates"][1:]),
                              "sec": round(it["sec"], 1)}), flush=True)
    s = summarize(items, cfg)
    (run / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "simulated"} | {"simulated": {
        k: v for k, v in s["simulated"].items() if k != "per_item"}}, indent=1))


if __name__ == "__main__":
    main()
