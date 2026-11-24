"""Run first-answer arms on a frozen panel (Stage 1 tuning, Stage 2 pilot, Stage 3 transfer).

Per item and variant, every arm gets a fresh Ledger, the same Task (same task
text and visible test), and a VisibleChecker. Hidden labels are computed only
after every arm has fixed its answer for that item, and are written next to,
never into, the arms' inputs.

A real-model run needs --approve; --backend fake runs the full pipeline with
the scripted model (dry run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DockerGrader, Engine, HFBackend, Ledger, VisibleChecker, chash, file_sha256,  # noqa: E402
                    grade_hidden, load_panel)
from policies import ARMS  # noqa: E402


def resolve_sampler(cfg, run):
    s = cfg.get("sampler")
    if isinstance(s, dict) and "from" in s:
        p = Path(s["from"])
        if not p.exists():
            raise SystemExit(f"sampler file {p} missing: run Stage 1 first")
        spec = json.loads(p.read_text())["selected"]
        (run / "sampler_used.json").write_text(json.dumps({"from": str(p), "sha256": file_sha256(p), "spec": spec}))
        return spec
    return s


def run_item(eng, task, arms, acfg, grader, variant):
    rec = {"task": task.task_id, "variant": variant, "arms": {}}
    codes = {}
    for name in arms:
        led = Ledger()
        eng.bind(led, task)
        vis = VisibleChecker(grader, task, led)
        t0 = time.perf_counter()
        out = ARMS[name](eng, task, acfg | {"reworded": variant == "reworded"}, vis)
        out["wall_s"] = round(time.perf_counter() - t0, 3)
        out["ledger"] = led.as_dict()
        out["code_hash"] = chash(out["code"])
        codes.setdefault(out["code_hash"], out["code"])
        rec["arms"][name] = out
    labels, sec = grade_hidden(grader, task, codes)
    rec["labels"] = labels
    rec["grade_s"] = round(sec, 3)
    return rec


def tune(eng, tasks, cfg, grader, run):
    rows = []
    for spec in cfg["sampler_grid"]:
        for seed in cfg["tuning_seeds"]:
            ok = 0
            for t in tasks:
                led = Ledger()
                eng.bind(led, t)
                out = ARMS["sampler"](eng, t, cfg["arm"] | {"sampler": spec, "seed": seed}, None)
                lab, _ = grade_hidden(grader, t, {chash(out["code"]): out["code"]})
                ok += next(iter(lab.values()))["all"]
                rows.append({"spec": spec, "seed": seed, "task": t.task_id, "correct": next(iter(lab.values()))["all"],
                             "positions": led.positions})
            print(json.dumps({"spec": spec, "seed": seed, "correct": ok, "n": len(tasks)}), flush=True)
    score = {}
    for i, spec in enumerate(cfg["sampler_grid"]):
        k = json.dumps(spec, sort_keys=True)
        score[k] = (sum(r["correct"] for r in rows if json.dumps(r["spec"], sort_keys=True) == k), -spec["temperature"], -i)
    best = max(score, key=score.get)
    out = {"selected": json.loads(best), "rule": cfg["selection_rule"],
           "scores": {k: v[0] for k, v in score.items()}, "rows": rows}
    (run / "tuned_sampler.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"selected": out["selected"], "scores": out["scores"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default="hf", choices=["hf", "fake"])
    ap.add_argument("--grader", default="docker", choices=["docker", "fake"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-dir", default=None, help="override (dry runs must not write into frozen run dirs)")
    ap.add_argument("--approve", action="store_true", help="required for a real-model run")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.backend == "hf" and not a.approve:
        raise SystemExit("real-model run refused without --approve")
    run = Path(a.run_dir or cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    _, tasks = load_panel(cfg["panel"])
    tasks = tasks[: a.limit] if a.limit else tasks
    if a.backend == "fake":
        from fake import FakeBackend, FakeGrader
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
        "config": a.config, "config_sha256": file_sha256(a.config), "panel_sha256": file_sha256(cfg["panel"]),
        "code_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "backend": a.backend, "grader": a.grader, "limit": a.limit, "started": time.strftime("%F %T")}, indent=1))
    if cfg["stage"] == "tune":
        return tune(eng, tasks, cfg, grader, run)
    acfg = dict(cfg["arm"])
    # Real runs read the Stage 1 selection (hash recorded); dry runs use a placeholder.
    acfg["sampler"] = resolve_sampler(acfg, run) if a.backend == "hf" else acfg["dry_run_sampler"]
    with open(run / "arms.jsonl", "w") as fh:
        for t in tasks:
            for variant in cfg["variants"]:
                rec = run_item(eng, t, cfg["arms"], acfg, grader, variant)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(json.dumps({"task": t.task_id, "variant": variant, **{
                    n: (rec["labels"][o["code_hash"]]["all"], o["ledger"]["positions"]) for n, o in rec["arms"].items()}}),
                    flush=True)


if __name__ == "__main__":
    main()
