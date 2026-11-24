"""Offline grading for the speculative-PowerMH run (Docker grader, never seen by any policy).

For every task, every distinct program in any state of any arm and in any direct attempt is graded on
the public assert (the policy's verifier) and on all asserts (correctness label). A separate
sequential sample measures the latency of one public check, which analyze_spec.py charges to every
public check a policy makes.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "first_answer"))
from common import DockerGrader, chash, grade_hidden, load_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/fresh40.json"))
    ap.add_argument("--run-dir")
    ap.add_argument("--latency-sample", type=int, default=24)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rd = Path(a.run_dir) if a.run_dir else (HERE / cfg["run_dir"]).resolve()
    _, tasks = load_panel(HERE / cfg["panel"])
    grader = DockerGrader()
    out, sample = {}, []
    for t in tasks:
        codes = {}
        for p in rd.glob(f"task{t.task_id}_*.json"):
            for s in json.loads(p.read_text())["states"].values():
                codes[chash(s["code"])] = s["code"]
        for p in rd.glob(f"direct_task{t.task_id}_*.json"):
            for at in json.loads(p.read_text())["attempts"]:
                codes[chash(at["code"])] = at["code"]
        if not codes:
            continue
        lab, sec = grade_hidden(grader, t, codes)
        out[t.task_id] = lab
        sample += [(t, c) for c in list(codes.values())[:1]]
        print("graded", t.task_id, len(codes), f"{sec:.1f}s", flush=True)
    (rd / "grades.json").write_text(json.dumps(out))
    lat = []
    for t, code in sample[:a.latency_sample]:
        t0 = time.perf_counter()
        grader.run([(f"{chash(code)}/lat", code, t.setup, t.visible[0])])
        lat.append(time.perf_counter() - t0)
    (rd / "check_latency.json").write_text(json.dumps(dict(seconds=lat, median=statistics.median(lat),
                                                           what="one public assert, one Docker grader call, sequential")))
    print("public-check latency median", round(statistics.median(lat), 3), "s over", len(lat))


if __name__ == "__main__":
    main()
