"""Freeze the task panels. No model calls.

dev16       MBPP development rows 0-15 of the harness (task_ids 11-26). Rows 16-63
            (the reserved 48) are never written into any panel.
pilot40     MBPP+ (EvalPlus, local copy) tasks never used by any earlier
            experiment in this repository (scan below), MBPP test or validation
            split ids 11-600, drawn in a seeded order; a candidate is kept only if
            its reference program passes its own visible and hidden asserts in the
            Docker grader. Visible = first assert; hidden = the other original
            asserts (the plus suite needs the Linux numpy image; secondary only).
humaneval40 HumanEval tasks whose docstring has at least one doctest example that
            can be turned into an assert with a maskable expected value; seeded
            order; the canonical solution must pass visible and hidden tests.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HARNESS, REPO, DockerGrader, chash, grade_hidden, humaneval_task, mask_expected_output, mbpp_task  # noqa: E402

FP = HARNESS / "feedback_policy_v2"


def used_mbpp_ids():
    used = {}
    src = json.loads((HARNESS / "mbpp_source.json").read_text())["rows"]
    used["dev16"] = [r["row"]["task_id"] for r in src[:16]]
    used["reserved48"] = [r["row"]["task_id"] for r in src[16:64]]
    used["independent_screen64"] = [r["row"]["task_id"] for r in
                                    json.loads((HARNESS / "independent_mh_screen/mbpp_source.json").read_text())["rows"]]
    pp = json.loads((FP / "power_protocol.json").read_text())
    used["power_protocol80"] = json.loads(pp["task_ids"]) if isinstance(pp["task_ids"], str) else pp["task_ids"]
    scan = set()
    paths = glob.glob(str(FP / "**/*.json*"), recursive=True) + glob.glob(str(REPO / "results/local_runs/**/*mbpp*.json*"), recursive=True)
    for p in paths:
        if any(x in p for x in ("linux_numpy", "mbppplus_rows", "humaneval_official", "/panels/")):
            continue
        try:
            txt = Path(p).read_text()
        except Exception:
            continue
        scan |= {int(m) for m in re.findall(r'"task(?:_id)?"\s*:\s*(\d+)', txt)}
    used["regex_scan_of_earlier_mbpp_artifacts"] = sorted(scan)
    return used


def validate(grader, task):
    lab, _ = grade_hidden(grader, task, {chash(task.reference): task.reference})
    v = next(iter(lab.values()))
    return v["all"], v["tests"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "panels"))
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--pilot-n", type=int, default=40)
    ap.add_argument("--transfer-n", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grader = DockerGrader()
    used = used_mbpp_ids()
    banned = set().union(*map(set, used.values()))

    src = json.loads((HARNESS / "mbpp_source.json").read_text())["rows"]
    dev = [mbpp_task(r["row"]) for r in src[:16]]
    dev_checks = {t.task_id: validate(grader, t) for t in dev}
    json.dump({"name": "dev16", "source": str(HARNESS / "mbpp_source.json") + " rows 0-15",
               "reference_validation": dev_checks, "tasks": [asdict(t) for t in dev]},
              open(out / "dev16.json", "w"), indent=1)

    rows = json.loads((FP / "population_replication56/mbppplus_rows.json").read_text())
    rows = rows["rows"] if isinstance(rows, dict) else rows
    rows = [r.get("row", r) for r in rows]
    pool = sorted((r for r in rows if 11 <= r["task_id"] <= 600 and r["task_id"] not in banned), key=lambda r: r["task_id"])
    order = np.random.default_rng(a.seed).permutation(len(pool))
    pilot, skipped = [], []
    for i in order:
        r = dict(pool[i])
        r["text"] = r["prompt"]
        r["test_setup_code"] = "\n".join(r.get("test_imports") or [])
        t = mbpp_task(r, family="mbppplus")
        ok, tests = validate(grader, t)
        if ok:
            pilot.append(t)
        else:
            skipped.append({"task_id": t.task_id, "tests": tests})
        if len(pilot) == a.pilot_n:
            break
    json.dump({"name": "pilot40", "source": "evalplus/mbppplus local copy " + str(FP / "population_replication56/mbppplus_rows.json"),
               "seed": a.seed, "pool_size": len(pool), "excluded_ids": {k: sorted(v) for k, v in used.items()},
               "skipped_reference_failures": skipped, "tasks": [asdict(t) for t in pilot]},
              open(out / "pilot40.json", "w"), indent=1)

    he = json.loads((FP / "humaneval_source_gate/humaneval_official_rows.json").read_text())
    he = he["rows"] if isinstance(he, dict) else he
    he = [r.get("row", r) for r in he]
    cand = [humaneval_task(r) for r in he]
    cand = [t for t in cand if t.visible and any(mask_expected_output(v) for v in t.visible)]
    order = np.random.default_rng(a.seed + 1).permutation(len(cand))
    transfer, skipped = [], []
    for i in order:
        t = cand[i]
        ok, tests = validate(grader, t)
        (transfer if ok else skipped).append(t if ok else {"task_id": t.task_id, "tests": tests})
        if len(transfer) == a.transfer_n:
            break
    json.dump({"name": "humaneval40", "source": str(FP / "humaneval_source_gate/humaneval_official_rows.json"),
               "seed": a.seed + 1, "pool_size": len(cand), "skipped_reference_failures": skipped,
               "tasks": [asdict(t) for t in transfer]}, open(out / "humaneval40.json", "w"), indent=1)
    print(json.dumps({"dev16_reference_ok": sum(v[0] for v in dev_checks.values()), "pilot40": len(pilot),
                      "pilot_pool": len(pool), "pilot_skipped": len(skipped), "humaneval40": len(transfer),
                      "banned_ids": len(banned)}, indent=1))


if __name__ == "__main__":
    main()
