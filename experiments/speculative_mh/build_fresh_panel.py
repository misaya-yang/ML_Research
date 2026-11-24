"""Freeze a fresh 40-task MBPP+ panel for the speculative-PowerMH test.

Pool: MBPP+ task ids 11-600 minus every id frozen as used in first_answer/panels/pilot40.json
(dev16, reserved48 = task_ids 27-74, earlier screens, regex scan of earlier artifacts), minus the
pilot40 roster and minus the localized dev24 roster. Order: SHA-256 of a published salt and the id.
The first 40 ids whose reference solution passes every assert in the Docker grader are kept.
"""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "first_answer"))
from common import HARNESS, DockerGrader, chash, grade_hidden, mbpp_task  # noqa: E402

SALT = "speculative-powermh-fresh40-20261123"


def main():
    fa = HERE.parent / "first_answer/panels/pilot40.json"
    pilot = json.loads(fa.read_text())
    dev24 = json.loads((HERE.parent / "first_answer_localized/dev24.json").read_text())["ordered_ids"]
    banned = set()
    for v in pilot["excluded_ids"].values():
        banned |= set(int(x) for x in v)
    banned |= {int(t["task_id"]) for t in pilot["tasks"]}
    banned |= {int(x["task_id"]) for x in pilot.get("skipped_reference_failures", [])}
    banned |= set(int(x) for x in dev24)
    rows = json.loads((HARNESS / "feedback_policy_v2/population_replication56/mbppplus_rows.json").read_text())
    rows = rows["rows"] if isinstance(rows, dict) else rows
    rows = [r.get("row", r) for r in rows]
    pool = [r for r in rows if 11 <= r["task_id"] <= 600 and r["task_id"] not in banned]
    assert not any(27 <= r["task_id"] <= 74 for r in pool)
    pool.sort(key=lambda r: hashlib.sha256(f"{SALT}:{r['task_id']}".encode()).hexdigest())
    grader = DockerGrader()
    keep, skipped = [], []
    for r in pool:
        r = dict(r)
        r["text"] = r["prompt"]
        r["test_setup_code"] = "\n".join(r.get("test_imports") or [])
        t = mbpp_task(r, family="mbppplus")
        lab, _ = grade_hidden(grader, t, {chash(t.reference): t.reference})
        v = next(iter(lab.values()))
        (keep if v["all"] else skipped).append(t if v["all"] else {"task_id": t.task_id, "tests": v["tests"]})
        if len(keep) == 40:
            break
    out = {"name": "fresh40", "salt": SALT, "pool_size": len(pool), "banned_count": len(banned),
           "source": "evalplus/mbppplus local copy (same file as pilot40)",
           "skipped_reference_failures": skipped, "tasks": [asdict(t) for t in keep]}
    (HERE / "panels/fresh40.json").write_text(json.dumps(out, indent=1))
    print("fresh40", len(keep), "pool", len(pool), "skipped", len(skipped), [t.task_id for t in keep])


if __name__ == "__main__":
    main()
