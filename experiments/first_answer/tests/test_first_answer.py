"""Dry-run tests: scripted model, in-process grader for scripted programs only."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import (POLICY, Engine, Ledger, LeakError, Task, VisibleChecker, chash, clean, keep_mask,  # noqa: E402
                    mask_expected_output, reword)
from fake import FakeBackend, FakeGrader, tiny_task  # noqa: E402
import policies  # noqa: E402
import fork_replay  # noqa: E402

CFG = {"fork_ratio": math.log(4), "k_alt": 2, "max_forks": 4, "max_forks_replay": 12, "extra_budget_factor": 2.0,
       "max_attempts": 16, "plan_extra_tokens": 96, "contrast_lambda": 1.0, "seed": 7,
       "sampler": {"strategy": "min_p", "params": {"p_min": 0.1}, "temperature": 0.7},
       "gate": {"oracle_items_min": 1, "flec_minus_likelihood_min": 1}}
DEV = HERE / "panels/dev16.json"


def setup(task=None, w=(0.55, 0.70)):
    task = task or tiny_task()
    eng = Engine(FakeBackend.from_tasks([task], w=w), cap=192)
    return eng, task, FakeGrader()


def run_arm(name, task=None, w=(0.55, 0.70), cfg=CFG):
    eng, task, g = setup(task, w)
    led = Ledger()
    eng.bind(led, task)
    out = policies.ARMS[name](eng, task, cfg, VisibleChecker(g, task, led))
    lab = g.run([("x", out["code"], "", t) for t in task.visible + task.hidden])[0]
    return out, led, all(v == "pass" for v in lab.values())


def test_direct_ledger_counts_prefill_and_decode():
    out, led, ok = run_arm("direct")
    eng, task, _ = setup()
    n = len(eng.encode(task.messages()))
    assert not ok and "WRONG_FAKE" in out["code"]
    assert led.prefill == n and led.decode == out["gen_len"] - 1 and led.positions == n + out["gen_len"] - 1


def test_flec_fixes_one_fork_and_pays_for_branch():
    d, dled, _ = run_arm("direct")
    out, led, ok = run_arm("flec")
    assert ok and out["decision"] == "fork"
    assert led.positions > dled.positions and led.exec_calls == 2
    assert led.positions <= dled.positions + round(CFG["extra_budget_factor"] * d["gen_len"])


def test_likelihood_control_keeps_confident_wrong_branch():
    out, _, ok = run_arm("likelihood")
    assert not ok  # mean log-prob favours the .55 branch: confidence alone picks the wrong program


def test_flec_free_when_greedy_passes():
    d, dled, dok = run_arm("direct", w=(0.45, 0.30))
    out, led, ok = run_arm("flec", w=(0.45, 0.30))
    assert dok and ok and out["decision"] == "greedy"
    assert led.positions == dled.positions and led.exec_calls == 1


def test_budget_abort_never_submits_partial_program():
    cfg = dict(CFG, extra_budget_factor=0.05)
    out, led, ok = run_arm("flec", cfg=cfg)
    assert out["decision"] == "fallback_greedy" and "WRONG_FAKE" in out["code"]
    out, _, _ = run_arm("best_of_n", cfg=cfg)
    assert out["decision"] == "fallback_greedy"


def test_best_of_n_respects_budget():
    d, dled, _ = run_arm("direct")
    out, led, ok = run_arm("best_of_n")
    assert led.positions <= dled.positions + round(CFG["extra_budget_factor"] * d["gen_len"])


def test_contrast_uses_masked_stream_and_counts_it():
    out, led, ok = run_arm("contrast")
    assert ok and out["steered"] and led.aux_prefill > 0 and led.aux_decode > 0


def test_plan_arm_extracts_plan_and_code():
    out, led, _ = run_arm("plan")
    assert "def add_one" in out["code"] and "plan" in out


def test_leak_guard_blocks_hidden_test_in_prompt():
    t = tiny_task()
    bad = Task(**{**t.__dict__, "text": t.text + " " + t.hidden[0]})
    eng = Engine(FakeBackend.from_tasks([bad]), cap=192)
    eng.bind(Ledger(), bad)
    with pytest.raises(LeakError):
        policies.arm_direct(eng, bad, CFG, None)


def test_visible_evidence_identical_and_hidden_absent_in_all_prompts():
    t = tiny_task()
    for kw in ({}, {"policy": policies.PLAN_POLICY}, {"variant": "masked_output"}, {"reworded": True}):
        c = t.user_content(**kw)
        assert "add_one(1)" in c
        assert all(h not in c for h in t.hidden)
        if kw.get("variant") != "masked_output":
            assert t.visible[0] in c
    assert "== ..." in t.user_content(variant="masked_output")


def test_mask_expected_output_forms():
    assert mask_expected_output("assert f(1) == [2, 3]") == "assert f(1) == ..."
    assert mask_expected_output("assert [2] == f(1)") == "assert ... == f(1)"
    assert mask_expected_output("assert f(1)") is None
    assert mask_expected_output("assert math.isclose(f(1), 2.0)") is None
    assert mask_expected_output("assert set(f([1, 2], 2)) == set([2, 1])") == "assert set(f([1, 2], 2)) == ..."
    assert mask_expected_output("assert sorted(f(1)) == sorted(g(2))") is None
    assert mask_expected_output('assert remove_Occ("hello","l") == "heo"') == 'assert remove_Occ("hello","l") == ...'


def test_reword_changes_only_framing():
    s, rule = reword("Write a python function to remove first and last occurrence of a given character from the string.")
    assert rule == 0 and s.endswith("remove first and last occurrence of a given character from the string.")
    s, rule = reword("Find the volume of a prism.")
    assert rule is None and s.endswith("Find the volume of a prism.")


def test_replay_simulation_matches_live_policies():
    tasks = [tiny_task("a"), Task(task_id="b", family="mbpp", text="Write a function to double a number.",
                                  visible=["assert dbl(2) == 4"], hidden=["assert dbl(3) == 6"],
                                  reference="def dbl(x):\n    return 2 * x")]
    eng = Engine(FakeBackend.from_tasks(tasks), cap=192)
    g = FakeGrader()
    items = [fork_replay.replay_item(eng, t, CFG, g) for t in tasks]
    for t, it in zip(tasks, items):
        sim = fork_replay.simulate(it, CFG)
        for name in ("flec", "likelihood"):
            led = Ledger()
            eng.bind(led, t)
            out = policies.ARMS[name](eng, t, CFG, VisibleChecker(g, t, led))
            lab = g.run([("x", out["code"], "", s) for s in t.visible + t.hidden])[0]
            assert sim[name][0] == all(v == "pass" for v in lab.values())
            assert sim[name][1] == led.positions, (name, sim[name], led.positions)
    s = fork_replay.summarize(items, CFG)
    assert s["oracle_single_fork_items"] == ["a", "b"] and s["simulated"]["flec_correct"] == 2


def test_keep_mask_port_matches_top_no():
    top_no = os.environ.get("TOP_NO_DIR")
    torch = pytest.importorskip("torch")
    if not top_no or not Path(top_no, "experiments/samplers.py").exists():
        pytest.skip("TOP_NO_DIR not set")
    sys.path.insert(0, str(Path(top_no, "experiments")))
    import samplers
    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = rng.normal(0, 3, size=500)
        lp = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
        for strat, kw in (("min_p", {"p_min": 0.1}), ("fixed_margin", {"margin": 2.0}), ("top_p", {"p": 0.9}),
                          ("top_nsigma", {"n_sigma": 1.0})):
            ref = samplers.get_keep_mask(torch.tensor(logits[None]), strat, **kw)[0].numpy()
            assert (keep_mask(lp, strat, **kw) == ref).all(), strat


@pytest.mark.skipif(not DEV.exists(), reason="panels not built")
def test_panels_never_contain_reserved_or_dev_overlap():
    dev = json.loads(DEV.read_text())
    pilot = json.loads((HERE / "panels/pilot40.json").read_text())
    reserved = set(map(str, pilot["excluded_ids"]["reserved48"]))
    dev_ids = {t["task_id"] for t in dev["tasks"]}
    pil_ids = {t["task_id"] for t in pilot["tasks"]}
    assert dev_ids == {str(i) for i in range(11, 27)}
    assert not (reserved & (dev_ids | pil_ids)) and not (dev_ids & pil_ids)
    banned = set().union(*[set(map(str, v)) for v in pilot["excluded_ids"].values()])
    assert not (banned & pil_ids)
    for t in dev["tasks"] + pilot["tasks"]:
        task = Task(**t)
        c = task.user_content()
        assert task.visible[0] in c and all(h not in c for h in task.hidden)
