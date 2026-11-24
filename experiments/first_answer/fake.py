"""Scripted character-level model and in-process grader for dry runs and tests.

FakeBackend.from_tasks(tasks) scripts, for every task, two fenced programs
that share a prefix: a wrong one (the reference with its first body line
replaced by a constant return, probability .55 at the fork) and the reference
(.45). With the expected output masked in the prompt the split becomes .70/.30,
so the contrast rule has something to act on. Greedy is therefore wrong, the
fork is plausible (ratio .45/.55 > 1/4), and the correct program is one
branch away. No weights are loaded.
"""

from __future__ import annotations

import math
import time

import numpy as np

EOS, GEN, OFF, V = 1, 3, 10, 10 + 256


def _enc(text):
    return [OFF + (ord(c) if ord(c) < 256 else ord("?")) for c in text]


def fenced(code):
    return "```python\n" + code.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n```"


def wrong_variant(code):
    lines = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def ") and i + 1 < len(lines):
            body = lines[i + 1]
            indent = body[: len(body) - len(body.lstrip())] or "    "
            return "\n".join(lines[: i + 1] + [indent + "return 'WRONG_FAKE'"])
    return code + "\n# unreachable"


class FakeBackend:
    stops = {EOS}

    def __init__(self, scripts):
        self.scripts = scripts  # list of (trigger, [(program_text, w_full, w_masked)])

    @classmethod
    def from_tasks(cls, tasks, w=(0.55, 0.70)):
        scripts = []
        for t in tasks:
            trigger = t.visible[0] if t.visible else t.task_id
            ok = fenced(t.reference)
            bad = fenced(wrong_variant(t.reference))
            scripts.append((trigger, [(bad, w[0], w[1]), (ok, 1 - w[0], 1 - w[1])]))
        return cls(scripts)

    def encode_chat(self, messages):
        return _enc("".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>") + [GEN]

    def decode(self, ids):
        return "".join(chr(i - OFF) for i in ids if i >= OFF)

    def _lp(self, ctx):
        k = len(ctx) - 1 - ctx[::-1].index(GEN)
        prompt, gen = self.decode(ctx[:k]), list(ctx[k + 1:])
        mass = np.zeros(V)
        masked = "== ..." in prompt or "\n    ..." in prompt
        for trigger, progs in self.scripts:
            masked_trigger = trigger.split("==", 1)[0] if "==" in trigger else trigger
            if trigger in prompt or (masked and masked_trigger in prompt):
                for text, wf, wm in progs:
                    toks = _enc(text) + [EOS]
                    if len(gen) < len(toks) and toks[: len(gen)] == gen:
                        mass[toks[len(gen)]] += wm if masked else wf
                break
        if mass.sum() == 0:
            mass[EOS] = 1.0
        p = mass / mass.sum() * (1 - 1e-6) + 1e-6 / V
        return np.log(p)

    def prefill(self, ids):
        return self._lp(list(ids)), tuple(ids)

    def step(self, cache, token):
        c = tuple(cache) + (token,)
        return self._lp(list(c)), c

    def clone(self, cache, keep):
        return tuple(cache[:keep])


class FakeGrader:
    """Runs trusted scripted programs in-process. Never used on model output."""

    def run(self, items):
        t0 = time.perf_counter()
        out = {}
        for key, code, setup, test in items:
            ns = {}
            try:
                exec(setup or "", ns)
                exec(code, ns)
                exec(test, ns)
                out[key] = "pass"
            except Exception:
                out[key] = "fail"
        return out, time.perf_counter() - t0


def tiny_task(task_id="t1"):
    from common import Task
    return Task(task_id=task_id, family="mbpp", text="Write a function to add one to a number.",
                visible=["assert add_one(1) == 2"], hidden=["assert add_one(5) == 6", "assert add_one(-1) == 0"],
                reference="def add_one(x):\n    return x + 1")


__all__ = ["FakeBackend", "FakeGrader", "fenced", "wrong_variant", "tiny_task", "math"]
