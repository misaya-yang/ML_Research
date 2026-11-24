"""Shared pieces for the first-answer steering experiment.

Everything a policy may see goes through Task.messages(); hidden tests live
only in Task.hidden and are read by grade_hidden() after an answer is fixed.
Every model forward goes through Engine, which charges a Ledger and runs the
leakage guard on the decoded input.

Compute unit: model positions = tokens pushed through a forward (prompt
prefill, one re-entry token after a cache crop, each decode step, and every
auxiliary stream). KV reuse is allowed for every arm alike. Visible-test
executions are counted separately (calls and seconds).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "results/local_runs/native-firstpass-1qr7b1va"

# Verbatim from HARNESS/independent_mh_screen/native_sampling.py.
POLICY = ("Return only the complete Python solution in one fenced python code block. "
          "Use the function name and argument order in the public test. Do not include tests.")
# Matched generic planning: same constraints, plus a task-agnostic plan request.
PLAN_POLICY = ("First write a short plan of at most three numbered lines saying what the function must compute. "
               "Then return the complete Python solution in one fenced python code block. "
               "Use the function name and argument order in the public test. Do not include tests.")
HUMANEVAL_POLICY = ("Complete the function below. Return only the complete Python solution, including the "
                    "signature, in one fenced python code block. Do not include tests.")


def clean(text):
    """Verbatim code extraction rule of the MBPP harness."""
    fence = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, flags=re.S | re.I)
    return (fence.group(1) if fence else text).strip()


def chash(code):
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- tasks

REWORD_RULES = [  # semantics-preserving framing rewrites, applied once, first match only
    (r"^Write a python function to ", "Implement a Python function that will "),
    (r"^Write a function to ", "Implement a function that will "),
    (r"^Write a python function that ", "Implement a Python function that "),
    (r"^Write a function that ", "Implement a function that "),
    (r"^Write a python function which ", "Implement a Python function which "),
    (r"^Write a function which ", "Implement a function which "),
]


def reword(text):
    """Template rewording of the task framing. Identifiers, literals and the
    requirement clause are left untouched, so the task semantics are unchanged.
    Returns (new_text, rule_index or None)."""
    for i, (pat, rep) in enumerate(REWORD_RULES):
        if re.match(pat, text, flags=re.I):
            return re.sub(pat, rep, text, count=1, flags=re.I), i
    return "Task description: " + text, None


def mask_expected_output(assert_line):
    """Mask the expected value of a visible assert, keeping the call verbatim.

    `assert f(x) == y` -> `assert f(x) == ...`, splicing into the original
    source span so quoting and spacing of the call are unchanged. Returns None
    unless the assert is a single-line, single comparison with exactly one side
    that calls a non-builtin function."""
    import builtins
    line = assert_line.strip()
    try:
        tree = ast.parse(line)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert) or "\n" in line:
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return None
    left, right = test.left, test.comparators[0]
    # A side "has a call" only if it calls something other than a builtin or a
    # module attribute (set(...), sorted(...), math.sqrt(...) belong to the value).
    has_call = lambda n: any(isinstance(m, ast.Call) and isinstance(m.func, ast.Name)
                             and not hasattr(builtins, m.func.id) for m in ast.walk(n))
    if has_call(left) and not has_call(right):
        node = right
    elif has_call(right) and not has_call(left):
        node = left
    else:
        return None
    b = line.encode()
    return (b[:node.col_offset] + b"..." + b[node.end_col_offset:]).decode()


def doctest_asserts(prompt, entry):
    """Visible examples of a HumanEval prompt as asserts (doctest form only)."""
    out = []
    lines = prompt.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(">>> ") and entry in s and i + 1 < len(lines):
            want = lines[i + 1].strip()
            if want and not want.startswith(">>>"):
                out.append(f"assert {s[4:]} == {want}")
    return out


@dataclass
class Task:
    task_id: str
    family: str
    text: str
    visible: list
    hidden: list
    setup: str = ""
    entry: str | None = None
    reference: str | None = None
    raw_prompt: str | None = None  # HumanEval prompt (signature + docstring)

    def user_content(self, policy=POLICY, variant="base", reworded=False):
        text = reword(self.text)[0] if reworded else self.text
        if self.family == "humaneval":
            prompt = self.raw_prompt
            if variant == "masked_output":
                for a in self.visible:
                    m = mask_expected_output(a)
                    if m is None:
                        continue
                    want = a.split("==", 1)[1].strip()
                    prompt = prompt.replace("\n" + " " * 4 + want, "\n    ...", 1)
            return (HUMANEVAL_POLICY if policy == POLICY else policy) + "\n" + prompt
        vis = self.visible[0]
        if variant == "masked_output":
            vis = mask_expected_output(vis) or vis
        return policy + "\nTask: " + text + "\nPublic test:\n" + vis

    def messages(self, **kw):
        return [{"role": "user", "content": self.user_content(**kw)}]

    def hidden_fragments(self):
        """Strings that must never reach a decoder input."""
        vis = " ".join(self.visible)
        frags = []
        for h in self.hidden:
            for line in h.splitlines():
                s = line.strip()
                if s.startswith("assert") and len(s) > 12 and s not in vis:
                    frags.append(s)
        return frags


def mbpp_task(row, family="mbpp"):
    tests = list(row["test_list"])
    return Task(task_id=str(row["task_id"]), family=family, text=row.get("text") or row.get("prompt"),
                visible=tests[:1], hidden=tests[1:], setup=row.get("test_setup_code", "") or "",
                reference=row.get("code"))


def humaneval_task(row):
    entry = row["entry_point"]
    return Task(task_id=row["task_id"], family="humaneval", text="", visible=doctest_asserts(row["prompt"], entry),
                hidden=[row["test"].rstrip() + f"\n\ncheck({entry})\n"], entry=entry,
                reference=row["prompt"] + row["canonical_solution"], raw_prompt=row["prompt"])


def load_panel(path):
    d = json.loads(Path(path).read_text())
    tasks = []
    for t in d["tasks"]:
        tasks.append(Task(**t))
    return d, tasks


# --------------------------------------------------------------------------- compute ledger


@dataclass
class Ledger:
    prefill: int = 0
    decode: int = 0
    reentry: int = 0
    aux_prefill: int = 0
    aux_decode: int = 0
    exec_calls: int = 0
    exec_s: float = 0.0
    model_s: float = 0.0
    events: list = field(default_factory=list)

    @property
    def positions(self):
        return self.prefill + self.decode + self.reentry + self.aux_prefill + self.aux_decode

    def as_dict(self):
        return {"positions": self.positions, "prefill": self.prefill, "decode": self.decode,
                "reentry": self.reentry, "aux_prefill": self.aux_prefill, "aux_decode": self.aux_decode,
                "exec_calls": self.exec_calls, "exec_s": round(self.exec_s, 4), "model_s": round(self.model_s, 4)}


class LeakError(RuntimeError):
    pass


def _norm(s):
    return re.sub(r"\s+", "", s)


# --------------------------------------------------------------------------- engine


class Engine:
    """Policy-facing model wrapper. backend implements encode_chat, decode,
    prefill(ids) -> (logprobs, cache), step(cache, token) -> (logprobs, cache),
    clone(cache, keep) and has .stops. Log-probabilities are float64 numpy."""

    def __init__(self, backend, cap=192):
        self.b = backend
        self.cap = cap
        self.stops = set(backend.stops)
        self.ledger = Ledger()
        self.forbidden = []

    def bind(self, ledger, task):
        self.ledger = ledger
        self.forbidden = [_norm(f) for f in task.hidden_fragments()]

    def _guard(self, ids):
        text = _norm(self.b.decode(ids))
        for f in self.forbidden:
            if f in text:
                raise LeakError(f"hidden test fragment reached a decoder input: {f[:60]}")

    def encode(self, messages):
        return self.b.encode_chat(messages)

    def prefill(self, ids, aux=False):
        self._guard(ids)
        t0 = time.perf_counter()
        lp, cache = self.b.prefill(ids)
        self.ledger.model_s += time.perf_counter() - t0
        if aux:
            self.ledger.aux_prefill += len(ids)
        else:
            self.ledger.prefill += len(ids)
        return lp, cache

    def step(self, cache, token, aux=False, reentry=False):
        t0 = time.perf_counter()
        lp, cache = self.b.step(cache, token)
        self.ledger.model_s += time.perf_counter() - t0
        if aux:
            self.ledger.aux_decode += 1
        elif reentry:
            self.ledger.reentry += 1
        else:
            self.ledger.decode += 1
        return lp, cache

    def clone(self, cache, keep):
        return self.b.clone(cache, keep)

    def text(self, ids):
        return self.b.decode(ids)

    def terminal(self, gen, cap=None):
        if gen and gen[-1] in self.stops:
            return "eos"
        if self.b.decode(gen).count("```") >= 2:
            return "fence"
        if len(gen) >= (cap or self.cap):
            return "length"
        return None


class HFBackend:
    def __init__(self, model_dir, device="mps"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        eos = self.model.generation_config.eos_token_id
        self.stops = set(eos if isinstance(eos, list) else [eos])

    def encode_chat(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            return_tensors="pt", return_dict=True).input_ids[0].tolist()

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _lp(self, out):
        return self.torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().double().numpy()

    def prefill(self, ids):
        with self.torch.inference_mode():
            out = self.model(input_ids=self.torch.tensor([ids], device=self.device), use_cache=True)
            return self._lp(out), out.past_key_values

    def step(self, cache, token):
        with self.torch.inference_mode():
            out = self.model(input_ids=self.torch.tensor([[token]], device=self.device),
                             past_key_values=cache, use_cache=True)
            return self._lp(out), out.past_key_values

    def clone(self, cache, keep):
        c = copy.deepcopy(cache)
        c.crop(keep)
        return c


# --------------------------------------------------------------------------- decoding primitives


def topk(lp, k=4):
    idx = np.argpartition(-lp, k)[:k]
    idx = idx[np.argsort(-lp[idx], kind="stable")]
    return [int(i) for i in idx], [float(lp[i]) for i in idx]


def code_region_state(text):
    """(in_code, in_comment, in_docstring) for the next token given text so far."""
    m = re.search(r"```(?:python|py)?[ \t]*\n", text, flags=re.I)
    body = text[m.end():] if m else text
    if m is None and text.lstrip().startswith("```"):
        return False, False, False
    line = body.rsplit("\n", 1)[-1]
    in_comment = line.lstrip().startswith("#")
    in_doc = (body.count('"""') + body.count("'''")) % 2 == 1
    return True, in_comment, in_doc


def fork_alternatives(tok_ids, tok_lps, ratio, k_alt):
    """Alternatives v != top-1 with log p(top) - log p(v) <= ratio."""
    top = tok_lps[0]
    return [(i, l) for i, l in zip(tok_ids[1:], tok_lps[1:]) if top - l <= ratio][:k_alt]


def run_decode(eng, cache, lp, ctx_len, gen, chooser, cap=None, budget=None, record=None):
    """Continue decoding from a state whose next-token log-probs are lp.

    gen holds tokens already decided after the context (their KV is in cache,
    except that the last decided token is not yet processed when lp is None).
    chooser(lp, pos) -> token. record(pos, lp) is called before each choice.
    Stops on the harness stop rule, or returns aborted=True when the ledger
    would exceed budget (an incomplete program is never submitted)."""
    gen = list(gen)
    seq_lp = []
    while True:
        pos = len(gen)
        if record is not None:
            record(pos, lp)
        v = chooser(lp, pos)
        seq_lp.append(float(lp[v]))
        gen.append(v)
        reason = eng.terminal(gen, cap)
        if reason:
            return {"gen": gen, "cache": cache, "reason": reason, "aborted": False, "seq_lp": seq_lp}
        if budget is not None and eng.ledger.positions + 1 > budget:
            return {"gen": gen, "cache": cache, "reason": "budget", "aborted": True, "seq_lp": seq_lp}
        lp, cache = eng.step(cache, v)


def greedy_chooser(lp, pos):
    return int(np.argmax(lp))


# --------------------------------------------------------------------------- top_no sampler port

def keep_mask(lp, strategy, **kw):
    """numpy port of top_no experiments/samplers.py get_keep_mask (commit
    85d849b832a7) for the strategies used here. Operates on log-probs, which
    differ from raw logits by a constant, so every rule is unchanged."""
    if strategy == "greedy":
        m = np.zeros_like(lp, dtype=bool)
        m[int(np.argmax(lp))] = True
        return m
    if strategy == "min_p":
        p_min = float(kw.get("p_min", 0.05))
        if p_min <= 0:
            return np.ones_like(lp, dtype=bool)
        return (lp.max() - lp) <= -math.log(p_min)
    if strategy == "fixed_margin":
        return (lp.max() - lp) <= float(kw.get("margin", 3.0))
    if strategy == "top_nsigma":
        return (lp.max() - lp) <= float(kw.get("n_sigma", 2.0)) * lp[np.isfinite(lp)].std(ddof=1)
    if strategy == "top_p":
        p = float(kw.get("p", 0.95))
        if p >= 1:
            return np.ones_like(lp, dtype=bool)
        order = np.argsort(-lp, kind="stable")
        probs = np.exp(lp[order])
        cum = np.cumsum(probs)
        remove = cum > p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        m = np.zeros_like(lp, dtype=bool)
        m[order[~remove]] = True
        return m
    if strategy == "eta":
        # eta-sampling (Hewitt et al., 2022): keep p > min(eps, sqrt(eps) * exp(-H)).
        eps = float(kw.get("epsilon", 9e-4))
        p = np.exp(lp)
        h = -float(np.sum(p[p > 0] * lp[p > 0]))
        m = p > min(eps, math.sqrt(eps) * math.exp(-h))
        m[int(np.argmax(lp))] = True
        return m
    if strategy == "none":
        return np.ones_like(lp, dtype=bool)
    raise ValueError(f"unsupported strategy {strategy}")


def sampler_chooser(spec, key):
    """Seeded inverse-CDF sampling after truncation; temperature after truncation (top_no order)."""
    strategy, params, temperature = spec["strategy"], spec.get("params", {}), float(spec["temperature"])

    def choose(lp, pos):
        if strategy == "greedy" or temperature <= 0:
            return int(np.argmax(lp))
        m = keep_mask(lp, strategy, **params)
        z = np.where(m, lp / temperature, -np.inf)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        u = np.random.default_rng(key + [pos]).random()
        return int(min(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1))
    return choose


def stable_int(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- graders


class DockerGrader:
    """MBPP/HumanEval assertions in the repository's confined CPython container."""

    def __init__(self):
        sys.path.insert(0, str(HARNESS / "feedback_policy_v2"))
        from grade_batch import grade
        self._grade = grade

    def run(self, items):
        """items: list of (key, code, setup, test). Returns {key: status}, seconds."""
        if not items:
            return {}, 0.0
        t0 = time.perf_counter()
        rows, _ = self._grade([list(x) for x in items])
        return {r[0]: r[1] for r in rows}, time.perf_counter() - t0


class VisibleChecker:
    """The only test access a policy gets: the visible assert(s), counted."""

    def __init__(self, grader, task, ledger):
        self.g, self.task, self.ledger, self.cache = grader, task, ledger, {}

    def __call__(self, code):
        h = chash(code)
        if h in self.cache:
            return self.cache[h]
        if not self.task.visible:
            self.cache[h] = None
            return None
        items = [(f"{h}/v{j}", code, self.task.setup, t) for j, t in enumerate(self.task.visible)]
        verdict, sec = self.g.run(items)
        self.ledger.exec_calls += 1
        self.ledger.exec_s += sec
        ok = all(verdict.get(k) == "pass" for k, *_ in items)
        self.cache[h] = ok
        return ok


def grade_hidden(grader, task, codes):
    """Offline labels: every assert (visible and hidden). Never passed to a policy."""
    items = []
    for h, code in codes.items():
        for j, t in enumerate(task.visible + task.hidden):
            items.append((f"{h}/{j}", code, task.setup, t))
    verdict, sec = grader.run(items)
    out = {}
    n = len(task.visible) + len(task.hidden)
    for h in codes:
        tests = [verdict.get(f"{h}/{j}", "missing") for j in range(n)]
        out[h] = {"visible": all(t == "pass" for t in tests[:len(task.visible)]) if task.visible else None,
                  "all": all(t == "pass" for t in tests), "tests": tests}
    return out, sec
