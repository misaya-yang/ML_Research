"""Development pilot: how many target decode positions does exact early
rejection remove from suffix-resampling power MH (PowerMH) on real text?

Policy (fixed horizon T, deterministic padding after EOS, single block):
  target  pi(x) ∝ prod_s p(x_s | h_s)^alpha
  proposal q(v|h) = p(v|h)^alpha / Z(h)      (temperature 1/alpha, no truncation)
  init    x ~ q;  each MH step: cut c ~ U{0..T-1}, regenerate x_c.. from q,
          accept iff U < exp(sum_{s>c} log Z(h'_s) - sum_{s>c} log Z(h_s)).
log Z(h) <= 0 for alpha >= 1, so the partial proposal sum is a monotone upper
bound and a pre-drawn U decides rejection exactly as soon as
partial < log U + Phi. Lazy and eager runs are pathwise identical under shared
U and per-(problem, step, position) token randomness, so this script runs the
eager chain once and counts, for every proposal, the decode positions the lazy
rule would have skipped. No lazy wall-clock is claimed here.

The log is complete: initial tokens and per-position log Z, and for every MH
step the cut, log U, proposal tokens, per-position log Z and decode seconds.
replay_powermh_headroom.py rebuilds the chain from these logs alone and finds,
for each rejection, the earliest token at which it was already decided.

Development only: MATH-500 rows are a screen, not a frozen held-out test.
"""

import argparse
import hashlib
import json
import math
import os
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j : k - 1] if depth == 0 else None


def norm(s):
    if s is None:
        return None
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


class Chain:
    def __init__(self, model, tok, device, alpha, T, eos_ids, seed):
        self.model, self.tok, self.device = model, tok, device
        self.alpha, self.T, self.eos = alpha, T, set(eos_ids)
        self.seed = seed
        self.decode_calls = 0
        self.prefill_positions = 0

    @torch.no_grad()
    def _logits(self, ids, past=None):
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def _q_and_logz(self, logits):
        lp = torch.log_softmax(logits.double().cpu(), dim=-1).numpy()
        a = self.alpha * lp
        m = a.max()
        logz = float(m + math.log(np.exp(a - m).sum()))
        q = np.exp(a - logz)
        return q, logz

    def _draw(self, q, key):
        r = np.random.default_rng(key).random()
        cdf = np.cumsum(q)
        return int(min(np.searchsorted(cdf, r * cdf[-1], side="right"), len(q) - 1))

    def generate(self, prompt_ids, prefix, step, problem):
        """Generate from len(prefix) to horizon. Returns tokens and log Z at
        every generated position s (log Z of the prefix h_s used to draw x_s).
        Forward passes are counted as decode calls except the prefill."""
        ids = torch.tensor([prompt_ids + prefix], device=self.device)
        self.prefill_positions += ids.shape[1]
        logits, past = self._logits(ids)
        toks, logzs = list(prefix), []
        s = len(prefix)
        t0 = time.time()
        while s < self.T:
            q, logz = self._q_and_logz(logits)
            v = self._draw(q, [self.seed, problem, step, s])
            toks.append(v)
            logzs.append(logz)
            s += 1
            if v in self.eos or s >= self.T:
                break
            logits, past = self._logits(torch.tensor([[v]], device=self.device), past)
            self.decode_calls += 1
        self.last_seconds = time.time() - t0
        return toks, logzs


def run_problem(chain, tok, row, problem, K, alpha, T):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["problem"]}]
    if tok.chat_template:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
                  f"{row['problem']}<|im_end|>\n<|im_start|>assistant\n")
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    toks, logz = chain.generate(prompt_ids, [], step=0, problem=problem)
    init_log = {"tokens": list(toks), "logz": list(logz), "sec": getattr(chain, "last_seconds", None)}
    # log Z per position for the current state; padding positions carry 0.
    cur_tok, cur_logz = toks, logz + [0.0] * (T - len(logz))
    init_text = tok.decode(cur_tok, skip_special_tokens=True)
    steps = []
    for k in range(1, K + 1):
        rng = np.random.default_rng([chain.seed, problem, k, 10**6])
        c = int(rng.integers(0, T))
        logu = math.log(rng.random())
        n_cur = len(cur_tok)
        if c >= n_cur:  # cut inside deterministic padding: identical proposal
            steps.append({"k": k, "c": c, "pad": True, "acc": True, "eager": 0, "lazy": 0,
                          "logu": logu})
            continue
        phi = sum(cur_logz[c + 1 :])
        new_tok, new_logz_tail = chain.generate(prompt_ids, cur_tok[:c], step=k, problem=problem)
        # new_logz_tail[j] is log Z at position c + j; positions > c are paid decodes
        tail = new_logz_tail[1:]
        eager = len(tail)
        partial, lazy, stop = 0.0, eager, None
        for j, lz in enumerate(tail):
            partial += lz
            if partial < logu + phi:
                lazy, stop = j + 1, c + 1 + j
                break
        total = sum(tail)
        acc = logu < total - phi
        assert acc == (stop is None), "lazy and eager decisions differ"
        steps.append({
            "k": k, "c": c, "pad": False, "acc": acc, "eager": eager, "lazy": lazy,
            "log_ratio": total - phi, "logu": logu, "cur_len": n_cur, "prop_len": len(new_tok),
            "tokens": new_tok[c:], "logz": list(new_logz_tail),
            "sec": getattr(chain, "last_seconds", None), "prefill": len(prompt_ids) + c,
        })
        if acc:
            cur_tok = new_tok
            cur_logz = cur_logz[:c] + new_logz_tail + [0.0] * (T - c - len(new_logz_tail))
    final_text = tok.decode(cur_tok, skip_special_tokens=True)
    gold = norm(row["answer"])
    return {
        "problem": problem, "unique_id": row.get("unique_id"), "level": row.get("level"),
        "init_len": len(toks), "final_len": len(cur_tok),
        "init_correct": norm(boxed(init_text)) == gold, "final_correct": norm(boxed(final_text)) == gold,
        "init_answer": boxed(init_text), "final_answer": boxed(final_text), "gold": row["answer"],
        "prompt_len": len(prompt_ids), "init": init_log, "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--rows", default="0:20")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)
    cfg = vars(args) | {"system": SYSTEM, "dtype": "bfloat16", "device": "mps",
                        "contract": "pathwise eager chain; lazy counts derived on shared randomness"}
    blob = json.dumps(cfg, sort_keys=True).encode()
    with open(os.path.join(args.out, "config.json"), "wb") as f:
        f.write(blob)
    with open(os.path.join(args.out, "config.sha256"), "w") as f:
        f.write(hashlib.sha256(blob).hexdigest() + "\n")
    rows = [json.loads(l) for l in open(args.data)]
    a, b = map(int, args.rows.split(":"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("mps").eval()
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")}
    chain = Chain(model, tok, "mps", args.alpha, args.T, eos, args.seed)
    t0 = time.time()
    with open(os.path.join(args.out, "raw.jsonl"), "w") as f:
        for i in range(a, b):
            t = time.time()
            rec = run_problem(chain, tok, rows[i], i, args.K, args.alpha, args.T)
            rec["sec"] = time.time() - t
            f.write(json.dumps(rec) + "\n")
            f.flush()
            e = sum(s["eager"] for s in rec["steps"])
            l = sum(s["lazy"] for s in rec["steps"])
            print(f"row {i} sec {rec['sec']:.1f} eager {e} lazy {l} acc "
                  f"{sum(s['acc'] and not s['pad'] for s in rec['steps'])} "
                  f"init_ok {rec['init_correct']} final_ok {rec['final_correct']}", flush=True)
    print(json.dumps({"wall_s": time.time() - t0, "decode_calls": chain.decode_calls,
                      "prefill_positions": chain.prefill_positions}), flush=True)


if __name__ == "__main__":
    main()
