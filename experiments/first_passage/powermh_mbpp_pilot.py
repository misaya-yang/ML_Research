"""Instruct-model pilot on MBPP: standard PowerMH, its exact early-rejection
run, low-temperature decoding and independent resampling, on the same rows,
prompt, stop rule and grader as the repository's MBPP harness.

PowerMH (standard, exact MH): target pi(x) ∝ prod_s p(x_s|h_s)^alpha over the
response up to the native stop (EOS, second code fence, or the token cap).
Proposal: cut c ~ U{0..n-1} inside the current response of length n, keep
x_<c, regenerate from q = p^alpha / Z (temperature 1/alpha, no truncation)
until the same stop rule. Acceptance:
    A = min(1, exp(sum_{s>c} log Z(h'_s) - sum_{s>c} log Z(h_s)) * n / n').
log Z <= 0 for alpha >= 1 and n' >= c + j + 1 after j decode forwards, so
    B_j = exp(partial_j - Phi) * n / (c + j + 1)
is a non-increasing upper bound on A. Early rejection draws U first and stops
the proposal at the first j with B_j < U. Same tokens, same U, same decision.

Engine: one HF model on MPS in bf16. The state keeps its KV cache; a proposal
clones it, crops it to prompt + c - 1 positions and re-enters with one forward,
so the kept prefix is reused, never re-prefilled. Every logit used for a
decision comes from a single-token forward with the same cache length in both
the eager and the early-rejection run, so the two runs should be bitwise equal.
Token randomness is indexed by (seed, task, step, position).

Outputs (run dir): config.json/.sha256 (frozen first), powermh_log.jsonl in the
R0 replay format, arms.jsonl (per problem: all arms, costs, verdicts), run.log.
"""

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "results/local_runs/native-firstpass-1qr7b1va"

# Copied verbatim from HARNESS/independent_mh_screen/native_sampling.py, which
# cannot be imported without loading a second model.
POLICY = ("Return only the complete Python solution in one fenced python code block. "
          "Use the function name and argument order in the public test. Do not include tests.")


def prompt_messages(row):
    return [{"role": "user", "content": POLICY + "\nTask: " + row["text"] + "\nPublic test:\n" + row["test_list"][0]}]


def clean(text):
    fence = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, flags=re.S | re.I)
    return (fence.group(1) if fence else text).strip()


class Engine:
    def __init__(self, model_dir, cap):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps").eval()
        torch.mps.synchronize()
        self.load_s = time.perf_counter() - t0
        eos = self.model.generation_config.eos_token_id
        self.stops = set(eos if isinstance(eos, list) else [eos])
        self.cap = cap

    @torch.inference_mode()
    def forward(self, cache, token):
        out = self.model(input_ids=torch.tensor([[token]], device="mps"), past_key_values=cache, use_cache=True)
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().double().numpy()
        return lp, out.past_key_values

    @torch.inference_mode()
    def start(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_tensors="pt", return_dict=True).input_ids[0].tolist()
        t0 = time.perf_counter()
        out = self.model(input_ids=torch.tensor([ids[:-1]], device="mps"), use_cache=True)
        torch.mps.synchronize()
        return ids, out.past_key_values, time.perf_counter() - t0

    def terminal(self, generated):
        if generated[-1] in self.stops:
            return "eos"
        text = self.tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if text.count("```") >= 2:
            return "fence"
        if len(generated) >= self.cap:
            return "length"
        return None


def draw_q(lp, alpha, key):
    a = alpha * lp
    m = a.max()
    logz = float(m + math.log(np.exp(a - m).sum()))
    q = np.exp(a - logz)
    cdf = np.cumsum(q)
    r = np.random.default_rng(key).random() * cdf[-1]
    return int(min(np.searchsorted(cdf, r, side="right"), len(q) - 1)), logz


def draw_topp(lp, temperature, top_p, key):
    p = np.exp(lp / temperature - np.logaddexp.reduce(lp / temperature))
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order])
    stop = min(int(np.searchsorted(cum, top_p)), len(cum) - 1)
    r = np.random.default_rng(key).random() * cum[stop]
    return int(order[min(int(np.searchsorted(cum[: stop + 1], r, side="right")), stop)])


def generate(eng, prompt_ids, src_cache, prefix, sampler, key, check=None):
    """Clone src_cache, crop to len(prompt)+len(prefix)-1, re-enter with one
    forward, then sample until the stop rule. Returns a dict with tokens
    (prefix + new), per-position log Z for new positions (q sampler only),
    the cache, counts and seconds. check=(logu, phi, n, c) enables early
    rejection on the bound B_j."""
    t0 = time.perf_counter()
    full = prompt_ids + prefix
    cache = copy.deepcopy(src_cache)
    cache.crop(len(full) - 1)
    lp, cache = eng.forward(cache, full[-1])
    gen, logzs, decodes, partial, stopped = [], [], 0, 0.0, False
    while True:
        pos = len(prefix) + len(gen)
        kind, arg = sampler
        if kind == "q":
            v, lz = draw_q(lp, arg, key + [pos])
            logzs.append(lz)
        elif kind == "greedy":
            v = int(lp.argmax())
        else:
            v = draw_topp(lp, arg[0], arg[1], key + [pos])
        gen.append(v)
        if kind == "q" and len(gen) > 1:
            partial += logzs[-1]
        reason = eng.terminal(prefix + gen)
        if reason:
            break
        if check is not None and len(gen) > 1:
            logu, phi, n, c = check
            if partial - phi + math.log(n / (c + len(gen))) < logu:
                stopped = True
                break
        lp, cache = eng.forward(cache, v)
        decodes += 1
    return {"tokens": prefix + gen, "logz": logzs, "cache": cache, "decodes": decodes,
            "reentry": 1, "reason": reason if not stopped else "early_reject",
            "stopped": stopped, "sec": time.perf_counter() - t0}


def powermh(eng, prompt_ids, base, task, cfg, lazy):
    seed, alpha, K = cfg["seed"], cfg["alpha"], cfg["K"]
    t0 = time.perf_counter()
    init = generate(eng, prompt_ids, base, [], ("q", alpha), [seed, task, 0])
    x, xlz, xcache = init["tokens"], init["logz"], init["cache"]
    log = {"init": {"tokens": list(x), "logz": list(xlz), "sec": init["sec"], "reason": init["reason"]},
           "steps": []}
    decodes, reentry, kept = init["decodes"], 1, 0
    for k in range(1, K + 1):
        rng = np.random.default_rng([seed, task, k, 10**6])
        n = len(x)
        c = int(rng.integers(0, n))
        logu = math.log(rng.random())
        phi = sum(xlz[c + 1:])
        prop = generate(eng, prompt_ids, xcache, x[:c], ("q", alpha), [seed, task, k],
                        check=(logu, phi, n, c) if lazy else None)
        decodes += prop["decodes"]
        reentry += 1
        kept += len(prompt_ids) + c - 1
        if prop["stopped"]:
            acc, log_ratio = False, None
        else:
            n2 = len(prop["tokens"])
            log_ratio = sum(prop["logz"][1:]) - phi + math.log(n / n2)
            acc = logu < log_ratio
        log["steps"].append({"k": k, "c": c, "logu": logu, "n": n, "acc": acc, "log_ratio": log_ratio,
                             "tokens": prop["tokens"][c:], "logz": prop["logz"], "decodes": prop["decodes"],
                             "stopped": prop["stopped"], "reason": prop["reason"], "sec": prop["sec"],
                             "prefill": 0, "kept_cached": len(prompt_ids) + c - 1})
        if acc:
            x, xlz, xcache = prop["tokens"], xlz[:c] + prop["logz"], prop["cache"]
    text = eng.tok.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"log": log, "final_tokens": x, "final_text": text, "final_code": clean(text),
            "init_tokens": log["init"]["tokens"], "decodes": decodes, "reentry": reentry,
            "kept_cached_positions": kept, "sec": time.perf_counter() - t0}


def iid_arm(eng, prompt_ids, base, task, spec, budget, max_attempts):
    attempts, spent = [], 0
    for a in range(1, max_attempts + 1):
        if spent >= budget:
            break
        if spec["kind"] == "q":
            key = [spec["seed"], task, 0] if a == 1 else [spec["seed"], task, 100 + a]
            sampler = ("q", spec["alpha"])
        elif spec["kind"] == "greedy_then_topp":
            key = [spec["seed"], task, a]
            sampler = ("greedy", None) if a == 1 else ("topp", (spec["temperature"], spec["top_p"]))
        g = generate(eng, prompt_ids, base, [], sampler, key)
        cost = g["decodes"] + g["reentry"]
        spent += cost
        text = eng.tok.decode(g["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        attempts.append({"a": a, "positions": cost, "sec": g["sec"], "reason": g["reason"],
                         "tokens": g["tokens"], "code": clean(text)})
    return attempts


def grade_codes(row, codes):
    sys.path.insert(0, str(HARNESS / "feedback_policy_v2"))
    from grade_batch import grade
    jobs = []
    for h, code in codes.items():
        for j, test in enumerate(row["test_list"]):
            jobs.append([f"{h}/{j}", code, row.get("test_setup_code", ""), test])
    rows, sec = grade(jobs)
    verdict = {r[0]: r[1] for r in rows}
    out = {}
    for h in codes:
        tests = [verdict[f"{h}/{j}"] for j in range(len(row["test_list"]))]
        out[h] = {"public": tests[0], "all": all(t == "pass" for t in tests), "tests": tests}
    return out, sec


def chash(code):
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", default=None, help="override, smoke only")
    ap.add_argument("--K", type=int, default=None, help="override, smoke only")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["run_dir"])
    if args.K is not None:
        cfg["K"] = args.K
    rows = json.loads((HARNESS / "mbpp_source.json").read_text())["rows"]
    a, b = map(int, (args.rows or cfg["rows"]).split(":"))
    eng = Engine(cfg["model_dir"], cfg["cap"])
    logf = (out / "powermh_log.jsonl").open("a")
    armf = (out / "arms.jsonl").open("a")
    print(json.dumps({"load_s": eng.load_s, "stops": sorted(eng.stops)}), flush=True)
    for idx in range(a, b):
        row = rows[idx]["row"]
        task = int(row["task_id"])
        prompt_ids, base, prefill_s = eng.start(prompt_messages(row))
        order = ["eager", "lazy"] if idx % 2 == 0 else ["lazy", "eager"]
        runs = {}
        for mode in order:
            runs[mode] = powermh(eng, prompt_ids, base, task, cfg, lazy=(mode == "lazy"))
        e, l = runs["eager"], runs["lazy"]
        same_states = all(se["acc"] == sl["acc"] and (not se["acc"] or se["tokens"] == sl["tokens"])
                          and se["c"] == sl["c"] for se, sl in zip(e["log"]["steps"], l["log"]["steps"]))
        lz_diff = 0.0
        for se, sl in zip(e["log"]["steps"], l["log"]["steps"]):
            m = min(len(se["logz"]), len(sl["logz"]))
            if m:
                lz_diff = max(lz_diff, max(abs(p - q) for p, q in zip(se["logz"][:m], sl["logz"][:m])))
        budget = e["decodes"] + e["reentry"]
        iid = {name: iid_arm(eng, prompt_ids, base, task, spec, cfg["iid_budget_factor"] * budget,
                             cfg["iid_max_attempts"]) for name, spec in cfg["iid_arms"].items()}
        codes = {chash(e["final_code"]): e["final_code"], chash(l["final_code"]): l["final_code"]}
        init_text = eng.tok.decode(e["init_tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        codes[chash(clean(init_text))] = clean(init_text)
        for atts in iid.values():
            for att in atts:
                codes[chash(att["code"])] = att["code"]
        verdicts, grade_s = grade_codes(row, codes)
        rec = {"row_idx": idx, "task": task, "prompt_len": len(prompt_ids), "prefill_s": prefill_s,
               "order": order, "pathwise_identical_states": same_states,
               "final_identical": e["final_tokens"] == l["final_tokens"], "max_abs_logz_diff": lz_diff,
               "low_temp": {"code_hash": chash(clean(init_text)), "positions": len(e["log"]["init"]["logz"]),
                            "sec": e["log"]["init"]["sec"]},
               "powermh": {"code_hash": chash(e["final_code"]), "positions": budget, "decodes": e["decodes"],
                           "reentry": e["reentry"], "kept_cached_positions": e["kept_cached_positions"],
                           "sec": e["sec"], "accepted": sum(s["acc"] for s in e["log"]["steps"])},
               "powermh_early_rejection": {"code_hash": chash(l["final_code"]), "positions": l["decodes"] + l["reentry"],
                                           "decodes": l["decodes"], "reentry": l["reentry"], "sec": l["sec"],
                                           "stopped_early": sum(s["stopped"] for s in l["log"]["steps"])},
               "iid": {name: [{k2: v2 for k2, v2 in att.items() if k2 != "tokens"} | {"code_hash": chash(att["code"])}
                              for att in atts] for name, atts in iid.items()},
               "verdicts": verdicts, "codes": codes, "grade_s": grade_s}
        armf.write(json.dumps(rec) + "\n")
        armf.flush()
        logf.write(json.dumps({"problem": task, "row_idx": idx, "prompt_len": len(prompt_ids),
                               "init": e["log"]["init"], "steps": e["log"]["steps"],
                               "final_len": len(e["final_tokens"]),
                               "init_correct": verdicts[chash(clean(init_text))]["all"],
                               "final_correct": verdicts[chash(e["final_code"])]["all"]}) + "\n")
        logf.flush()
        print(json.dumps({"task": task, "identical": same_states and rec["final_identical"],
                          "lz_diff": lz_diff, "pmh": verdicts[chash(e["final_code"])]["all"],
                          "low": verdicts[chash(clean(init_text))]["all"],
                          "pmh_pos": budget, "er_pos": rec["powermh_early_rejection"]["positions"],
                          "pmh_s": round(e["sec"], 1), "er_s": round(l["sec"], 1),
                          "acc": rec["powermh"]["accepted"]}), flush=True)


if __name__ == "__main__":
    main()
