"""First-answer policies. Each returns exactly one submitted program.

A policy receives the Engine (already bound to a fresh Ledger and the task's
leakage guard), the Task, its config, and a VisibleChecker. It never sees
hidden tests or hidden verdicts.

Arms
  direct        greedy decoding (the harness's low-temperature single answer)
  plan          matched generic planning: plan lines then code, one call, greedy
  sampler       one sample from the development-tuned local sampler
  best_of_n     sequence level: greedy, then sampler draws, first program passing
                the visible test, budget-capped; fallback greedy
  likelihood    same forks and branch completions as flec, branch chosen by mean
                token log-probability (confidence only); trigger = greedy fails
                the visible test
  contrast      USCD-like: at forks, argmax of lp + lam * (lp - lp_masked), where
                lp_masked conditions on the visible test with its expected output
                masked; no execution
  flec          proposed: fork-located example check. Greedy; if it fails the
                visible test, visit forks in increasing-margin order, complete
                each plausible alternative greedily, submit the first completion
                that passes the visible test; fallback greedy
"""

from __future__ import annotations

import numpy as np

from common import (PLAN_POLICY, clean, code_region_state, fork_alternatives, greedy_chooser, run_decode,
                    sampler_chooser, stable_int, topk)


def _messages(task, cfg, **kw):
    return task.messages(reworded=cfg.get("reworded", False), **kw)


def greedy_trace(eng, ids, k=4, cap=None):
    rec = {}
    lp0, cache = eng.prefill(ids)
    out = run_decode(eng, cache, lp0, len(ids), [], greedy_chooser, cap=cap,
                     record=lambda pos, lp: rec.__setitem__(pos, topk(lp, k)))
    out["lp0"], out["topk"], out["ctx_len"] = lp0, rec, len(ids)
    return out


def list_forks(eng, tr, ratio, k_alt):
    """Eligible fork positions of a greedy trace, sorted by margin then position."""
    forks = []
    gen = tr["gen"]
    for pos in range(len(gen)):
        ids, lps = tr["topk"][pos]
        top = gen[pos]
        if ids[0] != top:  # exact float tie; keep greedy's token as top-1
            j = ids.index(top) if top in ids else None
            if j is None:
                continue
            ids = [top] + ids[:j] + ids[j + 1:]
            lps = [lps[j]] + lps[:j] + lps[j + 1:]
        alts = fork_alternatives(ids, lps, ratio, k_alt)
        if not alts:
            continue
        in_code, in_comment, in_doc = code_region_state(eng.text(gen[:pos]))
        if not in_code or in_comment or in_doc:
            continue
        strs = [eng.text([top])] + [eng.text([a]) for a, _ in alts]
        if all(not s.strip() for s in strs):
            continue
        forks.append({"pos": pos, "margin": lps[0] - alts[0][1], "top": top, "top_lp": lps[0],
                      "alts": alts, "strs": strs})
    forks.sort(key=lambda f: (f["margin"], f["pos"]))
    return forks


def branch(eng, tr, pos, v, budget=None):
    """Force token v at generated position pos of a greedy trace, then continue greedily."""
    g = tr["gen"][:pos] + [v]
    reason = eng.terminal(g)
    if reason:
        return {"gen": g, "reason": reason, "aborted": False, "seq_lp": []}
    if budget is not None and eng.ledger.positions + 1 > budget:
        return {"gen": g, "reason": "budget", "aborted": True, "seq_lp": []}
    cache = eng.clone(tr["cache"], tr["ctx_len"] + pos)
    lp, cache = eng.step(cache, v)
    return run_decode(eng, cache, lp, tr["ctx_len"], g, greedy_chooser, budget=budget)


def _result(eng, gen, **extra):
    text = eng.text(gen)
    return {"code": clean(text), "text": text, "gen_len": len(gen), **extra}


def arm_direct(eng, task, cfg, vis):
    ids = eng.encode(_messages(task, cfg))
    lp, cache = eng.prefill(ids)
    out = run_decode(eng, cache, lp, len(ids), [], greedy_chooser)
    return _result(eng, out["gen"], reason=out["reason"])


def arm_plan(eng, task, cfg, vis):
    ids = eng.encode(_messages(task, cfg, policy=PLAN_POLICY))
    lp, cache = eng.prefill(ids)
    out = run_decode(eng, cache, lp, len(ids), [], greedy_chooser, cap=eng.cap + cfg["plan_extra_tokens"])
    text = eng.text(out["gen"])
    return _result(eng, out["gen"], reason=out["reason"], plan=text.split("```", 1)[0].strip())


def arm_sampler(eng, task, cfg, vis):
    ids = eng.encode(_messages(task, cfg))
    lp, cache = eng.prefill(ids)
    key = [int(cfg["seed"]), stable_int(task.task_id), 0]
    out = run_decode(eng, cache, lp, len(ids), [], sampler_chooser(cfg["sampler"], key))
    return _result(eng, out["gen"], reason=out["reason"], sampler=cfg["sampler"])


def _budget(eng, tr, cfg):
    return eng.ledger.positions + int(round(cfg["extra_budget_factor"] * len(tr["gen"])))


def arm_best_of_n(eng, task, cfg, vis):
    ids = eng.encode(_messages(task, cfg))
    tr = greedy_trace(eng, ids)
    code0 = clean(eng.text(tr["gen"]))
    log = {"greedy_visible": vis(code0), "attempts": []}
    if log["greedy_visible"] is not False:
        return _result(eng, tr["gen"], decision="greedy", log=log)
    budget = _budget(eng, tr, cfg)
    for a in range(1, cfg["max_attempts"] + 1):
        cache = eng.clone(tr["cache"], len(ids))
        key = [int(cfg["seed"]), stable_int(task.task_id), a]
        out = run_decode(eng, cache, tr["lp0"], len(ids), [], sampler_chooser(cfg["sampler"], key), budget=budget)
        if out["aborted"]:
            log["attempts"].append({"a": a, "aborted": True})
            break
        code = clean(eng.text(out["gen"]))
        ok = vis(code)
        log["attempts"].append({"a": a, "visible": ok, "gen_len": len(out["gen"])})
        if ok:
            return _result(eng, out["gen"], decision=f"attempt_{a}", log=log)
    return _result(eng, tr["gen"], decision="fallback_greedy", log=log)


def _fork_candidates(eng, task, cfg, vis, choose):
    ids = eng.encode(_messages(task, cfg))
    tr = greedy_trace(eng, ids)
    code0 = clean(eng.text(tr["gen"]))
    log = {"greedy_visible": vis(code0), "visited": []}
    if log["greedy_visible"] is not False:
        return _result(eng, tr["gen"], decision="greedy", log=log)
    budget = _budget(eng, tr, cfg)
    forks = list_forks(eng, tr, cfg["fork_ratio"], cfg["k_alt"])[: cfg["max_forks"]]
    log["eligible_forks"] = len(forks)
    greedy_mean = float(np.mean(tr["seq_lp"]))
    cands = [{"gen": tr["gen"], "mean_lp": greedy_mean, "fork": None}]
    for f in forks:
        for v, lpv in f["alts"]:
            b = branch(eng, tr, f["pos"], v, budget)
            if b["aborted"]:
                log["visited"].append({"pos": f["pos"], "tok": v, "aborted": True})
                return choose(eng, tr, cands, log, done=False)
            lps = tr["seq_lp"][: f["pos"]] + [lpv] + b["seq_lp"]
            c = {"gen": b["gen"], "mean_lp": float(np.mean(lps)),
                 "fork": {"pos": f["pos"], "margin": f["margin"], "greedy_tok": eng.text([f["top"]]),
                          "alt_tok": eng.text([v]), "line": eng.text(tr["gen"][:f["pos"]]).rsplit("\n", 1)[-1]}}
            stop = choose(eng, tr, cands + [c], log, done=None)
            cands.append(c)
            log["visited"].append({**c["fork"], "mean_lp": c["mean_lp"]})
            if stop is not None:
                return stop
    return choose(eng, tr, cands, log, done=True)


def arm_flec(eng, task, cfg, vis):
    def choose(eng, tr, cands, log, done):
        if done is None:  # called after each new candidate: execute it now
            c = cands[-1]
            ok = vis(clean(eng.text(c["gen"])))
            log["visited_visible"] = log.get("visited_visible", []) + [ok]
            if ok:
                return _result(eng, c["gen"], decision="fork", fork=c["fork"], log=log)
            return None
        return _result(eng, tr["gen"], decision="fallback_greedy", log=log)
    return _fork_candidates(eng, task, cfg, vis, choose)


def arm_likelihood(eng, task, cfg, vis):
    def choose(eng, tr, cands, log, done):
        if done is None:
            return None
        best = max(range(len(cands)), key=lambda i: (cands[i]["mean_lp"], -i))
        c = cands[best]
        return _result(eng, c["gen"], decision="greedy" if c["fork"] is None else "fork", fork=c["fork"], log=log)
    return _fork_candidates(eng, task, cfg, vis, choose)


def arm_contrast(eng, task, cfg, vis):
    ids = eng.encode(_messages(task, cfg))
    masked = task.user_content(variant="masked_output", reworded=cfg.get("reworded", False))
    if masked == task.user_content(reworded=cfg.get("reworded", False)):
        out = arm_direct(eng, task, cfg, vis)
        out["decision"] = "no_maskable_evidence"
        return out
    ids_m = eng.encode([{"role": "user", "content": masked}])
    lp, cache = eng.prefill(ids)
    lpm, cache_m = eng.prefill(ids_m, aux=True)
    gen, steered, lam = [], [], float(cfg["contrast_lambda"])
    while True:
        tids, tlps = topk(lp, 4)
        v = int(np.argmax(lp))
        alts = fork_alternatives([v] + [t for t in tids if t != v], [float(lp[v])] + [float(lp[t]) for t in tids if t != v],
                                 cfg["fork_ratio"], cfg["k_alt"])
        if alts:
            in_code, in_comment, in_doc = code_region_state(eng.text(gen))
            if in_code and not in_comment and not in_doc:
                score = lambda t: lp[t] + lam * (lp[t] - lpm[t])
                best = max([v] + [a for a, _ in alts], key=lambda t: (score(t), t == v))
                if best != v:
                    steered.append({"pos": len(gen), "greedy_tok": eng.text([v]), "chosen_tok": eng.text([best])})
                v = best
        gen.append(v)
        reason = eng.terminal(gen)
        if reason:
            return _result(eng, gen, reason=reason, steered=steered)
        lp, cache = eng.step(cache, v)
        lpm, cache_m = eng.step(cache_m, v, aux=True)


ARMS = {"direct": arm_direct, "plan": arm_plan, "sampler": arm_sampler, "best_of_n": arm_best_of_n,
        "likelihood": arm_likelihood, "contrast": arm_contrast, "flec": arm_flec}
