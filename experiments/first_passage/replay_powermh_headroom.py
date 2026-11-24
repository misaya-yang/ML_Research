"""Replay complete PowerMH logs and measure the headroom of exact early rejection.

Input: a run directory written by powermh_lazy_pilot.py (config.json, raw.jsonl).
The chain is rebuilt from the logs alone: initial tokens and per-position log Z,
then for each MH step the cut c, log U, proposal tokens and per-position log Z.
For every proposal the replay recomputes the accept decision, checks it against
the logged one, and for every rejection finds the earliest decode position at
which the rejection was already certain:

    first j with  sum_{r=1..j} log Z(h'_{c+r})  <  log U + sum_{s>c} log Z(h_s)
    (cut_mode "horizon": fixed horizon T, cut uniform on {0..T-1}), or
    first j with  sum_{r=1..j} log Z(h'_{c+r}) - Phi + log(n/(c+j+1)) < log U
    (cut_mode "response": cut uniform inside the current response of length n;
    the exact MH ratio carries n/n', and n' >= c+j+1 after j decode forwards).

Decode positions after that point are the headroom. They are counted two ways:
  raw    every decode forward the eager engine ran;
  cached only forwards at prefixes not already read earlier in the same run
         (an engine with exact prefix caching would get the others free).
Eager and lazy keep separate caches, since lazy reads fewer prefixes.
Prefill of the kept prefix is the same for both arms and reported separately.
Seconds are apportioned per decode position within each proposal (best case).
"""

import argparse
import json
import math
import os
import random


def quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    if not xs:
        return None
    ys = sorted(xs)
    return [ys[min(len(ys) - 1, int(round(q * (len(ys) - 1))))] for q in qs]


def replay_problem(rec, T, mode="horizon"):
    cur = list(rec["init"]["tokens"])
    cur_logz = list(rec["init"]["logz"]) + [0.0] * (T - len(rec["init"]["logz"]))
    root = ()
    eager_cache = {root} | {tuple(cur[:j]) for j in range(1, len(cur))}
    lazy_cache = set(eager_cache)
    out = {"problem": rec["problem"], "props": 0, "acc": 0, "rej": 0, "mismatch": 0,
           "eager_raw": 0, "lazy_raw": 0, "eager_cached": 0, "lazy_cached": 0,
           "prefill": 0, "kept_cached": 0, "sec_eager": 0.0, "sec_saved": 0.0, "rej_frac": [],
           "first_decode_rejections": 0, "numerics_c": 0.0}
    for st in rec["steps"]:
        if st.get("pad"):
            continue
        c, logu, y, lz = st["c"], st["logu"], st["tokens"], st["logz"]
        out["props"] += 1
        out["prefill"] += st.get("prefill", 0)
        out["kept_cached"] += st.get("kept_cached", 0)
        out["numerics_c"] = max(out["numerics_c"], abs(lz[0] - cur_logz[c]))
        phi = sum(cur_logz[c + 1:])
        tail = lz[1:]
        n = len(cur)
        if mode == "response" and st.get("n", n) != n:
            out["mismatch"] += 1
        partial, stop = 0.0, None
        for j, z in enumerate(tail, start=1):
            partial += z
            bound = partial - phi + (math.log(n / (c + j + 1)) if mode == "response" else 0.0)
            if stop is None and bound < logu:
                stop = j
        n2 = c + len(y)
        acc = logu < sum(tail) - phi + (math.log(n / n2) if mode == "response" else 0.0)
        if acc != st["acc"] or acc != (stop is None):
            out["mismatch"] += 1
        eager_n = len(tail)
        lazy_n = eager_n if stop is None else stop
        base = tuple(cur[:c])
        for j in range(1, eager_n + 1):
            key = base + tuple(y[:j])
            if key not in eager_cache:
                eager_cache.add(key)
                out["eager_cached"] += 1
            if j <= lazy_n and key not in lazy_cache:
                lazy_cache.add(key)
                out["lazy_cached"] += 1
        out["eager_raw"] += eager_n
        out["lazy_raw"] += lazy_n
        sec = st.get("sec")
        if sec is not None and lz:
            per = sec / len(lz)
            out["sec_eager"] += per * eager_n
            out["sec_saved"] += per * (eager_n - lazy_n)
        if acc:
            out["acc"] += 1
            cur = cur[:c] + list(y)
            cur_logz = cur_logz[:c] + list(lz) + [0.0] * max(0, T - c - len(lz))
        else:
            out["rej"] += 1
            if eager_n:
                out["rej_frac"].append(lazy_n / eager_n)
                out["first_decode_rejections"] += lazy_n == 1
    out["final_len_ok"] = len(cur) == rec.get("final_len", len(cur))
    return out


def pooled(rows, a, b):
    sa, sb = sum(r[a] for r in rows), sum(r[b] for r in rows)
    return 1 - sb / sa if sa else None


def bootstrap(rows, a, b, reps=10000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        v = pooled(sample, a, b)
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    T = cfg.get("T", cfg.get("cap"))
    mode = cfg.get("cut_mode", "horizon")
    log_name = cfg.get("log_name", "raw.jsonl")
    recs = [json.loads(l) for l in open(os.path.join(args.run_dir, log_name))]
    rows = [replay_problem(r, T, mode) for r in recs]
    rej_frac = [x for r in rows for x in r["rej_frac"]]
    props = sum(r["props"] for r in rows)
    summary = {
        "run": os.path.basename(os.path.normpath(args.run_dir)),
        "alpha": cfg.get("alpha"), "T": T, "K": cfg.get("K"), "cut_mode": mode, "problems": len(rows),
        "proposals": props, "accepted": sum(r["acc"] for r in rows),
        "rejected": sum(r["rej"] for r in rows),
        "decision_mismatches": sum(r["mismatch"] for r in rows),
        "final_state_rebuilt": all(r["final_len_ok"] for r in rows),
        "decode_positions": {
            "eager_raw": sum(r["eager_raw"] for r in rows),
            "lazy_raw": sum(r["lazy_raw"] for r in rows),
            "eager_cached": sum(r["eager_cached"] for r in rows),
            "lazy_cached": sum(r["lazy_cached"] for r in rows),
            "kept_prefix_prefill_both_arms": sum(r["prefill"] for r in rows),
            "kept_prefix_positions_reused_from_kv_cache": sum(r["kept_cached"] for r in rows),
        },
        "headroom_raw": pooled(rows, "eager_raw", "lazy_raw"),
        "headroom_cached": pooled(rows, "eager_cached", "lazy_cached"),
        "headroom_cached_problem_bootstrap95": bootstrap(rows, "eager_cached", "lazy_cached"),
        "per_problem_headroom_cached_q0_25_50_75_100": quantiles(
            [1 - r["lazy_cached"] / r["eager_cached"] for r in rows if r["eager_cached"]]),
        "rejection_decided_fraction_q0_25_50_75_100": quantiles(rej_frac),
        "rejections_decided_at_first_decode": sum(r["first_decode_rejections"] for r in rows),
        "decode_seconds_eager_best_case_saved": [sum(r["sec_eager"] for r in rows),
                                                 sum(r["sec_saved"] for r in rows)],
        "accuracy_low_temperature_init": sum(bool(rec.get("init_correct")) for rec in recs),
        "accuracy_powermh_final": sum(bool(rec.get("final_correct")) for rec in recs),
        "numerics_max_abs_logz_at_cut": max((r["numerics_c"] for r in rows), default=0.0),
        "note": "development rows; headroom before batching, prefetch or scheduler overhead",
    }
    with open(os.path.join(args.run_dir, "replay_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
