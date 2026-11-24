"""Observational headroom from existing MBPP development traces (no model calls).

Input: arms.jsonl from a finished PowerMH MBPP pilot (16 development rows,
Qwen2.5-1.5B-Instruct). Each row holds the greedy program (attempt 1 of
greedy_then_t1), temperature-1 and q-samples, the PowerMH final program, and
public/all-test verdicts for every distinct program.

Questions, all answered with hidden labels used offline only:
  1. Where can steering change the first answer at all? Items are split into
     greedy correct / greedy fails the visible test / greedy passes the visible
     test but fails a hidden one, and whether the bank holds a correct program.
  2. When greedy is wrong and a correct program exists, is the difference
     local? The two programs are tokenized and compared from their first
     divergent token. "Single-locus" means one contiguous replaced span and
     identical text elsewhere.
  3. How precise is the visible test as a selector inside the bank?

The bank programs were not generated from greedy's prefix, so every divergence
statistic here is observational. Same-prefix evidence needs fork_replay.py.
"""

import argparse
import difflib
import json
from pathlib import Path


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def compare(tok, a, b):
    ta, tb = tok.encode(a), tok.encode(b)
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    ops = [o for o in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes() if o[0] != "equal"]
    first_line = a[: len(tok.decode(ta[:d]))].count("\n")
    return {"divergence_token": d, "greedy_tokens": len(ta), "other_tokens": len(tb),
            "divergence_fraction": round(d / max(1, len(ta)), 3), "edit_regions": len(ops),
            "single_locus": len(ops) == 1, "divergence_line": first_line,
            "greedy_at_divergence": tok.decode(ta[d:d + 6]), "other_at_divergence": tok.decode(tb[d:d + 6])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms_jsonl")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.model_dir)
    recs = [json.loads(l) for l in Path(a.arms_jsonl).read_text().splitlines()]
    items, pub_pass, pub_pass_correct = [], 0, 0
    for r in recs:
        v = r["verdicts"]
        codes = {}
        for arm in r["iid"].values():
            for att in arm:
                codes.setdefault(att["code_hash"], att["code"])
        g = r["iid"]["greedy_then_t1"][0]
        gv = v[g["code_hash"]]
        correct = [h for h in codes if v[h]["all"]]
        for h in codes:
            if v[h]["public"] == "pass":
                pub_pass += 1
                pub_pass_correct += v[h]["all"]
        cat = ("greedy_correct" if gv["all"] else
               "greedy_fails_visible" if gv["public"] != "pass" else "greedy_passes_visible_fails_hidden")
        item = {"task": r["task"], "category": cat, "bank_distinct_programs": len(codes),
                "bank_correct_programs": len(correct), "greedy_tokens": len(tok.encode(g["code"]))}
        if not gv["all"] and correct:
            cmp = sorted((compare(tok, g["code"], codes[h]) | {"code_hash": h} for h in correct),
                         key=lambda c: (c["edit_regions"], -c["divergence_token"]))
            item["nearest_correct"] = cmp[0]
            item["any_single_locus_correct"] = any(c["single_locus"] for c in cmp)
            item["greedy_code"] = g["code"]
            item["nearest_correct_code"] = codes[cmp[0]["code_hash"]]
        items.append(item)
    cats = {c: [i["task"] for i in items if i["category"] == c]
            for c in ("greedy_correct", "greedy_fails_visible", "greedy_passes_visible_fails_hidden")}
    wrong = [i for i in items if i["category"] != "greedy_correct"]
    summary = {
        "source": a.arms_jsonl, "items": len(items), "categories": cats,
        "greedy_wrong_with_correct_in_bank": [i["task"] for i in wrong if i["bank_correct_programs"]],
        "greedy_wrong_single_locus_correct_in_bank": [i["task"] for i in wrong if i.get("any_single_locus_correct")],
        "visible_test_precision_in_bank": {"visible_pass_programs": pub_pass, "of_which_hidden_correct": pub_pass_correct},
        "note": "observational: bank programs were not generated from greedy's prefix",
        "per_item": items,
    }
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_item"}, indent=1))
    for i in wrong:
        if "nearest_correct" in i:
            n = i["nearest_correct"]
            print(f"task {i['task']} {i['category']}: regions={n['edit_regions']} div@{n['divergence_token']}/"
                  f"{n['greedy_tokens']} line {n['divergence_line']} greedy={n['greedy_at_divergence']!r} "
                  f"other={n['other_at_divergence']!r}")


if __name__ == "__main__":
    main()
