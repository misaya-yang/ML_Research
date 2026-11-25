"""Split originally-wrong samples into cap-truncated (T >= cell's max length) vs
answered-wrong, and compare same-answer inheritance and rescue (75% prefix kept)."""
import glob, json, sys
import numpy as np
root = sys.argv[1]
def num(x):
    try: return int(str(x).strip())
    except (TypeError, ValueError): return None
for f in sorted(glob.glob(f"{root}/external_multi/*/_rm25pct_full_x1.jsonl") + glob.glob(f"{root}/external_ministral_transfer/*/_rm25pct_full_x1.jsonl") + glob.glob(f"{root}/external_aime/_rm25pct_full_x3.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    cap = max(int(o[1]) for r in rows for o in r["all_answers"])
    out = {"trunc": [[], [], []], "answered": [[], [], []]}   # rescue, same, eff-pieces
    ntr = nwr = 0
    for r in rows:
        G = num(r["gold_answer"]); O = r["all_answers"]; Y = np.array([num(o[0]) == G for o in O]); T = np.array([int(o[1]) for o in O])
        mu = Y.mean()
        if not (0 < mu < 1): continue
        for i, o in enumerate(O):
            if Y[i] or not r["regen_answers"][i]: continue
            kind = "trunc" if int(o[1]) >= cap else "answered"
            ntr += kind == "trunc"; nwr += kind == "answered"
            regs = r["regen_answers"][i]
            out[kind][0].append(np.mean([num(x[0]) == G for x in regs]) / mu)
            out[kind][1].append(np.mean([str(x[0]).strip() == str(o[0]).strip() for x in regs]))
            out[kind][2].append((np.mean([num(x[0]) == G for x in regs]) / np.mean([int(x[1]) for x in regs])) / (mu / T.mean()))
    name = f"{rows[0]['model_name']}/{rows[0]['dataset']}"
    print(f"{name}: cap={cap}, failed originals: {ntr} at cap, {nwr} answered-wrong")
    for k in out:
        if out[k][0]:
            print(f"   {k:9s}: rescue/mu mean {np.mean(out[k][0]):.2f}, same-answer rate {np.mean(out[k][1]):.2f}, rescue efficiency (prefix sunk, mean) {np.mean(out[k][2]):.2f}x")
