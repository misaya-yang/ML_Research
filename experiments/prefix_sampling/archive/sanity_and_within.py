"""(1) Sanity check of value() on a synthetic case where a cutoff must help.
(2) Within-problem length vs correctness (removes between-problem Simpson effect)."""
import glob, json, math, sys, importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location("cr", "cutoff_replay.py")
src = open("cutoff_replay.py").read().split("cells = sorted")[0]   # defs only
ROOT = sys.argv[1]; ns = {"__name__": "defs", "sys": sys}; sys.argv = ["x", "."]; exec(src, ns)
value = ns["value"]
T = np.array([1] * 50 + [100] * 50); Y = np.array([True] * 50 + [False] * 50)
for seq, lab in [([None] * 300, "no cut"), ([1] * 300, "cut=1 unit")]:
    V = value(T, Y, seq); print("synthetic", lab, "P(success | b=8 units) =", round(V[8], 3), " b=128:", round(V[128], 3))
def num(x):
    try: return int(str(x).strip())
    except (TypeError, ValueError): return None
root = ROOT
for f in sorted(set(glob.glob(f"{root}/external_aime/_init.jsonl") + glob.glob(f"{root}/external_multi/*/_init.jsonl") + glob.glob(f"{root}/external_ministral_transfer/*/_init.jsonl"))):
    rows = [json.loads(l) for l in open(f)]
    aucs, ratio, capfrac = [], [], []
    for r in rows:
        G = num(r["gold_answer"]); T = np.array([int(a[1]) for a in r["all_answers"]]); Y = np.array([num(a[0]) == G for a in r["all_answers"]])
        if 0 < Y.mean() < 1:
            c, w = T[Y], T[~Y]
            aucs.append(((w[:, None] > c[None, :]).mean() + 0.5 * (w[:, None] == c[None, :]).mean()))
            ratio.append(np.median(w) / np.median(c))
        capfrac.append((T >= T.max()).mean() if T.max() >= 12288 else 0)
    print(f"{rows[0]['model_name']}/{rows[0]['dataset']}: within-problem AUC(wrong longer) median {np.median(aucs):.2f}, "
          f"median(wrong len / correct len) within problem {np.median(ratio):.2f} over {len(aucs)} mixed problems")
