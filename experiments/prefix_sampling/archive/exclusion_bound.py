"""Verifier regime: headroom from not re-deriving answers already refuted.

Hypothesis (before running): on hard problems, wrong answers concentrate on a few
attractors, so a large share of i.i.d. tokens before the first verified success
re-derives an already refuted answer. Upper bound: a sampler that skips such
traces at zero cost (= exact sampling from p restricted to non-refuted answers).
Realistic proxy: the repeat is detected after a fraction f of its own tokens.
Monte Carlo over i.i.d. draws (with replacement) from each problem's archive.
Reported: P(first success within B) averaged over problems, and the share of
pre-success tokens spent on repeats of refuted answers.
"""
import glob, json, sys
import numpy as np
root = sys.argv[1]
BUD = [4096, 8192, 16384, 32768, 65536]
FR = [0.0, 0.25, 0.5, 1.0]                  # detection fraction; 1.0 = plain i.i.d.
R = 400                                      # Monte Carlo runs per problem
def num(x):
    try: return int(str(x).strip())
    except (TypeError, ValueError): return None
rng = np.random.default_rng(20261125)
cells = sorted(set(glob.glob(f"{root}/external_aime/_init.jsonl") + glob.glob(f"{root}/external_multi/*/_init.jsonl") + glob.glob(f"{root}/external_ministral_transfer/*/_init.jsonl")))
for f in cells:
    rows = [json.loads(l) for l in open(f)]
    name = f"{rows[0]['model_name']}/{rows[0]['dataset']}"
    res = {fr: np.zeros((len(rows), len(BUD))) for fr in FR}
    waste_share, top_wrong_share = [], []
    for p, r in enumerate(rows):
        G = num(r["gold_answer"])
        A = [str(a[0]).strip() if a[0] is not None else None for a in r["all_answers"]]
        T = np.array([int(a[1]) for a in r["all_answers"]]); Y = np.array([num(a) == G for a in A])
        wrong = [a for a, y in zip(A, Y) if not y]
        if wrong:
            vals, cnt = np.unique([str(w) for w in wrong], return_counts=True)
            top_wrong_share.append(cnt.max() / len(wrong))
        N = len(A); spent_rep = spent_all = 0
        for run in range(R):
            order = rng.integers(0, N, 400)
            for fr in FR:
                refuted = set(); cost = 0; hit = None
                for s in order:
                    a = A[s]
                    if Y[s]:
                        cost += T[s]; hit = cost; break
                    rep = (a is not None) and (a in refuted)
                    c = T[s] * fr if rep else T[s]
                    if fr == 1.0:
                        spent_all += T[s]; spent_rep += T[s] if rep else 0
                    cost += c
                    if a is not None: refuted.add(a)
                    if cost > BUD[-1]: break
                for k, B in enumerate(BUD):
                    res[fr][p, k] += (hit is not None and hit <= B) / R
        if spent_all: waste_share.append(spent_rep / spent_all)
    print(f"\n=== {name}: {len(rows)} problems; median share of wrong samples on the single most common wrong answer: {np.median(top_wrong_share):.2f}")
    print(f"  share of pre-success i.i.d. tokens spent re-deriving an already-refuted answer: median over problems {np.median(waste_share):.2f}, mean {np.mean(waste_share):.2f}")
    print("  B       iid    detect@50%  detect@25%  oracle-skip(0%)")
    for k, B in enumerate(BUD):
        print(f"  {B:6d}  {res[1.0][:,k].mean():.3f}  {res[0.5][:,k].mean():.3f}       {res[0.25][:,k].mean():.3f}       {res[0.0][:,k].mean():.3f}")
