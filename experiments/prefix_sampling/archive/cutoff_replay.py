"""Exact offline replay of restart-with-cutoff policies in the verifier regime.

Hypothesis (stated before running): on long-CoT archives where failed traces are
2-4x longer than correct ones, a single per-model cutoff chosen by leave-one-
problem-out (LOO) raises P(first verified success within B tokens) by >=5 points
over plain i.i.d. retries at some budget where i.i.d. solves 30-80%, in >=3 of 5
AIME25 model cells. Kill: <2 points everywhere -> the temporal lever is small too.

Model of the world: each problem's 64/128 archived i.i.d. samples (answer, tokens)
are the per-problem sampling distribution (draw with replacement). Truncating an
i.i.d. trace at c tokens is an exact simulation of generating only c tokens.
Policy: attempts i=1,2,... with cutoff c_i; attempt costs min(T, c_i, remaining);
succeeds iff correct and T <= min(c_i, remaining); stop at first success.
Verifier = exact-match integer grader (oracle). No model calls.
"""
import glob, json, sys, math
import numpy as np
root = sys.argv[1]
U = 128                                   # token unit
BUD = [4096, 8192, 16384, 32768, 65536]
CUTS = [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, None]
LUBY_U = [512, 1024, 2048, 4096, 8192]
def num(x):
    try: return int(str(x).strip())
    except (TypeError, ValueError): return None
def luby(i):                              # 1-indexed Luby sequence
    k = 1
    while (1 << k) - 1 < i: k += 1
    if i == (1 << k) - 1: return 1 << (k - 1)
    return luby(i - (1 << (k - 1)) + 1)
BMAX = BUD[-1] // U
def value(T, Y, seq):
    """V[b] = P(success within b units) for cutoff sequence seq (units, None=inf)."""
    nat = len(seq)
    V = np.zeros(BMAX + 1)
    b = np.arange(BMAX + 1)
    for i in range(nat - 1, -1, -1):
        c = seq[i] if seq[i] is not None else 10**9
        cap = np.minimum(c, b)[None, :]               # (1,B)
        cost = np.minimum(T[:, None], cap)            # (N,B)
        succ = (Y[:, None]) & (T[:, None] <= cap)
        nxt = V[np.maximum(b[None, :] - cost, 0)]
        nxt[:, 0] = 0
        newV = np.where(succ, 1.0, nxt).mean(0)
        newV[0] = 0
        V = newV
    return V
cells = sorted(set(glob.glob(f"{root}/external_aime/_init.jsonl") + glob.glob(f"{root}/external_multi/*/_init.jsonl") + glob.glob(f"{root}/external_ministral_transfer/*/_init.jsonl")))
for f in cells:
    rows = [json.loads(l) for l in open(f)]
    name = f"{rows[0]['model_name']}/{rows[0]['dataset']}"
    probs = []
    for r in rows:
        G = num(r["gold_answer"])
        T = np.array([max(1, math.ceil(int(a[1]) / U)) for a in r["all_answers"]])
        Y = np.array([num(a[0]) == G for a in r["all_answers"]])
        probs.append((T, Y))
    minT = min(int(T.min()) for T, _ in probs)
    nat = min(400, BMAX // minT + 2)
    pol = {}
    for c in CUTS:
        cu = None if c is None else c // U
        pol[f"cut{c}"] = [value(T, Y, [cu] * nat) for T, Y in probs]
    for u in LUBY_U:
        seq = [u // U * luby(i + 1) for i in range(nat)]
        pol[f"luby{u}"] = [value(T, Y, seq) for T, Y in probs]
    P = len(probs)
    print(f"\n=== {name}: {P} problems, N={len(probs[0][0])}, mean acc {np.mean([Y.mean() for _, Y in probs]):.3f}")
    print("  B      iid(no cut)  LOO-cut  [chosen c, mode]  LOO-luby  oracle-per-problem-cut  in-sample-best-cut")
    for B in BUD:
        bu = B // U
        base = np.array([v[bu] for v in pol["cutNone"]])
        cutkeys = [k for k in pol if k.startswith("cut")]
        lubkeys = [k for k in pol if k.startswith("luby")]
        M = {k: np.array([v[bu] for v in pol[k]]) for k in pol}
        def loo(keys):
            out, ch = [], []
            for p in range(P):
                best = max(keys, key=lambda k: (M[k].sum() - M[k][p], k == "cutNone"))
                out.append(M[best][p]); ch.append(best)
            return np.array(out), max(set(ch), key=ch.count)
        lc, cmode = loo(cutkeys)
        ll, lmode = loo(lubkeys)
        orc = np.max(np.stack([M[k] for k in cutkeys]), 0)
        ins = max(cutkeys, key=lambda k: M[k].mean())
        d = lc - base
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, P, P)].mean() for _ in range(4000)]
        print(f"  {B:6d}  {base.mean():.3f}        {lc.mean():.3f}   [{cmode}]  {ll.mean():.3f} [{lmode}]  {orc.mean():.3f}   {M[ins].mean():.3f} [{ins}]   LOO-cut minus iid {d.mean():+.3f} CI[{np.quantile(bs,.025):+.3f},{np.quantile(bs,.975):+.3f}]")
    # unbounded-budget expected tokens to first success, per problem with acc>0
    ratios = []
    for T, Y in probs:
        if not Y.any(): continue
        e_iid = (T.mean()) / Y.mean()
        best = min((np.minimum(T, c // U).mean() / ((Y & (T <= c // U)).mean())) for c in CUTS[:-1] if (Y & (T <= c // U)).any())
        ratios.append(e_iid / min(best, e_iid))
    print(f"  unbounded budget: oracle per-problem cutoff speedup in E[tokens to first success]: median {np.median(ratios):.2f}x, mean {np.mean(ratios):.2f}x over {len(ratios)} problems")
