"""Error inheritance from archived prefix regenerations (Prefix Consistency archive).

For an originally WRONG sample with answer a, regenerate from the kept prefix
(75/50/25% of its tokens). Compare:
  same_wrong = P(regeneration answers a)      vs  base = P(fresh sample answers a)
  rescue     = P(regeneration correct)        vs  mu   = P(fresh sample correct)
Efficiency (prefix as sunk cost), per problem: (mean rescue / mean regen new tokens) / (mu / mean fresh tokens).
Averaged within problem first, then equally over problems with 0 < mu < 1.
"""
import glob, json, sys
import numpy as np
root = sys.argv[1]
def num(x):
    try: return int(str(x).strip())
    except (TypeError, ValueError): return None
def key(a): return None if a is None else str(a).strip()
jobs = [(f, int(f.split("_rm")[1].split("pct")[0])) for f in sorted(glob.glob(f"{root}/external_aime/_rm*pct_full_x3.jsonl") + glob.glob(f"{root}/external_multi/*/_rm25pct_full_x1.jsonl") + glob.glob(f"{root}/external_ministral_transfer/*/_rm25pct_full_x1.jsonl"))]
print("cell | kept prefix | problems | same-wrong-answer rate vs fresh base rate | rescue vs fresh mu | rescue efficiency, prefix sunk (median over problems) | correct-stays-correct | unanswered-original rescue")
for f, rm in jobs:
    rows = [json.loads(l) for l in open(f)]
    name = f"{rows[0]['model_name']}/{rows[0]['dataset']}"
    S = {k: [] for k in ("same", "base", "rescue", "mu", "eff", "stay", "none_rescue")}
    for r in rows:
        G = num(r["gold_answer"]); O = r["all_answers"]; N = len(O)
        A = [key(o[0]) for o in O]; T = np.array([int(o[1]) for o in O]); Y = np.array([num(a) == G for a in A])
        mu = Y.mean()
        if not (0 < mu < 1): continue
        same, base, resc, eff, stay, nres, rtoks = [], [], [], [], [], [], []
        for i in range(N):
            regs = r["regen_answers"][i]
            if not regs: continue
            ra = [key(x[0]) for x in regs]; rc = [num(x) == G for x in ra]; rt = [int(x[1]) for x in regs]
            others = np.arange(N) != i
            mu_i = Y[others].mean(); Tbar = T[others].mean()
            if Y[i]:
                stay.append(np.mean(rc)); continue
            if A[i] is None:
                nres.append(np.mean(rc)); continue
            same.append(np.mean([x == A[i] for x in ra]))
            base.append(np.mean([A[j] == A[i] for j in range(N) if j != i]))
            resc.append(np.mean(rc))
            rtoks.append(np.mean(rt))
        if same:
            S["same"].append(np.mean(same)); S["base"].append(np.mean(base)); S["rescue"].append(np.mean(resc)); S["mu"].append(mu)
            S["eff"].append((np.mean(resc) / np.mean(rtoks)) / (mu / T.mean()))  # ratio of means, prefix sunk
        if stay: S["stay"].append(np.mean(stay))
        if nres: S["none_rescue"].append(np.mean(nres))
    m = lambda k: np.nanmean(S[k]) if S[k] else float("nan")
    print(f"{name} | {100-rm}% | {len(S['same'])} | {m('same'):.2f} vs {m('base'):.2f} | {m('rescue'):.2f} vs {m('mu'):.2f} | {np.nanmedian(S['eff']):.2f}x | {m('stay'):.2f} | {m('none_rescue'):.2f} (n={len(S['none_rescue'])})")
