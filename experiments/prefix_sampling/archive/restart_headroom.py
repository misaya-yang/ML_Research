"""Offline headroom check: after a verified failure, how much does resuming
from a prefix of the failed trace beat a fresh restart, per unit of generated
text?  Reads Thought-Anchors chunks_labeled.json (accuracy[i] = fraction of
resamples from just before chunk i that end correct).  No model calls.

Cost proxy: characters still to be generated, taken from the base trace
(remaining chars after the resume point; a fresh restart costs the full trace).
This is a proxy: resampled continuations have their own lengths.
"""
import glob, json, sys, statistics as st

root = sys.argv[1]
FRACS = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9]


def load(model, kind):
    out = []
    for f in sorted(glob.glob(f"{root}/{model}/*/{kind}/*/chunks_labeled.json")):
        ch = json.load(open(f))
        ch = sorted(ch, key=lambda c: c["chunk_idx"])
        lens = [len(c["chunk"]) for c in ch]
        tot = sum(lens)
        start = [sum(lens[:i]) / tot for i in range(len(ch))]
        acc = [c["accuracy"] for c in ch]
        out.append((f.split("/")[-2], start, acc))
    return out


def at_frac(start, acc, f):
    # resume point = first chunk whose start fraction >= f
    for s, a in zip(start, acc):
        if s >= f:
            return s, a
    return start[-1], acc[-1]


for model in ["deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b"]:
    inc = load(model, "incorrect_base_solution")
    cor = load(model, "correct_base_solution")
    if not inc:
        continue
    print(f"\n=== {model}: {len(inc)} failed base traces, {len(cor)} correct base traces")
    print("fixed resume fraction f: mean P(success|prefix of FAILED trace), "
          "mean efficiency vs fresh restart = [M_f/(1-f)]/M_0, #problems with eff>1")
    for f in FRACS:
        ms, effs = [], []
        for name, start, acc in inc:
            m0 = acc[0]
            s, a = at_frac(start, acc, f)
            ms.append(a)
            if m0 > 0:
                effs.append((a / (1 - s)) / m0)
        print(f"  f={f:.2f}  M_f={st.mean(ms):.3f}  median_eff={st.median(effs):.2f}  "
              f"mean_eff={st.mean(effs):.2f}  eff>1: {sum(e > 1 for e in effs)}/{len(effs)}")
    # where is the failure decided along failed traces: first chunk where accuracy
    # drops below half of M_0 and stays below thereafter
    dec = []
    for name, start, acc in inc:
        m0 = acc[0]
        idx = len(acc) - 1
        for i in range(len(acc)):
            if all(a < 0.5 * m0 for a in acc[i:]):
                idx = i
                break
        dec.append(start[idx])
    dec.sort()
    print("  failure-decided position (fraction of trace), sorted:",
          " ".join(f"{d:.2f}" for d in dec))
    print("  M_0 on failed-trace problems:", " ".join(f"{a[0]:.2f}" for _, _, a in inc))
    # oracle (best resume point in hindsight) -- an upper bound, not a policy
    orc = []
    for name, start, acc in inc:
        m0 = acc[0]
        if m0 <= 0:
            continue
        orc.append(max((a / (1 - s)) / m0 for s, a in zip(start, acc) if s < 0.98))
    print(f"  oracle best-resume efficiency (upper bound): median={st.median(orc):.2f}")

    # no-feedback branching: residual outcome variance at depth f for a prefix
    # drawn from the model. Mixture of the correct path (weight M_0) and the
    # failed path (weight 1-M_0) of the same problem, assuming each base path
    # is a typical draw given its outcome.
    byname = {n: (st_, a) for n, st_, a in cor}
    print("  residual outcome variance fraction E[M_f(1-M_f)]/[M_0(1-M_0)] (no feedback):")
    for f in FRACS[1:]:
        r = []
        for name, s_i, a_i in inc:
            if name not in byname:
                continue
            s_c, a_c = byname[name]
            m0 = (a_i[0] + a_c[0]) / 2
            if m0 <= 0 or m0 >= 1:
                continue
            _, mi = at_frac(s_i, a_i, f)
            _, mc = at_frac(s_c, a_c, f)
            r.append((m0 * mc * (1 - mc) + (1 - m0) * mi * (1 - mi)) / (m0 * (1 - m0)))
        print(f"    f={f:.2f}: median={st.median(r):.2f} mean={st.mean(r):.2f} (n={len(r)})")
