"""Thought Anchors (R1-distill 14B/8B, MATH): share of outcome variance resolved
in the first few chunks (sentences). Under the tree-ANOVA identity, any per-step
coupling of k samples can only exploit variance resolved while samples are still
co-located, i.e. in the first ~log k nats of the trace; this bounds that share.
Mixture estimator as in restart_headroom.py: correct path weight M0, failed 1-M0."""
import glob, json, sys, statistics as st
root = sys.argv[1]
def load(model, kind):
    out = {}
    for f in sorted(glob.glob(f"{root}/{model}/*/{kind}/*/chunks_labeled.json")):
        ch = sorted(json.load(open(f)), key=lambda c: c["chunk_idx"])
        cum = 0; pos = []
        for c in ch: pos.append(cum); cum += len(c["chunk"])
        out[f.split("/")[-2]] = ([c["accuracy"] for c in ch], pos, cum)
    return out
for model in ["deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b"]:
    inc, cor = load(model, "incorrect_base_solution"), load(model, "correct_base_solution")
    print(f"\n{model}: {len(inc)} problems; mean base-trace length {st.mean(v[2] for v in inc.values()):.0f} chars")
    for j in [1, 2, 3, 5, 10]:
        r = []
        for n, (ai, _, _) in inc.items():
            if n not in cor: continue
            ac = cor[n][0]; m0 = (ai[0] + ac[0]) / 2
            if not (0 < m0 < 1) or len(ai) <= j or len(ac) <= j: continue
            mi, mc = ai[j], ac[j]
            r.append(1 - (m0 * mc * (1 - mc) + (1 - m0) * mi * (1 - mi)) / (m0 * (1 - m0)))
        print(f"  after first {j:2d} chunk(s): share of outcome variance resolved: median {st.median(r):+.2f}, mean {st.mean(r):+.2f} (n={len(r)}; resample noise ~ +/-0.1)")
