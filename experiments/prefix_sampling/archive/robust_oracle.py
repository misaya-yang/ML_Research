"""Robust localization headroom on failed traces: resume at the last chunk
before the failure is decided (accuracy stays < M0/2 afterwards). Compare with
a fixed resume fraction. Smoothed accuracy (window 5) to limit max-of-noise bias."""
import glob, json, sys, statistics as st
root = sys.argv[1]
for model in ["deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b"]:
    effs, effs_smooth, fracs = [], [], []
    for f in sorted(glob.glob(f"{root}/{model}/*/incorrect_base_solution/*/chunks_labeled.json")):
        ch = sorted(json.load(open(f)), key=lambda c: c["chunk_idx"])
        L = [len(c["chunk"]) for c in ch]; T = sum(L)
        s = [sum(L[:i]) / T for i in range(len(ch))]
        a = [c["accuracy"] for c in ch]
        m0 = a[0]
        if m0 <= 0:
            continue
        sm = [st.mean(a[max(0, i - 2):i + 3]) for i in range(len(a))]
        d = next((i for i in range(len(a)) if all(x < 0.5 * m0 for x in a[i:])), len(a) - 1)
        j = max(d - 1, 0)                     # last chunk before the decided failure
        effs.append((a[j] / (1 - s[j])) / m0)
        effs_smooth.append((sm[j] / (1 - s[j])) / m0)
        fracs.append(s[j])
    print(model, f"n={len(effs)}",
          f"resume-before-decision eff: median={st.median(effs):.2f} mean={st.mean(effs):.2f};",
          f"smoothed median={st.median(effs_smooth):.2f};",
          f"resume position median={st.median(fracs):.2f}")
