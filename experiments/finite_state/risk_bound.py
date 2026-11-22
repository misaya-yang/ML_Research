# Exact check of the cold-start risk bound R_Q(b,t) <= C0 * Gamma_t, Gamma_t = E_P[min(W,t)]/t (cold_risk_bound.md, Lemma R).
# numpy 2.0 + macOS Accelerate emits spurious FP warnings in matmul; we silence them and assert finiteness instead.
import numpy as np
np.seterr(all='ignore')
rng = np.random.default_rng(3)
def kernel(w, q):
    K = q[None, :]*np.minimum(1, w[None, :]/w[:, None]); K[np.diag_indices(len(w))] += 1-K.sum(1); return K
def cold_risk(K, q, h, mu, b, t):
    f = h - mu; nu = q.copy()
    for _ in range(b): nu = nu @ K                 # law of X_b from a Q start
    g1 = np.zeros_like(f); g2 = np.zeros_like(f)
    for _ in range(t):                             # backward recursion for E[(sum_{i<t} f(X_{b+i}))^2 | X_b]
        Kg1 = K @ g1; g2 = f*f + 2*f*Kg1 + K @ g2; g1 = f + Kg1
    r = (nu @ g2)/t**2
    assert np.isfinite(r) and r >= -1e-12, r
    return r
def instance(n):
    kind = rng.integers(3)
    if kind == 0: w = np.exp(rng.normal(0, rng.uniform(1, 5), n))                 # lognormal weights
    elif kind == 1: w = rng.pareto(rng.uniform(.3, 1.5), n) + rng.uniform(1e-4, 1)  # heavy tail
    else: w = np.r_[np.full(n//2, rng.uniform(1e-4, 1e-2)), rng.uniform(1, 10**rng.uniform(1, 4), n - n//2)]  # dark + heavy
    q = rng.dirichlet(np.full(n, rng.uniform(.3, 3))); q = np.maximum(q, 1e-12); q /= q.sum()
    w = w/(q @ w)                                                                    # normalize E_Q W = 1
    return w, q
def p_median(w, P):                                # d = inf{v : P(W > v) <= 1/2}
    o = np.argsort(w); tail = 1 - np.cumsum(P[o]); return w[o][np.argmax(tail <= .5)]
worst, med_ratio = [], []
for it in range(150):
    n = int(rng.integers(20, 120)); w, q = instance(n); P = q*w; K = kernel(w, q)
    assert np.isfinite(K).all() and abs(P.sum()-1) < 1e-9
    eta = q @ np.minimum(1, w); ws = np.sort(np.unique(w)); d = p_median(w, P)
    for t in [8, 64, 512, 2048]:
        b = int(rng.choice([0, 0, t//4]))
        Gam = (P @ np.minimum(w, t))/t
        hd = (w > d).astype(float)
        hs = [(w >= v).astype(float) for v in ws[::max(1, len(ws)//12)]] + [(w <= v).astype(float) for v in ws[::max(1, len(ws)//12)]]
        hs += [rng.integers(0, 2, n).astype(float) for _ in range(3)] + [rng.random(n) for _ in range(2)] + [hd]
        best = 0.0
        for h in hs:
            if h.min() != h.max(): best = max(best, cold_risk(K, q, h, P @ h, b, t)/Gam)
        worst.append((best, t, b, eta, Gam))
        if hd.min() != hd.max() and d < t/4: med_ratio.append(cold_risk(K, q, hd, P @ hd, b, t)/Gam)
worst.sort(reverse=True)
print('instances x horizons checked:', len(worst), '(all risks finite)')
print('max R/Gamma_t over all h tried: %.3f   (lemma constant C0 = 92)' % worst[0][0])
for r in worst[:5]: print('  ratio=%.3f t=%d b=%d eta=%.2e Gamma_t=%.3e' % r)
print('median of per-case max ratio: %.3f' % np.median([r[0] for r in worst]))
m = np.array(med_ratio)
print('tightness probe h=1{W>d}, cases with d<t/4: n=%d, R/Gamma_t quantiles (5%%,50%%,95%%) = %.3f %.3f %.3f' % ((len(m),) + tuple(np.quantile(m, [.05, .5, .95]))))
