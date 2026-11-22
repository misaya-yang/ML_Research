# Checks for cold_risk_bound.md: Lemma D (cold-law domination) and Conjecture R' (beyond-window discount).
#   Lemma D : for w(x) >= d,  (Q K^i)(x) <= 3 min(1, (i+1)/w(x)) P(x)
#   R'      : R_Q(b,t) <= C * Xi,  Xi = E_P min(W/t, 1, L/W) + tau_P(L)^2 + 1/t,  L = b+t
import numpy as np
np.seterr(all='ignore')
rng = np.random.default_rng(11)
def kernel(w, q):
    K = q[None, :]*np.minimum(1, w[None, :]/w[:, None]); K[np.diag_indices(len(w))] += 1-K.sum(1); return K
def cold_risk(K, q, h, mu, b, t):
    f = h - mu; nu = q.copy()
    for _ in range(b): nu = nu @ K
    g1 = np.zeros_like(f); g2 = np.zeros_like(f)
    for _ in range(t):
        Kg1 = K @ g1; g2 = f*f + 2*f*Kg1 + K @ g2; g1 = f + Kg1
    r = (nu @ g2)/t**2; assert np.isfinite(r) and r >= -1e-12; return r
def instance(n):
    kind = rng.integers(4)
    if kind == 0: w = np.exp(rng.normal(0, rng.uniform(1, 5), n))
    elif kind == 1: w = rng.pareto(rng.uniform(.3, 1.5), n) + rng.uniform(1e-4, 1)
    elif kind == 2: w = np.r_[np.full(n//2, rng.uniform(1e-4, 1e-2)), rng.uniform(1, 10**rng.uniform(1, 4), n - n//2)]
    else:  # far tail: light bulk plus a few far atoms carrying P-mass theta at weight ~M
        m = int(rng.integers(1, 4)); w = np.r_[rng.uniform(.2, 3, n - m), 10**rng.uniform(2, 7, m)]
    q = rng.dirichlet(np.full(n, rng.uniform(.3, 3)))
    if kind == 3:  # set the far atoms' P-mass to theta in [.01,.6]
        theta = rng.uniform(.01, .6); m = int((w > 50).sum()); P = q*w
        P[w > 50] *= theta/P[w > 50].sum(); P[w <= 50] *= (1-theta)/P[w <= 50].sum(); q = P/w; q /= q.sum()
    q = np.maximum(q, 1e-14); q /= q.sum(); w = w/(q @ w); return w, q
def p_median(w, P):
    o = np.argsort(w); tail = 1 - np.cumsum(P[o]); return w[o][np.argmax(tail <= .5)]
dom_worst = 0.0; rows = []
for it in range(200):
    n = int(rng.integers(20, 100)); w, q = instance(n); P = q*w; K = kernel(w, q); d = p_median(w, P)
    heavy = w >= d
    nu = q.copy()
    for i in range(513):                                   # Lemma D along the path
        bound = 3*np.minimum(1, (i+1)/w[heavy])*P[heavy]
        dom_worst = max(dom_worst, np.max(nu[heavy]/bound)); nu = nu @ K
    ws = np.sort(np.unique(w))
    for t in [16, 128, 1024]:
        b = int(rng.choice([0, t//2, 4*t])); L = b + t
        Gam = (P @ np.minimum(w, t))/t
        Xi = P @ np.minimum.reduce([w/t, np.ones_like(w), L/w]) + (P @ (w > L))**2 + 1/t
        hs = [(w >= v).astype(float) for v in ws[::max(1, len(ws)//10)]] + [rng.random(n) for _ in range(2)]
        R = max(cold_risk(K, q, h, P @ h, b, t) for h in hs if h.min() != h.max())
        rows.append((R/Gam, R/Xi, Gam/Xi, t, b))
r = np.array(rows)
print('Lemma D: max over instances, i<=512, w>=d of QK^i / [3 min(1,(i+1)/w) P] = %.3f  (must be <= 1)' % dom_worst)
print('cases: %d' % len(r))
print("Lemma R : max R/Gamma_t = %.3f" % r[:, 0].max())
print("Conj R' : max R/Xi      = %.3f   median %.3f" % (r[:, 1].max(), np.median(r[:, 1])))
print("looseness of Gamma_t vs Xi: max Gamma_t/Xi = %.1f, cases with Gamma_t/Xi > 10: %d" % (r[:, 2].max(), (r[:, 2] > 10).sum()))
