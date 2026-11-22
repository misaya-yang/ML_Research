# Simulation: expected P labels per virtual-height jump at lag k under a tail profile (small_overlap_joint_frontier.md, Lemma E4).
import numpy as np
rng = np.random.default_rng(2)
def sample_W_P(n, D1, r, eps_low):
    W = D1*rng.random(n)**(-1/r); W[rng.random(n)<eps_low] = 0.5; return W
def tests_at_lag(k, D1, r, eps_low, reps=1500, Jmax=20000):
    N = np.zeros(reps); low = 0
    for t in range(reps):
        G = np.cumsum(rng.exponential(size=min(k, Jmax))); V = k/G; V = V[V >= 1]
        W = sample_W_P(len(V), D1, r, eps_low); ok = np.nonzero(W <= V)[0]
        if len(ok): N[t] = ok[0]+1
        else: N[t] = len(V); low += 1
    return N.mean(), low/reps
for (eta0, r) in [(1/256, .5), (1/256, .25), (1/1024, .5)]:
    D1 = 1/eta0; D_half = D1*2**(1/r); kstar = 4*np.ceil(2*np.log2(2/eta0))*D_half
    print(f'eta0=1/{int(1/eta0)} r={r} D_half={D_half:.0f} k*~{kstar:.2e}')
    for k in [int(D1/4), int(D1), int(D_half), int(4*D_half), int(16*D_half), int(kstar)]:
        m, lowp = tests_at_lag(k, D1, r, eps_low=eta0/2)
        print(f'   lag k={k:>9d}  E[#P tests]={m:7.2f}  P(no eligible high height)={lowp:.3f}')
