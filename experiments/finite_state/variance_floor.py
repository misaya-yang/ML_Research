# Variance-scaled dynamics floor (variance_scaled_floor.md, Theorem V): exact finite-t check.
# 8 states: bulk B (weight 1) and light D (weight 0.1, Q-mass ~1.1rho, h-fraction ~0.1), each split h=0/1; sticky M1 (weight t/c1), M2 (weight t/c2),
# each split h=1/h=0. P(M1)=P(M2)=rho/2 in BOTH worlds, so the P- and Q-laws of W are identical.
# World theta puts h=1 mass a1=theta*a on M1 and a2=(1-theta)*a on M2 (a=rho/2); bulk/light h-fractions (sB,sD)
# solve a 2x2 system so that P(h=1) and Q(h=1) are also identical. Only the signed (W,h) pairing differs.
import numpy as np
np.seterr(all='ignore')
def Phi(c, beta):   # first-order CTMC coefficient of one sticky level
    return 2*(c-1+np.exp(-c))/c**2 - np.exp(-c*beta)*(2/c**2)*(1-(1+c)*np.exp(-c))
def instance(theta, rho, s0, c1, c2, t, wD=0.1, sD0=0.1):
    q1, q2 = rho*c1/(2*t), rho*c2/(2*t); qD = (rho-q1-q2)/(1-wD); qB = 1-q1-q2-qD   # wB=1: tau_P(v)<=rho for v>=1
    wB = 1.0; pB, pD = qB*wB, qD*wD; a = rho/2
    YP = s0*pB + sD0*pD; YQ = s0*qB + sD0*qD - a*(theta-.5)*(c1-c2)/t
    sB, sD = np.linalg.solve([[qB, qD], [pB, pD]], [YQ, YP])
    w = np.array([wB, wB, wD, wD, t/c1, t/c1, t/c2, t/c2])
    q = np.array([(1-sB)*qB, sB*qB, (1-sD)*qD, sD*qD, theta*a/w[4], (rho/2-theta*a)/w[4], (1-theta)*a/w[6], (rho/2-(1-theta)*a)/w[6]])
    h = np.array([0, 1, 0, 1, 1, 0, 1, 0.])
    assert q.min() > 0 and abs(q.sum()-1) < 1e-12 and abs(q @ w - 1) < 1e-12
    return w, q, h
def gate(theta, rho, s0, c1, c2, t, b, k):
    w, q, h = instance(theta, rho, s0, c1, c2, t); P = q*w; n = len(w)
    K = q[None, :]*np.minimum(1, w[None, :]/w[:, None]); K[np.diag_indices(n)] += 1-K.sum(1)
    mu = P @ h; f = h-mu; nu = q.copy()
    for _ in range(b): nu = nu @ K
    g1 = np.zeros(n); g2 = np.zeros(n)
    for _ in range(t):
        Kg1 = K @ g1; g2 = f*f + 2*f*Kg1 + K @ g2; g1 = f + Kg1
    R = (nu @ g2)/t**2; V = mu*(1-mu); Gam = (P @ np.minimum(w, t))/t; eta = q @ np.minimum(1, w)
    return dict(G=R-V/k, R=R, V=V, Gam=Gam, P=P, q=q, w=w, h=h, eta=eta, mu=mu, Qh=q @ h)
def kl(a, b): return float(np.sum(a*np.log(a/b)))
print('Delta (tilt) = 1/4; eps = rho*a*|dPhi|/8... i.e. eps = rho*|Phi1-Phi2|/16.  Quadratic theory: KL_P*rho*dPhi^2/eps^2 ~ 128, KL_Q*rho*t*dPhi^2/eps^2 ~ 64(c1+c2)')
print('case | t rho eps | theta* | G(-),G(+) /eps | R/rho, Phibar/2 | Var/(kR) | Gam/rho | eta | invariants (max diff of W-law P,Q; P(h=1); Q(h=1)) | KL_P*rho*dPhi^2/eps^2  KL_Q*rho*t*dPhi^2/eps^2')
for k, beta, c1, c2 in [(2, 2, 1, 4), (4, 2, 1, 4), (6, 0, 3, 10)]:
    P1, P2 = Phi(c1, beta), Phi(c2, beta); Pbar = (P1+P2)/2; sigma = (k*Pbar-1)/2 - 0.1*0.1/0.9; assert sigma > 0
    for t in [2**10, 2**12, 2**14]:
        for rho in [0.01, 0.05]:
            eps = rho*abs(P1-P2)/16; b = int(beta*t); s0 = sigma*rho; sg = np.sign(P1-P2)
            G = lambda th: gate(th, rho, s0, c1, c2, t, b, k)['G']
            lo, hi = 0.02, 0.98; glo = G(lo); assert glo*G(hi) < 0
            for _ in range(40):
                mid = (lo+hi)/2; gm = G(mid)
                if gm*glo > 0: lo, glo = mid, gm
                else: hi = mid
            th = (lo+hi)/2; D = 4*eps/(rho*abs(P1-P2))
            m, p, z = gate(th-sg*D, rho, s0, c1, c2, t, b, k), gate(th+sg*D, rho, s0, c1, c2, t, b, k), gate(th, rho, s0, c1, c2, t, b, k)
            lawdiff = max(np.abs(np.bincount(np.unique(m['w'], return_inverse=True)[1], m['P']) - np.bincount(np.unique(p['w'], return_inverse=True)[1], p['P'])).max(),
                          np.abs(np.bincount(np.unique(m['w'], return_inverse=True)[1], m['q']) - np.bincount(np.unique(p['w'], return_inverse=True)[1], p['q'])).max())
            print('k=%d b=%dt c=(%g,%g) | %5d %.2f %.5f | %.3f | %+.2f %+.2f | %.3f %.3f | %.3f | %.2f | %.2f | %.0e %.0e %.0e | %.1f  %.1f' % (
                k, beta, c1, c2, t, rho, eps, th, m['G']/eps, p['G']/eps, z['R']/rho, Pbar/2, z['V']/(k*z['R']), z['Gam']/rho, z['eta'],
                lawdiff, abs(m['mu']-p['mu']), abs(m['Qh']-p['Qh']),
                max(kl(p['P'], m['P']), kl(m['P'], p['P']))*rho*(P1-P2)**2/eps**2, max(kl(p['q'], m['q']), kl(m['q'], p['q']))*rho*t*(P1-P2)**2/eps**2))
