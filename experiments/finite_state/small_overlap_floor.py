# Small-overlap witness (variance_scaled_floor.md, Theorem S and Proposition N): exact finite-t check.
# One W-law, five levels: dark (weight 1/d^2), startup U (weight 1, Q-mass u/d), bulk (weight d, carries the overlap),
# sticky M1, M2 (weights t/c1, t/c2, P-mass rho/2 each). Overlap eta ~ (1+u)/d, tau_P(v) <= rho for v >= d.
# Labels are Bernoulli per level; given the (label-free) level chain, R is EXACTLY
#   R(th) = th' C th + sum_l th_l (1-th_l) S_l,   C_ll' = E[(O_l-p_l)(O_l'-p_l')],  S_l = E sum_{segments in l} o_g^2.
# Two tilts on the SAME W-law: S (startup label th_U) and V (sticky split theta, as in Theorem V).
# Compensating dark/bulk labels keep P(h=1), Q(h=1) (hence mu, Var) exactly invariant.
import numpy as np
np.seterr(all='ignore')
def Phi0(c): return (2/c)*(1-(2-(2+c)*np.exp(-c))/c)
def phi0(c): return 1-(1-np.exp(-c))/c
def base(t, d, rho, cs=(3., 10.), u=1.0):
    wD = 1/d**2; w = np.array([wD, 1.0, d] + [t/c for c in cs])
    qM = [rho/2*c/t for c in cs]; qU = u/d
    qD, qB = np.linalg.solve([[1, 1], [wD, d]], [1-qU-sum(qM), 1-qU-rho])
    q = np.array([qD, qU, qB] + qM); assert q.min() > 0 and abs(q.sum()-1) < 1e-12 and abs(q @ w-1) < 1e-12
    return w, q
def kernel(w, q):
    K = q[None, :]*np.minimum(1, w[None, :]/w[:, None]); np.fill_diagonal(K, 0); np.fill_diagonal(K, 1-K.sum(1)); return K
def moments(w, q, t, b=0):
    K = kernel(w, q); n = len(w); E = np.eye(n); r = np.diag(K)-q          # r = same-name stay probability
    G1 = np.zeros((n, n)); G2 = np.zeros((n, n, n))
    for _ in range(t):                                                      # backward: E_x[(sum e_l)(sum e_l')]
        KG1 = K @ G1; KG2 = np.einsum('xy,ylm->xlm', K, G2)
        G2 = E[:, :, None]*E[:, None, :] + E[:, :, None]*KG1[:, None, :] + KG1[:, :, None]*E[:, None, :] + KG2; G1 = E + KG1
    nu = q.copy(); pis = []
    for _ in range(b): nu = nu @ K
    for _ in range(t): pis.append(nu); nu = nu @ K
    nu0 = pis[0]; M1 = nu0 @ G1/t; M2 = np.einsum('x,xlm->lm', nu0, G2)/t**2
    pis = np.array(pis); S = np.zeros(n)
    for l in range(n):                                                      # S_l = (1/t^2) sum_i pi_i(l)[1+2 sum_{m=1}^{t-1-i} r^m]
        rl = r[l]; m = np.arange(t)[::-1]; geo = np.where(rl < 1, rl*(1-rl**m)/(1-rl), m)
        S[l] = (pis[:, l] @ (1+2*geo))/t**2
    p = q*w; C = M2 - np.outer(p, M1) - np.outer(M1, p) + np.outer(p, p)
    return dict(p=p, q=q, w=w, C=C, S=S, M1=M1)
def R(m, th): return th @ m['C'] @ th + np.sum(th*(1-th)*m['S'])
def gate(m, th, k): mu = m['p'] @ th; return R(m, th) - mu*(1-mu)/k
def calibrate(m, th, k, j=2):                       # choose th[j] (bulk label) so that G=0: G is quadratic in th[j]
    xs = np.array([0., .5, 1.]); ys = []
    for x in xs: tt = th.copy(); tt[j] = x; ys.append(gate(m, tt, k))
    roots = [z.real for z in np.roots(np.polyfit(xs, ys, 2)) if abs(z.imag) < 1e-12 and 0 < z.real < 1]
    if not roots: return None
    tt = th.copy(); tt[j] = min(roots); return tt
def compensate(m, th, ref, jd=0, jb=2):              # solve dark/bulk labels so P(h=1), Q(h=1) equal ref's
    p, q = m['p'], m['q']; tt = th.copy(); tt[[jd, jb]] = 0
    rhs = [p @ ref - p @ tt, q @ ref - q @ tt]; tt[[jd, jb]] = np.linalg.solve([[p[jd], p[jb]], [q[jd], q[jb]]], rhs); return tt
def kl(m, a, b, which):
    mass = m[which]; A = np.r_[mass*a, mass*(1-a)]; B = np.r_[mass*b, mass*(1-b)]; keep = (A > 0) | (B > 0)
    return max(np.sum(A[keep]*np.log(A[keep]/B[keep])), np.sum(B[keep]*np.log(B[keep]/A[keep])))
def tilt(m, ref, j, D, k):
    lo, hi = ref.copy(), ref.copy(); lo[j] -= D; hi[j] += D
    if j >= 3: lo[4], hi[4] = 1-lo[3], 1-hi[3]                              # V-tilt moves theta on M1 and 1-theta on M2
    lo, hi = compensate(m, lo, ref), compensate(m, hi, ref)
    for v in (lo, hi): assert v.min() >= 0 and v.max() <= 1 and 0 < v[0] < 1 and 0 < v[2] < 1, v   # compensators strictly inside
    return gate(m, lo, k), gate(m, hi, k), kl(m, lo, hi, 'p'), kl(m, lo, hi, 'q')

if __name__ == '__main__':
    # 0) cross-check the quadratic-form risk against an explicit cell chain (each level split into h=1/h=0 cells)
    t, d, rho = 2**11, 64., 0.1; m = moments(*base(t, d, rho), t); th = np.array([.3, .6, .2, .9, .4])
    w2 = np.repeat(m['w'], 2); q2 = np.ravel(np.c_[m['q']*th, m['q']*(1-th)]); h2 = np.tile([1., 0.], 5)
    K2 = kernel(w2, q2); mu = (q2*w2) @ h2; f = h2-mu; g1 = np.zeros(10); g2 = np.zeros(10)
    for _ in range(t): Kg1 = K2 @ g1; g2 = f*f + 2*f*Kg1 + K2 @ g2; g1 = f + Kg1
    print('cross-check: quadratic-form R = %.15f, cell-chain R = %.15f' % (R(m, th), (q2 @ g2)/t**2))
    k, cs = 6, (3., 10.); A1 = sum(.5*(1-phi0(c)) for c in cs); dPhi = Phi0(cs[0])-Phi0(cs[1]); Pbar = (Phi0(cs[0])+Phi0(cs[1]))/2
    print('k=%d b=0 c=%s u=1 wU=1: A1=%.4f  Phi0=(%.4f,%.4f).  Tilt Delta=1/4; eps := min|G(+-)| (honest margin).' % (k, cs, A1, Phi0(cs[0]), Phi0(cs[1])))
    print('S slope prediction a_U = rhoB*(betaB*rhoB-(1-sB)*A1*rho) + 2*rho*Pbar*(1-sB)/d,  betaB = thD/2+thU+1/2-5sB/2')
    print('   t     rho  rhoB |     d      eta    Gamma | S: a/pred  G-/eps G+/eps  KL_P*d/D^2 KL_Q*d/D^2  KL_P/eps^2 | V: G-/eps G+/eps  KL_P/eps^2 | K=gain  | inv(mu,Qh) | CS-ratio  excess/e_low')
    for t in [2**14, 2**16, 2**18]:
        for rho in [0.05, 0.1]:
            for fr in [1/16, 0.5, 1.0]:
                rB = fr*rho; d = rB*t; w, q = base(t, d, rho, cs); m = moments(w, q, t)
                eta = q @ np.minimum(1, w); Gam = m['p'] @ np.minimum(w, t)/t
                refS = calibrate(m, np.array([.5, .5, 0., 1., 1.]), k); refV = calibrate(m, np.array([.5, .5, 0., .5, .5]), k)
                if refS is None or refV is None: print('%7d %.2f %.4f | %7.0f  n/a: no bulk label gives G=0 (calibration needs d*rho>>1 and 2k*rhoB<1)' % (t, rho, rB, d)); continue
                sB = refS[2]
                gm, gp, kp, kq = tilt(m, refS, 1, .25, k); a = (gp-gm)/.5; eps = min(abs(gm), abs(gp))
                pred = rB*((.25+.5+.5-2.5*sB)*rB-(1-sB)*A1*rho) + 2*rho*Pbar*(1-sB)/d
                lo = refS.copy(); lo[1] -= .25; lo = compensate(m, lo, refS); inv = max(abs(m['p'] @ (lo-refS)), abs(m['q'] @ (lo-refS)))
                vm, vp, vkp, vkq = tilt(m, refV, 3, .25, k); epsV = min(abs(vm), abs(vp))
                # Proposition N ingredients: (i) |R1-R0| <= 18.17 sqrt(Gamma * sum_l D_l^2 E O_l) on random compensated label pairs
                rng = np.random.default_rng(int(t+1000*rho+fr*10)); worst = 0.0
                for _ in range(200):
                    x0 = rng.uniform(.05, .95, 5); x1 = x0 + rng.normal(0, .1, 5); x1[[0, 2]] = x0[[0, 2]]; x1 = compensate(m, np.clip(x1, .01, .99), x0)
                    if x1.min() <= 0 or x1.max() >= 1: continue
                    D = x1-x0; worst = max(worst, abs(R(m, x1)-R(m, x0))/(18.17*np.sqrt(Gam*np.sum(D**2*m['M1']))))
                # (ii) excess occupation of levels below d is at most e_low = (1/t) sum_i lambda(d)^i
                lam_d = 1 - q @ np.minimum(1, w/d); e_low = (1-lam_d**t)/(t*(1-lam_d)); low = w < d
                exc = np.sum(np.maximum(m['M1'][low]-2*m['p'][low], 0))
                print('%7d %.2f %.4f | %7.0f %.2e %.4f | %5.3f  %+.2f %+.2f  %6.2f %6.2f  %9.0f | %+.2f %+.2f  %8.0f | %7.1f | %.0e | %.3f  %.3f' % (
                    t, rho, rB, d, eta, Gam, a/pred, gm/eps, gp/eps, kp*d/.0625, kq*d/.0625, kp/eps**2,
                    vm/epsV, vp/epsV, vkp/epsV**2, (vkp/epsV**2)/(kp/eps**2), inv, worst, exc/e_low))
