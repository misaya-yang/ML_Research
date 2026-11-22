# Exact check: weighted dynamics floor family with the soak level lowered from T^2 to C_s*T (small_overlap_joint_frontier.md, Prop. E3).
import numpy as np
def risk(c, T, kappa, H):
    l, m = 2*kappa, 400*T*kappa
    qM = c/T; qH = (1-l-(m-l)*qM)/(H-l); qL = 1-qM-qH
    v = ((m-l)*qM - .9*l)/((H-l)*qH); u = (.9+qM-qH*v)/qL
    w = np.array([l,l,m,H,H]); hv = np.array([1,-1,-1,1,-1.])
    q = np.array([qL*(1+u)/2, qL*(1-u)/2, qM, qH*(1+v)/2, qH*(1-v)/2])
    assert q.min()>0 and abs(q.sum()-1)<1e-12
    P = q*w; P/=P.sum(); assert abs(P@hv)<1e-9 and abs(P@hv**2-(P@hv)**2-1)<1e-9
    K = q[None,:]*np.minimum(1, w[None,:]/w[:,None]); K[np.diag_indices(5)] += 1-K.sum(1)
    g1 = np.zeros(5); g2 = np.zeros(5)
    for _ in range(T):              # backward recursion for E[S^2 | X_n]
        Kg1 = K@g1; g2 = hv**2 + 2*hv*Kg1 + K@g2; g1 = hv + Kg1
    return (q@g2)/T**2
T, kappa = 2**14, 1/4096
for lab, H in [('T^2', float(T)**2), ('3000T', 3000.*T), ('1000T', 1000.*T), ('300T', 300.*T)]:
    cs = np.array([.9, 1.0, 1.1, 1.2]); R = np.array([risk(c,T,kappa,H) for c in cs])
    slope = (risk(1.05+1e-3,T,kappa,H)-risk(1.05-1e-3,T,kappa,H))/2e-3
    print(f'H={lab:6s} R(.9..1.2)={np.round(R,4)}  dR/dc@1.05={slope:.4f}  soak visit~T*qH={T*(1)/H*T:.2e}' if False else
          f'H={lab:6s} R(.9,1.0,1.1,1.2)={np.round(R,4)}  dR/dc@1.05={slope:.4f}')
for Cs in [30, 10, 3, 1]:
    H = Cs*float(T); R = [risk(c, T, kappa, H) for c in (.9, 1.2)]
    sl = (risk(1.051, T, kappa, H) - risk(1.049, T, kappa, H))/2e-3
    print(f'H={Cs}T    R(.9)={R[0]:.4f} R(1.2)={R[1]:.4f} dR/dc@1.05={sl:.4f}')
