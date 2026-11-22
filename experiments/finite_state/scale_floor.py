"""Deterministic three-color unknown-normalization hardness sanity check."""
import math
import numpy as np
from scipy.special import rel_entr

def law(eta,T,s,gamma):
    a=2*eta
    qh=1/(2*T)
    ql=a-qh
    raw=np.array([gamma,1.,ql/qh])
    q=np.array([1-s*a,s*ql,s*qh])
    norm=q@raw
    p=q*raw/norm
    return q,p,raw,norm

def risk(q,p,raw,T):
    A=q[None,:]*np.minimum(1,raw[None,:]/raw[:,None])
    rho=1-A.sum(1); K=A+np.diag(rho)
    nu=q.copy(); out=1/T
    tails=np.zeros(3); g=np.zeros((T,3))
    for l in range(1,T):
        tails=rho*(1+tails)
        g[T-1-l]=tails
    for row in g:
        out+=2*(nu@row)/T**2
        nu=nu@K
    return out
def lim(s):return (1+math.exp(-s))/s-2*(1-math.exp(-s))/s**2
if __name__=='__main__':
    for eta,T in [(1/64,8192),(1/256,32768),(1/1024,131072)]:
        gamma=eta*1e-8;s0,s1=1,1.03
        q,p,h,Z=law(eta,T,s0,gamma);qq,pp,hh,ZZ=law(eta,T,s1,gamma)
        assert np.array_equal(h,hh) and min(np.minimum(q,p).sum(),np.minimum(qq,pp).sum())>eta
        x,y=risk(q,p,h,T),risk(qq,pp,hh,T);dP=rel_entr(p,pp).sum();dQ=rel_entr(q,qq).sum()
        print(eta,T,Z,ZZ,x,y,y-x,lim(s1)-lim(s0), dP,dQ)
