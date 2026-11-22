"""Fixed half-variance risk-gate witness. Float finite recursion; no learned model."""
import math, csv, sys
import numpy as np
from scipy.special import rel_entr
from scipy.optimize import brentq

def J(c):
    return 2*(c+math.expm1(-c))/(c*c)

def law(eta,T,s,gamma):
    a=2*eta; root=brentq(lambda v:J(v)-.5,2,3)
    Hbase=root/(2*T); Lbase=a-Hbase
    H=s*Hbase; L=s*Lbase
    dp=.5-H; dm=.5-L; B=.7
    # Two positive/dark raw levels, each with both publicly named signs.
    q=np.array([2*dp-B,2*dm-B,B-dp,B-dm,L,H])
    raw=np.array([gamma,gamma,2*gamma,2*gamma,1.,Lbase/Hbase])
    h=np.array([1.,-1.,1.,-1.,-1.,1.])
    Z=q@raw; p=q*raw/Z
    assert abs(q@h)<1e-14 and abs(p@h)<1e-14 and np.all(q>0)
    assert np.minimum(q,p).sum()>=eta
    return q,p,raw,h,Z,root

def risk(q,p,raw,h,T):
    A=q[None,:]*np.minimum(1,raw[None,:]/raw[:,None])
    K=A+np.diag(1-A.sum(1))
    g=h-p@h; m=np.zeros(len(h)); v=m.copy()
    for _ in range(T):
        tail=K@m
        v=g*g+2*g*tail+K@v
        m=g+tail
    return q@v/T**2

def main():
    rows=[]
    for eta,T in [(1/64,32768),(1/256,131072),(1/1024,524288)]:
        gamma=eta*1e-11
        q0,p0,w,h,Z0,c=law(eta,T,.97,gamma)
        q1,p1,w1,h1,Z1,c=law(eta,T,1.03,gamma)
        assert np.array_equal(w,w1) and np.array_equal(h,h1)
        r0=risk(q0,p0,w,h,T);r1=risk(q1,p1,w,h,T)
        assert r1<.5<r0
        row=dict(eta_floor=eta,T=T,raw_normalizer_unsafe=Z0,raw_normalizer_safe=Z1,
          risk_unsafe=r0,risk_safe=r1,lim_unsafe=J(.97*c),lim_safe=J(1.03*c),
          KL_P=float(rel_entr(p0,p1).sum()),KL_Q=float(rel_entr(q0,q1).sum()))
        rows.append(row); print(row,file=sys.stderr,flush=True)
    wr=csv.DictWriter(sys.stdout,rows[0]);wr.writeheader();wr.writerows(rows)
if __name__=='__main__':main()
