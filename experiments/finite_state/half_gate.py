"""Finite CPU check of normalized-weight-disclosure half-variance lower witness."""
from fractions import Fraction as F
from math import exp
import numpy as np
from scipy.optimize import brentq
def J(c):
    return 2*(c-1+exp(-c))/c**2
def instance(T,a,c,s,gamma):
    Hb=c/(2*T); Lb=a-Hb; QH=s*Hb; QL=s*Lb; B=F(7,10)
    dM=F(1,2)-QL; dP=F(1,2)-QH
    raw=[gamma,2*gamma,gamma,2*gamma,F(1),Lb/Hb]
    q=[2*dM-B,B-dM,2*dP-B,B-dP,QL,QH]; h=[-1,-1,1,1,-1,1]
    Z=sum(x*y for x,y in zip(raw,q)); p=[x*y/Z for x,y in zip(raw,q)]
    assert min(q)>0 and sum(q)==sum(p)==1
    assert not sum(e*x for e,x in zip(h,q)) and not sum(e*x for e,x in zip(h,p))
    return np.array(q,float),np.array(p,float),np.array(h,float),np.array(raw,float),float(Z)
def risk(q,p,h,raw,T):
    K=q[None,:]*np.minimum(1.,raw[None,:]/raw[:,None]); K+=np.diag(1.-K.sum(1))
    # Conditional first/second moment recursion from the full 6-state kernel.
    u=h.copy(); v=h*h
    for _ in range(1,T):
        ku=K@u
        v=h*h+2*h*ku+K@v
        u=h+ku
    return float(q@v/T**2)
root=brentq(lambda c:J(c)-.5,2.,3.); c=F(round(root*10**7),10**7)
print('J root',root,'used',c)
for eta in (F(1,64),F(1,512),F(1,4096)):
    a=2*eta; T=int(32/eta); gamma=F(1,10**12); rows=[]
    for s in (F(19,20),F(21,20)):
        q,p,h,raw,Z=instance(T,a,c,s,gamma)
        R=risk(q,p,h,raw,T);ov=float(np.minimum(p,q).sum())
        rows.append((q,p,R,Z,ov))
    q0,p0,R0,Z0,ov0=rows[0]; q1,p1,R1,Z1,ov1=rows[1]
    print(f'T={T} eta0={float(eta):.7g} R0={R0:.9g} R1={R1:.9g} Z0={Z0:.9g} Z1={Z1:.9g} overlap={min(ov0,ov1):.9g} KLP={p0@np.log(p0/p1):.9g} KLQ={q0@np.log(q0/q1):.9g}')
    assert R0>.5>R1
