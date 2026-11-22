"""Simultaneously valid in T for a pre-chosen nonconstant binary outcome."""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'curve'))
from risk_profile import filt
def eps(n,delta):
    return np.sqrt(np.log(2/delta)/(2*n)) if n else 1.
def ci_count(b,n,delta):
    return (0. if b==0 else beta.ppf(delta/2,b,n-b+1),1. if b==n else beta.ppf(1-delta/2,b+1,n-b))
def combine(f0,f1,p):
    den=(1.-p)*f0+p*f1
    return np.divide(f0*f1,den,out=np.zeros_like(den),where=den>0)
def bridge(C,D,delta):
    # S=log(un-normalized target / q); estimating log Z, all cut values covered.
    e=eps(len(C),delta/2);v=eps(len(D),delta/2)
    lower=-np.inf;upper=np.inf
    for b in np.quantile(np.r_[C,D],np.linspace(.05,.95,19)):
        u=np.exp(np.minimum(0,D-b)).mean();w=np.exp(np.minimum(0,b-C)).mean()
        if u>v:lower=max(lower,b+np.log(u-v)-np.log(min(1,w+e)))
        if w>e:upper=min(upper,b+np.log(min(1,u+v))-np.log(w-e))
    return lower,upper
def make_band(raw_s,h,delta=.05,z_bound=(0.,0.)):
    h=np.asarray(h,dtype=bool);raw_s=np.asarray(raw_s);n=len(h);k=h.sum()
    s,inv,cnt=np.unique(raw_s,return_inverse=True,return_counts=True);N=len(s)
    a1=np.bincount(inv,weights=h.astype(float),minlength=N);a0=cnt-a1
    if k==0 or k==n:return None
    F0=np.cumsum(a0)/(n-k);F1=np.cumsum(a1)/k
    p0,p1=ci_count(k,n,delta/4)
    e0=eps(n-k,delta/4);e1=eps(k,delta/4)
    f0=np.maximum(0,F0-e0);f1=np.maximum(0,F1-e1)
    l=np.minimum(combine(f0,f1,p0),combine(f0,f1,p1))
    f0=np.minimum(1,F0+e0);f1=np.minimum(1,F1+e1)
    u=np.maximum(combine(f0,f1,p0),combine(f0,f1,p1))
    F=np.cumsum(cnt)/n;psi=np.zeros(N);tail=0.
    for i in range(N-1,-1,-1):
        psi[i]=F[i]+tail
        if i:tail=(tail+cnt[i]/n)*np.exp(s[i-1]-s[i])
    e=eps(n,delta/4)
    with np.errstate(divide='ignore'):
        low_a=np.minimum(0,z_bound[0]+np.log(np.maximum(psi-e,0))-s)
        hi_a=np.minimum(0,z_bound[1]+np.log(np.minimum(1.,psi+e))-s)
    # Conditioning on the success event guarantees l<=u and actual rates between
    # these arrays. Valid increasing/decreasing envelopes tighten estimates.
    l=np.maximum.accumulate(l);u=np.minimum.accumulate(u[::-1])[::-1]
    low_a=np.maximum.accumulate(low_a[::-1])[::-1];hi_a=np.minimum.accumulate(hi_a)
    return (l,u,low_a,hi_a)
def at(band,T):
    if T==1:return 1.,1.
    if band is None:return 1/T,1.
    l,u,la,ua=band;lt=np.log(T)
    # For the low MSE use the left endpoint and the maximal allowed CDF.
    low=np.r_[1/T,filt(ua,lt)]@np.diff(np.r_[0,u,1.])
    high=np.r_[filt(la,lt),1.]@np.diff(np.r_[0,l,1.])
    return max(1./T,low),min(1.,high)
