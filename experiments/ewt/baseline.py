"""P-only stationary spectral certificate plus maximal cold-to-stationary coupling."""
import sys,os
from pathlib import Path
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.special import softmax
from scipy.stats import beta
HERE=Path(__file__).resolve().parent
ROOT=HERE/'sources'
ARC=HERE
sys.path[:0]=[str(ROOT),str(ARC/'audit'),str(ARC/'curve')]
os.environ.setdefault('EWT_DATA',str(ARC/'data'))
from capture import Model,read,Draw
from runtime_paths import new_run_dir
from pilot import ew
from bands import at,make_band
N=(128,256,1024,4096,8192,16384,29400); T=1024
EVAL_ALPHA=.025
DS=.01; DP=.005; DH=.005; DM=.0025  # DS+DP+DH+2*DM=.025.

def cp(s,n,d):
    return (0. if s==0 else beta.ppf(d/2,s,n-s+1),1. if s==n else beta.ppf(1-d/2,s+1,n-s))

def hazards(x,u): # sorted pilot log W and input log W >=0.
    k=np.searchsorted(x,u,side='right')
    tails=np.r_[np.logaddexp.accumulate(-x[::-1])[::-1],-np.inf]
    return (k+np.exp(np.minimum(np.log(len(x)),u+tails[k])))/len(x)

def moments(a):
    a=np.asarray(a,dtype=np.longdouble);g1=np.ones_like(a);g2=np.ones_like(a)
    k=a>1e-9
    a=a[k]
    with np.errstate(divide='ignore'):
        log=np.log1p(-a);x=np.exp(T*log)
    s0=-np.expm1(T*log)/a
    s1=((1-a)*s0-T*x)/a
    g1[k]=np.clip(s0/T,0,1)
    g2[k]=np.clip((s0+2*s1)/T**2,0,1)
    return g1.astype(float),g2.astype(float)

def ebe_upper(x,alpha):
    n=len(x); ll=np.log(2/alpha)
    return min(1., float(x.mean()+np.sqrt(2*x.var(ddof=1)*ll/n)+7*ll/(3*(n-1))))

def cpvar(lo,hi):
    L=min(lo*(1-lo),hi*(1-hi));m=np.clip(.5,lo,hi);U=m*(1-m);M=max(1-lo,hi)
    return L,U,M

def trans(d1,d2,su,vl,vu,M):
    lip=2*M*d1-d1*d1 if d1<=M else M*M
    u1=(su-.5)*(vl if su<=.5 else vu)+lip
    def f(v):return (su-.5)*v+2*np.sqrt(su*v*d2)+d2
    vals=[f(vl),f(vu)]
    if su<.5 and su*d2>0:
        v=su*d2/(.5-su)**2
        if vl<=v<=vu: vals.append(f(v))
    return u1,max(vals)

def calc(i,doc,ws,gs):
    p=M.law(ws);q=M.law(ws,.72,1.)
    G=np.array([M.ix[u] for u in gs]);j=int(np.argmin(abs(softmax(p.e,axis=1)[np.arange(len(G)),G]-.55)))
    d=Draw(p,q,p,j,G[j]);rng=np.random.default_rng(489772+i);raw,h=d.sample(rng,max(N));w=raw-(p.z-q.z)
    rows=[]
    for n in N:
        # Use all P draws for stationary risks; coupon split for tight correction.
        count=min(1024,n//4)
        x=np.sort(w[:count]);z=w[count:n];hi=z>0
        B=hazards(x,z[hi]);phi=np.maximum(0.,B-np.sqrt(np.log(2/DH)/(2*count)))
        a=np.exp(-z[hi])*phi
        geom1,geom2=moments(a)
        g1=np.zeros(len(z));g2=np.zeros(len(z))
        g1[hi]=(-np.expm1(-z[hi]))*geom1
        g2[hi]=(-np.expm1(-z[hi]))*geom2
        t1=ebe_upper(g1,DM);t2=ebe_upper(g2,DM)
        st1,su=at(make_band(w[:n],h[:n],DS),T)
        ml,mh=cp(h[:n].sum(),n,DP)
        vl,vu,M1=cpvar(ml,mh)
        u1,u2=trans(t1,t2,su,vl,vu,M1)
        rows.append(dict(id=doc,d=i,P=n,T=T,stationary_lower=st1,stationary_norm_U=su,m_lower=ml,m_upper=mh,var_lower=vl,var_upper=vu,
           coupling_d1_hat=g1.mean(),coupling_d2_hat=g2.mean(),
           coupling_d1_U=t1,coupling_d2_U=t2,ub_L1=u1,ub_L2=u2,best=min(u1,u2)))
    return rows
if __name__=='__main__':
    M=Model(read('train'))
    out=[]
    for i,(doc,ws,gs) in enumerate(islice(ew(),6)):out.extend(calc(i,doc,ws,gs))
    df=pd.DataFrame(out);df.to_csv(new_run_dir('ewt-baseline')/'baseline.csv',index=False,mode='x')
    print(df[df.P.eq(max(N))].round(6).to_string(index=False))
    print(df.assign(cert=df.best<0).groupby('P').cert.sum())
