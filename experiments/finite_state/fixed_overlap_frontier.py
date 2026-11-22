"""Core deterministic CPU checks for the fixed-overlap cold IMH frontier.

Dependencies: Python 3.9+ with NumPy. Run from the repository root:
    python3 experiments/finite_state/fixed_overlap_frontier.py

These are numerical checks of finite-state recurrences and moment bounds,
not formal verification of the probabilistic proof or end-to-end tests.
"""
import numpy as np

def threshold_pairs():

    def panel(P,Q,h,T,c):
        w=P/Q; f=h-P@h; v=P@(f**2)
        eta=np.minimum(P,Q).sum(); ac=P@(1/np.maximum(w,c)); mc=c*ac
        K=Q[None,:]*np.minimum(1,w[None,:]/w[:,None]); K+=np.diag(1-K.sum(1))
        breaks=np.r_[c,np.unique(w[w>c]),np.inf]
        rt=1-np.array([P@(1/np.maximum(w,z)) for z in breaks])
        Ms=[]; Ps=[]
        for z in breaks[:-1]:
            low=w<=z; p=P*low; p/=p.sum(); Ps.append(p)
            Ms.append(np.where(low[:,None],p[None,:],np.eye(len(P))))
        powers=[np.eye(len(P))]
        for j in range(1,T): powers.append(powers[-1]@K)
        true=short=trunc=e1=e2=e10=e11=e1d=0.
        for i in range(T):
            bi=Q@powers[i]
            mass=np.diff(rt**(i+1)); ai=mass@np.array(Ps)
            true+=bi@(f*f); short+=ai@(f*f); trunc+=ai@(f*f); e1+=(bi-ai)@(f*f); e1d+=(bi-ai)@(f*f)
            for lag in range(1,T-i):
                true+=2*(bi*f)@powers[lag]@f
                trunc+=2*(ai*f)@powers[lag]@f
                am=np.einsum('i,ijk->jk',np.diff(rt**lag),Ms)
                short+=2*(ai*f)@am@f
                e1+=2*((bi-ai)*f)@powers[lag]@f
                e2+=2*(ai*f)@(powers[lag]-am)@f
                e10+=2*((bi-ai)*f)@am@f
                e11+=2*((bi-ai)*f)@(powers[lag]-am)@f
        assert abs((true-short)-(e1+e2)) <= 1e-10
        assert abs(e1-e10-e11-e1d) <= 1e-10
        T2=T*T; true/=T2; short/=T2; trunc/=T2
        main=abs(np.sqrt(max(true,0))-np.sqrt(max(trunc,0)))
        extra=abs(trunc-short)
        assert main<=np.sqrt((1-ac)*(2-ac))/(ac*T)+1e-14
        assert extra<=2*v/(ac*mc*T)+1e-14
        return eta,ac,mc,true,short,main,extra
    rng=np.random.default_rng(14)
    for r in range(500):
        P,Q=rng.dirichlet(np.ones(4),size=2); h=rng.uniform(size=4); T=rng.integers(3,60)
        eta=np.minimum(P,Q).sum(); s=np.argsort(P/Q); u=np.searchsorted(np.cumsum(P[s]),eta/4)
        c=max(1,(P/Q)[s[u]].item())
        panel(P,Q,h,T,c)
    print('500 direct exact-matrix quadrature instances validated; two decomposition inequalities and exact four-event disjoint complement.')
    print('rare spike 4-state',panel(np.array([.03,.3,.5,.17]),np.array([.5,.4,.098,.002]),np.array([0,.5,.5,1.]),96,1.))


def correction_variances():
    """Finite-state quadrature of hybrid IPW variance, independent of sampled paths."""
    from math import sqrt

    def run(T,p,q,h,eta0,c):
        w=p/q;f=h-p@h;v=p@(f*f)
        a=p@(1/np.maximum(c,w));r=1-a;r0=sqrt(1-eta0/2)
        b=np.unique(np.r_[0,w,c,np.inf])
        rb=1-q@np.minimum(1,w[:,None]/np.maximum(b[None,:],1e-300))
        z=np.arange(T+1)[:,None]; D=np.diff(rb[None,:]**z,axis=1)
        raw=[];px=[]
        for s in b[:-1]:
            J=w<=s; pj=p*J;pj=pj/pj.sum() if pj.sum()>0 else pj
            px.append(pj);raw.append(np.where(J[:,None],pj[None,:],np.eye(len(p))))
        px=np.array(px);raw=np.array(raw)
        low=b[:-1]<c;high=~low
        yL=D[1:,low]@px[low];yH=D[1:,high]@px[high]
        kL=np.einsum('kn,nij->kij',D[:,low],raw[low]); kH=np.einsum('kn,nij->kij',D[:,high],raw[high])
        states=np.arange(T); pi=(1-r0)*r0**states
        e=[0.]*4;s=[0.]*4
        a2=f*f
        for i in range(T):
            lag=np.arange(1,T-i)
            e[0]+=2*((yL[i]*f)@(kH[lag]@f).T).sum()/T**2
            s[0]+=4*((yL[i]*a2)@(kH[lag]@a2).T).sum()/T**3/pi[i]
            e[1]+=2*((yH[i]*f)@(kL[lag]@f).T).sum()/T**2
            s[1]+=4*sum(((yH[i]*a2)@(kL[lag]@a2).T)/pi[lag-1])/T**3
            e[2]+=2*((yL[i]*f)@(kL[lag]@f).T).sum()/T**2
            s[2]+=4*sum(((yL[i]*a2)@(kL[lag]@a2).T)/pi[lag-1])/T**4/pi[i]
            e[3]+=yL[i]@a2/T**2;s[3]+=yL[i]@(a2*a2)/T**4/pi[i]
        assert min(s)>=0
        FC=p[w<=c].sum()
        U= r/(1-r0)/(1-r/r0)
        bnd=[4*v/FC*U/T**2,4*v/FC*U/T**2,4*U*U/T**4,U/T**4]
        for x,ub in zip(s,bnd):assert x<=ub+1e-18
        return np.array(s),v
    rng=np.random.default_rng(1021);maxscaled=[0.]*4
    for T in (64,128,512):
        for target_v in (.25, .01,.001):
            for _ in range(12):
                p,q=rng.dirichlet(np.ones(6),size=2)
                h=rng.uniform(0,1,6)*sqrt(target_v)
                eta0=np.minimum(p,q).sum()
                ix=np.argsort(p/q);c=max(1,(p/q)[ix[np.searchsorted(np.cumsum(p[ix]),eta0/4)]])
                secs,v=run(T,p,q,h,eta0,c)
                maxscaled=np.maximum(maxscaled,secs*np.array([T*T/v,T*T/v,T**4,T**4]))
    print('108 deterministic checks of the disjoint IPS second-moment bounds; maxima after scaling:',
          ' '.join(f'{x:.6g}' for x in maxscaled))


def proposal_information():
    def risk(q,p,g,T):
     w=p/q; K=q[None,:]*np.minimum(1,w[None,:]/w[:,None]); K+=np.diag(1-K.sum(1))
     a=q.copy(); b=a*g; c=a@(g*g)
     for j in range(1,T):
      a=a@K; bp=b@K
      c+=2*(bp@g)+a@(g*g)
      b=bp+a*g
     return c/T**2

    def task(T,lam,eta,v):
     M=lam*T; u=min(v*1e-9,1e-11)
     pH=(1-eta)/(1-1/M); qH=pH/M
     ql=2*eta
     qd=1-ql-qH
     pd=np.array([1/6,1/3,1/2])
     qD=np.array([(1-eta)/2,(1-eta)/2,eta])
     p=np.r_[u*pd,eta-u-qH,pH]
     q=np.r_[qd*qD,ql,qH]
     assert abs(np.minimum(q,p).sum()-eta)<1e-14
     a3=-(1/6)/(1/2-2*eta/(3*(1-eta)))
     e=np.array([-1,1-2*eta*a3/(1-eta),a3,0,0])
     assert abs(p@e)+abs(q@e)<1e-14
     s=((v-u/4)/(1-u))**.5
     mark=np.array([.5,.5,.5,s,s])
     upper=np.r_[2*s*np.ones(3),1,-(1-pH)/pH]
     def one(d,theta):
      b=d*upper+theta*e
      pr=(1+b[:,None]*np.array([1,-1]))/2
      assert np.min(pr)>0
      return risk((q[:,None]*pr).reshape(-1),(p[:,None]*pr).reshape(-1),(mark[:,None]*np.array([1,-1])).reshape(-1),T)/v
     A=one(0,0); C=(one(.9,0)-A)/.9**2; d=((.5-A)/C)**.5
     th=min(.02,(1-d)/8)
     lo=one(d,-th); hi=one(d,th)
     assert lo<.5<hi
     print(T,lam,eta,v,'d',d,'pair',lo,hi,'scaled slope',eta*T*v**.5*(hi-lo)/(2*th))
    for Lam in [.5,.8]:
     for Eta in [.01,.04]:
      for V in [.0025,.25]:
       task(16384,Lam,Eta,V)


def target_information():
    def risk(q,p,g,T):
     w=p/q; K=q[None,:]*np.minimum(1,w[None,:]/w[:,None]); K+=np.diag(1-K.sum(1))
     a=q.copy(); b=a*g; c=a@(g*g)
     for j in range(1,T):
      a=a@K; bp=b@K
      c+=2*(bp@g)+a@(g*g)
      b=bp+a*g
     return c/T**2

    def task(T,lam,eta,v,gamma,alpha):
     pH=(1-eta)/(1-(alpha/gamma+(1-alpha)/lam)/T)
     p=np.array([0,alpha*pH,(1-alpha)*pH])
     q=np.array([0.,p[1]/(gamma*T),p[2]/(lam*T)])
     p[0]=eta-q[1]-q[2]; q[0]=1-q[1]-q[2]
     assert np.allclose(np.minimum(p,q).sum(),eta,atol=5e-16)
     def one(d,theta):
      base=np.sqrt(v)*d; m=-base*p[0]/pH; a=.25
      t0=(v-v*d*d*p[0]/pH)/(a*a*pH)
      t=np.array([t0,t0])+v*theta*np.array([1,-p[1]/p[2]])
      probs=np.array([1., *[x for z in t for x in [z/2,1-z,z/2]]])
      marks=np.r_[base,m-a,m,m+a,m-a,m,m+a]
      assert min(probs) > 0
      re=[1,3,3]; p7=np.repeat(p,re)*probs; q7=np.repeat(q,re)*probs
      assert abs(p7@marks)<1e-15 and abs(p7@(marks*marks)-v)<1e-15
      return risk(q7,p7,marks,T)/v
     A=one(0,0); C=one(1,0)-A; d=((.5-A)/C)**.5
     th=.1
     lo=one(d,th); hi=one(d,-th)
     assert lo<.5<hi
     w=p/q; B=q[None,:]*np.minimum(1,w[None,:]/w[:,None]); rho=1-B.sum(1); B+=np.diag(rho)
     pi=q.copy(); ret=2*np.cumsum(rho[:,None]**np.arange(T)[None,:],axis=1)-1
     coeff=np.zeros(3)
     for j in range(T):
      coeff+=pi*ret[:,T-1-j]/T**2
      pi=pi@B
     print(T,lam,eta,v,'d',d,'pair',lo,hi,'slopes',(hi-lo)/(2*th),'reuse',coeff[1:]/p[1:])
    for T in [2048,8192]:
     for Lam in [.5,.8]:
      for V in [.0025,.015625]:
       task(T,Lam,.12,V,.0005,.0001)

    def binary(T,lam,eta,v,gamma,alpha):
     pH=(1-eta)/(1-(alpha/gamma+(1-alpha)/lam)/T)
     p=np.array([0.,alpha*pH,(1-alpha)*pH])
     q=np.array([0.,p[1]/(gamma*T),p[2]/(lam*T)])
     p[0]=eta-q[1]-q[2]; q[0]=1-q[1]-q[2]
     assert abs(np.minimum(p,q).sum()-eta)<1e-14
     def one(d,theta):
      m=-d*p[0]/pH
      mu=np.array([d,m,m])+theta*np.array([0,1,-p[1]/p[2]])
      probs=(1+mu[:,None]*np.array([1,-1]))/2
      assert min(probs.flatten())>0
      return risk((q[:,None]*probs).flatten(),(p[:,None]*probs).flatten(),np.tile([1,-1],3)*np.sqrt(v),T)/v
     A=one(0,0); C=(one(.9,0)-A)/.9**2; d=((.5-A)/C)**.5
     th=.05
     lo,hi=one(d,-th),one(d,th)
     assert lo<.5<hi
     print('binary',T,lam,eta,v,'d',d,'pair',lo,hi,'slope',(hi-lo)/(2*th))
    for T in [2048,8192]:
     for Lam in [.5,.8]:
      binary(T,Lam,.12,.25,.0005,.002)


if __name__ == '__main__':
    for check in (threshold_pairs, correction_variances, proposal_information, target_information):
        print(f'=== {check.__name__} ===', flush=True)
        check()
    print('All core CPU checks passed.')
