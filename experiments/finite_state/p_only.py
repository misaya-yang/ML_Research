"""Zero-proposal-query high-mark IMH audit, plus finite exact controls."""
import numpy as np

def exact_risk(K, nu, h, m, t):
    a=np.zeros(len(h)); v=a.copy(); g=h-m
    for _ in range(t):
        tail=K@a
        v=g*g+2*g*tail+K@v
        a=g+tail
    return nu@v/t**2

def kernels(p,q):
    w=p/q
    a=p[None,:]/np.maximum(w[None,:],w[:,None])
    K=a+np.diag(1-a.sum(1))
    wb=np.r_[w,0.]
    pb=np.r_[p,0.]
    ab=pb[None,:]/np.maximum(1,np.maximum(wb[None,:],wb[:,None]))
    Kb=ab+np.diag(1-ab.sum(1))
    b=np.r_[q*(w>=1),q[w<1].sum()]
    return w,K,Kb,b

def one_audit(p,h,w,t,n,rng):
    # The code simulating the *audit* uses only fresh P and its normalized
    # weight. The exact Q array is not read, even to produce the start.
    n_calls=0
    def draw():
        nonlocal n_calls
        n_calls+=1
        return rng.choice(len(h)-1,p=p)
    def jump(k,x):
        u=0.
        for j in range(k,0,-1):
            u += (1-u) * (-np.expm1(np.log(rng.random())/j))
            if u*w[x]>1: break
            y=draw()
            if u*w[y]<=1: return y
        return x
    S=np.empty(n)
    for s in range(n):
        x=draw()
        if w[x]<1 or rng.random()*w[x]>1: x=len(h)-1
        i,j=sorted(rng.integers(t,size=2))
        x=jump(i,x)
        y=jump(j-i,x)
        m1=h[draw()];m2=h[draw()]
        S[s]=(h[x]-m1)*(h[y]-m2)
    return S,n_calls

def run():
    rng=np.random.default_rng(601)
    h=np.array([-1.,1.,-1.,1.,0.])
    # Ratio of rare group's mass p/q stays much larger than horizon.
    boxes=[]
    for qH in [.00003,.0004,.02]:
        p=np.array([.1,.4,.3,.2]); q=np.array([.7,.3-qH,qH*.6,qH*.4])
        w,K,Kb,newq=kernels(p,q); eta=np.minimum(p,q).sum()
        for T in [32,256,2048]:
            e=exact_risk(K,q,h[:-1],p@h[:-1],T)
            ea=exact_risk(Kb,newq,h,p@h[:-1],T)
            err=abs(e-ea)
            assert err<=32/(T*eta)
            boxes.append((qH,T,eta,e,ea,err))
        v,n=one_audit(p,h,np.r_[w,0],256,6000,rng)
        m=boxes[-2][4]
        print(f'monte-carlo qH={qH:g} P-only draws {n/len(v):.3f}/rep; exact surrogate={m:.8f}; estimate {v.mean():.8f} ±{1.96*v.std(ddof=1)/len(v)**.5:.8f}')
    print('qH,T,eta,true-risk,surrogate-risk,abs-bias')
    for b in boxes:
        print(','.join(f'{x:.9g}' for x in b))
if __name__=='__main__': run()
