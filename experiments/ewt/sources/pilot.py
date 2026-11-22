"""Normalized model pilot; do not infer results for source-starved hidden normalize case."""
import numpy as np,os,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'
os.environ.setdefault('EWT_DATA',str(DATA))
from scipy.special import softmax
from capture import Model,read,Draw,heldout
from runtime_paths import new_run_dir
def ew():
    test=read('test')
    # No previous test outcomes influence selection; skip used 32-doc final,
    # original risk audits' heavily reused prefix and suffix, and CPU reuse IDs.
    fam=lambda s:s.rsplit('-',1)[0]
    skip={fam(x[0]) for x in test[:220]+test[-170:]+heldout()}
    for fname,key in [
        (DATA/'cold_previous_panel.csv','sent_id'),
        (DATA/'stationary_previous_panel.csv','id')]:
        with open(fname) as f: skip.update(fam(row[key]) for row in csv.DictReader(f))
    for j,s in enumerate(sorted(test,key=lambda x:hashlib.sha256(('dual-normalized-earlypilot-v1'+x[0]).encode()).digest())):
        if fam(s[0]) in skip or not 12<=len(s[1])<=24: continue
        yield s;skip.add(fam(s[0]))
def run():
    from itertools import islice
    model=Model(read('train'))
    T=1024;N=4096;B=128; NP=65536;REPS=5700;a=.0125;r=.5
    out_dir=new_run_dir("ewt-high-mark")
    hsum=[]
    for d,(doc,ws,gs) in enumerate(islice(ew(),6)):
        p=model.law(ws);G=np.array([model.ix[u] for u in gs]);j=int(np.argmin(abs(softmax(p.e,axis=1)[np.arange(len(G)),G]-.55)))
        q=model.law(ws,.72,1.)
        pdraw=Draw(p,q,p,j,G[j]);qdraw=Draw(p,q,q,j,G[j]);rp=np.random.default_rng(1019091+d);rq=np.random.default_rng(1021085+d)
        logz=p.z-q.z # Included in the normalized-score oracle here, not a hidden evaluator leak.
        ps,ph=pdraw.sample(rp,NP);w=np.exp(ps-logz)
        pv=p.moments()[0][j,G[j]];var=pv*(1-pv) # never used in either confidence calculation
        # Independent overlap pilot, predicted finite-lag loss budget.
        eta_hat=np.minimum(1,1/w[:1024]).mean()
        etal=max(0,eta_hat-np.sqrt(np.log(1/a)/(2*1024)))
        rr=iter(range(1024,NP)); scores=[]
        def draw():return next(rr)
        def hh(y):return int(ph[y]) if y>=0 else 0
        def jump(l,x):
            u=0.
            for k in range(l,0,-1):
                u += (1-u)*(-np.expm1(np.log(rp.random())/k))
                if x>=0 and u*w[x]>1: break
                y=draw()
                if u*w[y]<=1: return y
            return x
        for v in range(REPS):
            x=draw()
            if w[x]<1 or rp.random()*w[x]>1: x=-1
            i,jj=sorted(rp.integers(T,size=2))
            x=jump(i,x);y=jump(jj-i,x);u=hh(draw());v=hh(draw())
            scores.append((hh(x)-u)*(hh(y)-v)-r*(u-v)**2/2)
        used=next(rr); # next unused index is exactly number used
        # An endpoint coupling for high-mark-only. Bound exact average bad
        # probability, not only its upper asymptotic order. Binary U span 2.
        low=1-etal
        err=2*np.sum((np.r_[2*(T-np.arange(T))-1]+np.r_[0,2*(T-np.arange(1,T))])*low**np.arange(T))/T**2
        mean=np.mean(scores);hw=(2+r/2)*np.sqrt(np.log(1/a)/(2*REPS));u_a=mean+err+hw
        # Independent full-chain evaluation, 4096 trajectories. Also use an
        # independent fixed-cost prefix from the same truth bank as a pilot
        # baseline with fresh P anchors (separate from the tested estimator).
        out=np.empty(N);anchors_ph=pdraw.sample(np.random.default_rng(1031129+d),2*N)[1].reshape(N,2)
        for at in range(0,N,B):
            s,h=qdraw.sample(rq,B*T);s=(s-logz).reshape(B,T);h=h.reshape(B,T)
            z=s[:,0].copy();x=h[:,0].copy();count=x.astype(np.int32)
            for k in range(1,T):
                v=np.log(rq.random(B))<=s[:,k]-z;z=np.where(v,s[:,k],z);x=np.where(v,h[:,k],x);count+=x
            out[at:at+B]=count/T
        truth=np.mean((out-pv)**2)-r*var;tr_se=np.std((out-pv)**2,ddof=1)/np.sqrt(N)
        rep=(used//(T+2)); y1,y2=anchors_ph[:rep].T.astype(float)
        path=(out[:rep]-y1)*(out[:rep]-y2)-r*(y1-y2)**2/2
        direct_m=path.mean();direct_u=direct_m+2*np.sqrt(np.log(1/a)/(2*rep))
        result=dict(d=d,doc=doc,target_mean_eval=pv,eta_hat=eta_hat,eta_lower=etal,signed_truth=truth,truth_se=tr_se,signed_p_only=mean,p_only_bias_allowance=err,p_only_upper=u_a,scored_P_used=used,scored_P_acquired=NP,scored_Q_used=0,direct_signed=direct_m,direct_upper=direct_u,direct_paths=rep)
        hsum.append(result);print(result,flush=True)
        with open(out_dir/'pilot.csv','w') as f:
            a0=csv.DictWriter(f,hsum[0]);a0.writeheader();a0.writerows(hsum)

if __name__=='__main__':run()
