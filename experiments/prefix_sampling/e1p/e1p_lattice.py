"""Independent finite categorical oracle and sequence-level lattice test."""
import math,json
from pathlib import Path

def fp32_floor(u):
    import numpy as np
    a=np.float32(u)
    if float(a)>u:a=np.nextafter(a,np.float32(0))
    return a

def advance(u,lo,hi,p,bits,rng):
    """Decode conditional residual in float64, refreshing only after 40 bits."""
    bits-=math.log2(p)
    if bits>40:return rng.random(),bits,True
    w=hi-lo
    if not w>0:raise ValueError("Non-positive interval after numerical rounding.")
    return min(max((u-lo)/w,0.),math.nextafter(1.,0.)),bits,False

def toy():
    import numpy as np
    from scipy.stats import chisquare
    from e1p_predictor import exact_lattice
    probs={(0,):[.59,.25,.16],(1,):[.51,.29,.20],(2,):[.70,.20,.10],():[.62,.25,.13]}
    leaf=[];good=[]
    def tree(path,lo,hi):
        if len(path)==3:
            y=(sum(path)%3==0);leaf.append((path,lo,hi,y))
            if y:good.append((lo,hi))
            return
        w=hi-lo
        for j,p in enumerate(probs.get(path,probs[path[:1]])):
            up=hi if j==2 else lo+w*p
            tree((*path,j),lo,up);lo=up
    tree((),0.,1.)
    a=np.array([r-l for _,l,r,_ in leaf]);assert abs(a.sum()-1)<1e-12
    counts=100000;d=np.random.default_rng(47127).random(counts)
    rev=np.array([int(f"{j:03b}"[::-1],2) for j in range(8)])
    q=(d[:,None]+rev[None,:]/8)%1.
    lloc=np.zeros(q.shape,int)
    for w in range(3):
        previous=lloc.copy()
        for prefix in range(3**w):
            mask=previous==prefix
            if not mask.any():continue
            if w==0:p=probs[()]
            else:p=probs[(prefix//(3**(w-1)),)]
            cm=np.cumsum([0,*p]);j=np.searchsorted(cm[1:],q[mask],side="right")
            q[mask]=(q[mask]-cm[j])/np.array(p)[j];lloc[mask]=3*prefix+j
    tol={}
    for j in range(8):
        x=np.bincount(lloc[:,j],minlength=27)
        tol[str(j)]=float(chisquare(x,counts*a/a.sum()).pvalue)
    match={}
    for k in (2,4,8):
        x=float(np.array([y for _,_,_,y in leaf])[lloc[:,:k]].any(1).mean())
        ref=exact_lattice(good,k)
        match[str(k)]={"observed":x,"exact":ref,"abs_error":abs(x-ref)}
    return {"draws_per_point":counts,"point_chisquare_p":tol,"pass":match,
            "passed":bool(min(tol.values())>.01 and max(x["abs_error"] for x in match.values())<.005)}

if __name__=="__main__":
    r=toy();Path("lattice_toy_validation.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
    assert r["passed"]
