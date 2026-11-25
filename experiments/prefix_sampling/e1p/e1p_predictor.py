"""Predict lattice gains from IID only; validate on exact synthetic trees first."""
import argparse,json,math
import numpy as np
from collections import defaultdict
from pathlib import Path
from e1p_sampling import logs,sha,write_frozen,seed
KS=(2,4,8);DELTA=(np.arange(4096)+.5)/4096
REV={k:np.array([int(f"{j:0{int(math.log2(k))}b}"[::-1],2) for j in range(k)])/k for k in KS}
def moment(f,n,r):
    # U-statistic from distinct donors is exact for r <= n. For higher order
    # moments the available n Bernoulli labels cannot identify an unbiased
    # estimate, so extrapolate from the last identified moment.
    if r==0:return 1.
    h=min(r,n)
    if h>f:return 0.
    z=math.comb(f,h)/math.comb(n,h)
    if r>n and z:return z*(f/n)**(r-n)
    return z
def empirical(rows):
    out=[]
    def walk(rows,pos,lo,hi):
        n=len(rows)
        if n<4 or all(len(t)<=pos for t,y in rows):
            out.append((lo,hi,sum(not y for t,y in rows),n));return
        grp=defaultdict(list)
        for seq,y in rows:grp[seq[pos] if len(seq)>pos else -1].append((seq,y))
        width=hi-lo
        for j,(key,sub) in enumerate(sorted(grp.items(),key=lambda a:(-len(a[1]),a[0]))):
            r=hi if j==len(grp)-1 else lo+width*len(sub)/n
            walk(sub,pos+1,lo,r);lo=r
    # Collapse shared one-child runs iteratively in the caller below.
    find_limit=max(map(lambda x:len(x[0]),rows))
    if find_limit>700:
        import sys
        sys.setrecursionlimit(max(sys.getrecursionlimit(),find_limit+100))
    walk(rows,0,0.,1.)
    assert len(out)<=len(rows) and abs(out[-1][1]-1.)<1e-12
    return out
def bank_prediction(rows):
    _,y=zip(*rows);n=len(y);f=n-sum(y);mu=1-f/n
    leaves=empirical(rows)
    tails=np.array([z[1] for z in leaves]); moments=np.array([[moment(f,n,r) for r in range(9)] for _,_,f,n in leaves])
    q={}
    for k in KS:
        loc=np.searchsorted(tails,(DELTA[:,None]+REV[k])%1.,side="right")
        fail=np.ones(len(DELTA))
        for j in range(k):
            u=loc[:,j]
            total=(loc==u[:,None]).sum(1)
            first=np.ones(len(DELTA),bool) if j==0 else ~(loc[:,:j]==u[:,None]).any(1)
            fail*=np.where(first,moments[u,total],1.)
        iid=1-moment(f,n,k)
        coupled=float(1-fail.mean())
        hi=min(1,k*mu)-(1-(1-mu)**k)
        q[str(k)]={"predicted_iid":iid,"predicted_lattice":coupled,"predicted_gain":coupled-iid,
                   "mu":mu,"gmax":hi,"normalized_gain":(coupled-iid)/hi if hi>=.02 else None,
                   "num_leaves":len(leaves)}
    return q

def exact_lattice(success,k):
    segments=[]
    for l,r in success:
        for off in REV[k]:
            x=(l-off)%1;w=r-l
            if x+w<=1:segments.append((x,x+w))
            else:segments.extend(((x,1.),(0.,x+w-1.)))
    total=right=0.
    for l,r in sorted(segments):
        total+=max(0.,r-max(right,l));right=max(right,r)
    return total

def synthetic():
    rng=np.random.default_rng(27019)
    metrics={k:[] for k in KS}
    for trial in range(200):
        depth=int(rng.integers(4,9));all_leaves=[]
        def grow(seq,lo,hi,d,logit):
            if d==depth or (d>=2 and rng.random()<.42):
                y=int(rng.random() < 1/(1+np.exp(-logit)))
                all_leaves.append((seq,lo,hi,y));return
            n=int(rng.integers(2,7));w=sorted(rng.dirichlet(np.full(n,.8)),reverse=True)
            width=hi-lo
            for j,v in enumerate(w):
                r=hi if j==n-1 else lo+width*v
                grow((*seq,j),lo,r,d+1,logit+rng.normal(scale=.5)); lo=r
        grow((),0.,1.,0,rng.normal())
        idx=np.searchsorted([r for seq,l,r,y in all_leaves],rng.random(32),side="right")
        obs=[(all_leaves[i][0],all_leaves[i][3]) for i in idx]
        pr=bank_prediction(obs)
        success=[(l,r) for seq,l,r,y in all_leaves if y]
        mu=sum(r-l for l,r in success)
        for k in KS:
            gt=exact_lattice(success,k)
            g0=1-(1-mu)**k
            p=pr[str(k)]
            metrics[k].append((p['predicted_lattice']-gt,p['predicted_gain']-(gt-g0)))
    data={}
    for k,x in metrics.items():
        x=np.array(x)
        data[str(k)]={"lattice_pass_bias":float(x[:,0].mean()),"lattice_pass_rmse":float(np.sqrt(np.mean(x[:,0]**2))),
                      "gain_bias":float(x[:,1].mean()),"gain_rmse":float(np.sqrt(np.mean(x[:,1]**2)))}
    data["validated"]=abs(data["8"]["lattice_pass_bias"])<=.01 and abs(data["8"]["gain_bias"])<=.01
    data["design"]="200 varied trees, max depth 4–8, Dirichlet branching 2–6, exact full-tree arithmetic lattice, 32 IID draws each"
    return data
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("validate","freeze"));p.add_argument("--stage",choices=("F1F2","F3"),default="F1F2")
    a=p.parse_args()
    if a.mode=="validate":
        x=synthetic();write_frozen("predictor_validation.json",x);print(json.dumps(x,indent=2));return
    cert=json.loads(Path("predictor_validation.json").read_text());assert cert["validated"],"Synthetic calibration gate failed."
    formats=["F1","F2"] if a.stage=="F1F2" else ["F3"]
    src={t:sha(f"iid_{t}.jsonl") for t in formats}
    out={}
    for t in formats:
        by=defaultdict(list)
        for r in logs(f"iid_{t}.jsonl"):
            by[r["id"]].append((r["token_ids"],int(r["correct"])))
        if t=="F3":assert len(by)==32 and all(len(rr)==24 for rr in by.values())
        else:assert len(by)==64 and all(len(rr)==32 for rr in by.values())
        out[t]={i:bank_prediction(rr) for i,rr in sorted(by.items())}
    dest=f"predictions_{a.stage}.json";write_frozen(dest,{"iid_sha256":src,"prediction":out,"validation_sha256":sha("predictor_validation.json")})
    p=Path("prediction_seals.json");seal=json.loads(p.read_text()) if p.exists() else {}
    if dest in seal:assert seal[dest]==sha(dest)
    else:
        seal[dest]=sha(dest);p.write_text(json.dumps(seal,indent=2)+"\n")
        with Path("PROTOCOL_LOG.md").open("a") as f:f.write(f"\n- Frozen before lattice: `{dest}`, SHA-256 `{seal[dest]}`.\n")
    print(dest,seal[dest])
if __name__=="__main__":main()
