"""Read-only metrics and gate verdicts for the frozen E1-P contract."""
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import chi2,spearmanr
from e1p_sampling import logs,seed,sha,write_frozen
KS=(2,4,8)
def byq(rows):
    d=defaultdict(list)
    for r in rows:d[r["id"]].append(r)
    return {i:sorted(x,key=lambda r:r["j"]) for i,x in d.items()}
def iid_pass(rows,k):
    n=len(rows);f=n-sum(rows)
    return 1-(math.comb(f,k)/math.comb(n,k) if f>=k else 0)
def group_pass(groups,k):
    return sum(any(g[:k]) for g in groups)/len(groups)
def exactness():
    A=byq(logs("exactness_iid_F1.jsonl"));B=byq(logs("exactness_lattice_F1.jsonl"))
    assert len(A)==len(B)==5 and set(A)==set(B) and all(len(a)==400 for a in [*A.values(),*B.values()])
    report={}
    for feat in ("first_token","answer"):
        s=0.;df=0;detail=[]
        for i in sorted(A):
            def get(r):return r["token_ids"][0] if feat=="first_token" else r["answer"]
            a=Counter(map(get,A[i]));b=Counter(map(get,B[i]));pool=a+b
            keys=sorted((k for k in pool if pool[k]>=10),key=str)
            c=np.array([[d[k] for k in keys]+[sum(d[k] for k in pool if k not in keys)] for d in (a,b)])
            c=c[:,c.sum(0)>0]
            if c.shape[1]<2:continue
            e=c.sum(0)/2
            z=float(np.sum((c-e)**2/e));df+=c.shape[1]-1;s+=z
            detail.append({"id":i,"categories":c.shape[1],"chi2":z,"df":c.shape[1]-1})
        # This is the specified marginal diagnostic, not proof of exact marginals:
        # lattice outcomes are clustered so row-independence is not guaranteed.
        report[feat]={"stat":s,"df":df,"p":float(chi2.sf(s,df)) if df>0 else None,"details":detail}
    report["passed"]=all(v["p"] is not None and v["p"]>.01 for v in list(report.values()))
    write_frozen("real_model_exactness.json",report)
    print(json.dumps(report,indent=2));return report["passed"]
def scoring(stage):
    formats=("F1","F2") if stage=="F1F2" else ("F3",)
    seal=json.loads(Path("prediction_seals.json").read_text());p=f"predictions_{stage}.json";assert seal[p]==sha(p)
    blob=json.loads(Path(p).read_text());assert blob["iid_sha256"]=={t:sha(f"iid_{t}.jsonl") for t in formats}
    display={}; per=[];pl=[]
    for t in formats:
        a=byq(logs(f"iid_{t}.jsonl"));b=byq(logs(f"lattice_{t}.jsonl"));p0=blob["prediction"][t]
        assert set(a)==set(b)==set(p0)
        expected=24 if t=="F3" else 32
        assert len(a)==(32 if t=="F3" else 64)
        rng=np.random.default_rng(seed("paired-question-bootstrap",t));kid=sorted(a)
        display[t]={}
        for k in KS:
            vals=[];spec=[];split=[]
            for i in kid:
                rr=a[i];ll=b[i];assert [x["j"] for x in rr]==[x["j"] for x in ll]==list(range(expected))
                assert all(l["group"]==j//8 and l["point"]==j%8 for j,l in enumerate(ll))
                yy=[r["correct"] for r in rr];y=[r["correct"] for r in ll];groups=[y[j:j+8] for j in range(0,len(ll),8)]
                ii=iid_pass(yy,k);la=group_pass(groups,k);pr=p0[i][str(k)]
                vals.append((ii,la));gain=la-ii
                successpairs=np.mean([g[0]*g[j] for g in groups for j in range(1,k)])
                row={"format":t,"id":i,"k":k,"iid":ii,"lattice":la,"realized_gain":gain,
                     "predicted_gain":pr["predicted_gain"],"gmax":pr["gmax"],
                     "co_success_excess":float(successpairs-np.mean(yy)**2),
                     "mixed_groups":sum(0<sum(g)<8 for g in groups)/len(groups),
                     "iid_cap_rate":sum(r["capped"] for r in rr)/len(rr),
                     "lattice_cap_rate":sum(r["capped"] for r in ll)/len(ll)}
                if pr["gmax"]>=.02:
                    spec.append((gain/pr["gmax"],pr["predicted_gain"]/pr["gmax"]))
                    if k==8:pl.append((pr["predicted_gain"],gain))
                per.append(row)
                for j in range(0,len(ll),8):
                    root=ll[j];prefix=root["token_ids"]
                    with np.load(root["npz"]) as f:ss=-np.log2(f["prob"].astype(float))
                    c=np.r_[0.,np.cumsum(ss)]
                    for x in ll[j+1:j+k]:
                        tt=next((h for h,(a,b) in enumerate(zip(prefix,x["token_ids"])) if a!=b),min(len(prefix),len(x["token_ids"])))
                        split.append((tt,c[tt]))
            vals=np.asarray(vals);sp=np.asarray(spec);g=vals[:,1]-vals[:,0]
            bt=rng.integers(len(g),size=(10000,len(g)))
            ent=[r for r in per if r["k"]==k and r["format"]==t]
            alpha={"iid":float(vals[:,0].mean()),"lattice":float(vals[:,1].mean()),"gain":float(g.mean()),
                   "gain_ci95":np.quantile(g[bt].mean(1),[.025,.975]).tolist(),
                   "pred":float(np.mean([v["predicted_gain"] for v in ent])),
                   "co_success_excess":float(np.mean([v["co_success_excess"] for v in ent])),
                   "mixed_groups":float(np.mean([v["mixed_groups"] for v in ent])),
                   "iid_cap_rate":float(np.mean([v["iid_cap_rate"] for v in ent])),
                   "lattice_cap_rate":float(np.mean([v["lattice_cap_rate"] for v in ent])),
                   "normalized_q":len(spec),"normalized_gain":None,"normalized_ci95":None,
                   "split_depth_median":float(np.median(np.array(split)[:,0])),
                   "split_bits_median":float(np.median(np.array(split)[:,1])),
                   "refresh_step_median":None}
            refresh=[r["refresh_step"] for rr in b.values() for r in rr if r["refresh_step"] is not None]
            if refresh:alpha["refresh_step_median"]=float(np.median(refresh))
            if len(spec):
                idx=rng.integers(len(sp),size=(10000,len(sp)))
                alpha.update(normalized_gain=float(sp[:,0].mean()),normalized_prediction=float(sp[:,1].mean()),
                             normalized_ci95=np.quantile(sp[:,0][idx].mean(1),[.025,.975]).tolist())
            display[t][str(k)]=alpha
    rho=float(spearmanr(*zip(*pl)).statistic) if len(pl)>1 else float("nan")
    rho=rho if np.isfinite(rho) else None
    tol=all(abs(d["gain"]-d["pred"])<=max(.30*abs(d["pred"]),.01) for vv in display.values() for d in vv.values())
    f1=display.get("F1",{}).get("8",{})
    gate=bool(f1 and f1["normalized_gain"] is not None and f1["normalized_gain"]>=.3
              and f1["normalized_ci95"][0]>0 and tol and rho is not None and rho>=.5)
    out={"stage":stage,"formats":display,"spearman":rho,"predicate_tolerance_met":tol,"will_run_F3":gate}
    write_frozen(f"report_{stage}.json",out)
    with open(f"per_problem_{stage}.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    if stage=="F1F2" and gate:
        acc=byq(logs("iid_F2.jsonl"))
        eligible=sorted(((-sum(r["correct"] for r in rr)/len(rr),i) for i,rr in acc.items() if not all(r["correct"] for r in rr)))
        assert len(eligible)>=32
        write_frozen("problems_F3.json",{"ids":[i for _,i in eligible[:32]],"source_iid_sha256":sha("iid_F2.jsonl")})
    print(json.dumps(out,indent=2));return gate
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("exactness","F1F2","F3"));a=p.parse_args()
    ok=exactness() if a.stage=="exactness" else scoring(a.stage)
    if a.stage!="F3":raise SystemExit(0 if ok else 2)
