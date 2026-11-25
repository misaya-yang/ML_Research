"""A strict-budget share-or-restart rule. Known failed original; at most two new attempts.
One try is capped at a fixed slice, then spend the rest on a fresh fallback.
Group by observable cost/confidence; learn no model, choose only a dev routing rule.
"""
from pathlib import Path
from bisect import bisect_right
from collections import defaultdict, Counter
import hashlib,json,statistics as S,random

def load(p):
    return {x['problem_num']:x for x in map(json.loads,p.read_text().splitlines())}
def num(x):
    try:return int(str(x).strip())
    except (ValueError,TypeError):return None
def key(pid):
    return hashlib.sha256(('capped-prefix-v1:'+str(pid)).encode()).digest()

BUDGETS=(2048,4096,8192,16384)
SLICES=(0.25,0.5,0.75)
FEATS=('none','short_base','tail_drop','base_conf')
rng=random.Random(1251553)

def cell(root,departures):
    orig=load(root/'_init.jsonl')
    by={c:load(root/f'_rm{100-c}pct_full_x{n}.jsonl') for c,n in departures.items()}
    rows={}
    mismatches=0; skipped=0
    for pid,r in sorted(orig.items()):
        O=r['all_answers']; G=num(r['gold_answer']); assert G is not None
        T=[int(x[1]) for x in O]; Y=[num(x[0])==G for x in O]
        N=len(O); good=sorted(t for t,y in zip(T,Y) if y)
        assert N>=20 and all(t>0 for t in T)
        maps={}
        for c,z in by.items():
            row=z[pid]; P=row['all_answers']
            assert num(row['gold_answer'])==G and len(row['regen_answers'])==len(P)
            if P==O: maps[c]={i:i for i in range(N)}; continue
            mismatches+=1
            a,b=Counter(map(tuple,O)),Counter(map(tuple,P))
            u={tuple(x):j for j,x in enumerate(P) if b[tuple(x)]==1}
            maps[c]={i:u[tuple(x)] for i,x in enumerate(O) if a[tuple(x)]==1 and tuple(x) in u}
        result=[]
        for i,y in enumerate(Y):
            if y:continue
            if not all(i in x for x in maps.values()): skipped+=1; continue
            conf=r["confidences"][i] or {}
            f={'none':0,'short_base':T[i],
               'tail_drop': (conf.get('tail_conf') or 0)-(conf.get('mean_conf') or 0),
               'base_conf':conf.get('mean_conf') or 0}
            out={}
            for B in BUDGETS:
                def fresh(b):return bisect_right(good,b)/(N-1)
                Z={(0,0):fresh(B)}
                for cut in SLICES:
                    cap=int(B*cut)
                    def val(t,y):
                        return float(y and t<=cap) or fresh(B-min(t,cap))
                    v=(sum(val(t,y) for t,y in zip(T,Y))-val(T[i],False))/(N-1)
                    Z[(0,cut)]=v
                    for d,dat in by.items():
                        a=dat[pid]["regen_answers"][maps[d][i]][0]
                        Z[d,cut]=val(int(a[1]),num(a[0])==G)
                out[B]=Z
            result.append((f,out))
        if result: rows[pid]=result
    print('problems with shifted indexing',mismatches,'unmatched originally-wrong rows excluded',skipped)
    return rows

def evaluation(R, base_actions=None, router=True):
    ids=sorted(R,key=key); d,t=ids[:len(ids)//2],ids[len(ids)//2:]
    def perf(ids, B, rule):
        return [S.mean(s[B][rule(f)] for f,s in R[i]) for i in ids]
    print('dev/test problems',len(d),len(t),'failed initial traces',sum(len(R[i]) for i in t))
    for B in BUDGETS:
        actions=set(next(iter(R.values()))[0][1][B])
        fixed=max([a for a in actions if a[0]==0],key=lambda a:(S.mean(perf(d,B,lambda f:a)),-a[1]))
        unconditional=max(actions,key=lambda a:(S.mean(perf(d,B,lambda f:a)),-a[0],-a[1]))
        base=perf(t,B,lambda f:fixed)
        u=perf(t,B,lambda f:unconditional)
        candidates=[]; pure=[]
        for feat in FEATS:
            if feat=='none':continue
            th=S.median(f[feat] for i in d for f,_ in R[i])
            def choose(lt, options):
                samples=[[s[B] for f,s in R[i] if (f[feat]<th)==lt] for i in d]
                return max(options,key=lambda a:(S.mean(S.mean(s[a] for s in rs) for rs in samples if rs),-a[0],-a[1]))
            for options, dest in ((actions,candidates),([a for a in actions if a[0]==0],pure)):
                a0,a1=choose(True,options),choose(False,options)
                fn=lambda f,a0=a0,a1=a1,t=th,g=feat:a0 if f[g]<t else a1
                dest.append((S.mean(perf(d,B,fn)),feat,a0,a1,fn))
        best=max(candidates+[(S.mean(perf(d,B,lambda f:unconditional)),'none',unconditional,unconditional,lambda f:unconditional)],key=lambda x:(x[0],x[1]=='none'))
        ctl=max(pure+[(S.mean(perf(d,B,lambda f:fixed)),'none',fixed,fixed,lambda f:fixed)],key=lambda x:(x[0],x[1]=='none'))
        a=perf(t,B,best[-1]); control=perf(t,B,ctl[-1]);deltas=[x-y for x,y in zip(a,control)]
        print('fresh with access to routing covariates',B,'score',round(S.mean(control),4),'policy',ctl[1:4])
        z=sorted(S.mean(rng.choices(deltas,k=len(deltas))) for _ in range(4000))
        print(B,'fresh/fixed hybrid/routed',*[round(S.mean(x),4) for x in (base,u,a)],'dev baseline',fixed,'dev fixed',unconditional,'route',best[1:4],'diff CI',round(z[100],4),round(z[3900],4))

roots=[(Path('external_aime'),{25:3,50:3,75:3})]
roots.extend((p,{75:1}) for p in sorted(Path('external_multi').glob('*_jsonl')) if len(list(p.glob('*.jsonl')))==2)
for p,c in roots:
    print(p.name);evaluation(cell(p,c))
