"""Exact fixed-depth independent suffix replay on Claude's archived panel.

Uses the original remaining-character cost proxy. Initial checked failure is
sunk for all arms. A success terminates under a checked-answer benchmark;
does not estimate unverifiable single-answer quality. No model calls.
"""
import hashlib, json, pathlib, statistics, sys

root=pathlib.Path(sys.argv[1])
depths=(0,.25,.5,.7)
budgets=(1.,2.)
models=('deepseek-r1-distill-qwen-14b','deepseek-r1-distill-llama-8b')

for model in models:
    rows=[]
    for f in sorted((root/model).glob('*/incorrect_base_solution/*/chunks_labeled.json')):
        arr=sorted(json.loads(f.read_text()),key=lambda a:a['chunk_idx'])
        L=sum(len(x['chunk']) for x in arr)
        starts=[]; s=0
        for x in arr:
            starts.append(s/L); s+=len(x['chunk'])
        arms={}
        for d in depths:
            i=next((i for i,s in enumerate(starts) if s>=d),len(starts)-1)
            q=arr[i]['accuracy']; c=1-starts[i]
            arms[d]=tuple(1-(1-q)**int((B+1e-9)//c) for B in budgets)
        rows.append((f.parent.name,arms))
    assert len(rows)==20
    rows.sort(key=lambda r:hashlib.sha256(('prefix_budget_v1:'+r[0]).encode()).digest())
    dev,test=rows[:10],rows[10:]
    scores=lambda p,d,b:statistics.mean(x[d][b] for _,x in p)
    print(model)
    for b,B in enumerate(budgets):
        selected=max(depths,key=lambda d:(scores(dev,d,b),-d))
        print('  budget',B,'dev-depth',selected,'holdout',len(test),
              'fresh',round(scores(test,0,b),4),
              'frozen-depth',round(scores(test,selected,b),4),
              'quarter',round(scores(test,.25,b),4),
              'midpoint',round(scores(test,.5,b),4))
