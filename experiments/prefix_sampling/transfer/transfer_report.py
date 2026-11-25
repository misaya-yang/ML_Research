import json,statistics as st,sys,random
from pathlib import Path
f=Path(sys.argv[1] if len(sys.argv)>1 else 'transfer_frozen.jsonl')
rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]
assert len(rows)==len({r['index'] for r in rows})
B=256
print('completed',len(rows),'base correct',sum(not x['failed'] for x in rows),'base capped',sum(x['base']['n']==768 and not x['base']['eos'] for x in rows))
failed=[x for x in rows if x['failed']]
print('failed',len(failed),'short failed',sum(x['short'] for x in failed))
rng=random.Random(17467)
a=['full_fresh','split_fresh','fixed_fork','routed']
for name,q in [('all failed',failed),('short failed',[x for x in failed if x['short']]),('long failed',[x for x in failed if not x['short']])]:
 if not q:continue
 print(name,len(q))
 v=lambda r,arm:any(z['ans']==r['gold'] for z in r['arms'][arm])
 for arm in a:
  x=[r['arms'][arm] for r in q]
  assert all(sum(r['n'] for r in z)<=B for z in x)
  print(arm,'rescued',sum(v(r,arm) for r in q),'new tokens',sum(sum(z['n'] for z in v) for v in x),
    'mean refilled positions',round(st.mean(sum(z['prefill'] for z in v) for v in x),1))
 for ref in a[:2]:
  d=[int(v(r,'routed'))-int(v(r,ref)) for r in q]
  z=sorted(st.mean(rng.choices(d,k=len(d))) for _ in range(5000))
  print('routed vs',ref,'W/L',d.count(1),d.count(-1),'95% descriptive paired bootstrap',z[125],z[4875],
     'discordant cases',[(r['index'],dv) for r,dv in zip(q,d) if dv])
