"""Explain the previously frozen 4096-token Ministral branch selection."""
from pathlib import Path
import json,hashlib,statistics as st
from bisect import bisect_right
P=Path('external_multi/Ministral-3-14B-Reasoning-2512_aime2025_jsonl')
ld=lambda f:{r['problem_num']:r for r in map(json.loads,(P/f).read_text().splitlines())}
O=ld('_init.jsonl'); R=ld('_rm25pct_full_x1.jsonl')
def answer(x):
 try:return int(str(x).strip())
 except (TypeError,ValueError):return None
by={}
for i,r in O.items():
 G=answer(r['gold_answer']); a=r['all_answers']; z=R[i]['regen_answers']
 assert R[i]['all_answers']==a and len(z)==len(a)
 fresh=[(x[1],answer(x[0])==G) for x in a]; N=len(a)
 good=sorted(t for t,y in fresh if y)
 rows=[]
 for j,(cost,y) in enumerate(fresh):
  if y:continue
  k=z[j][0]; B=4096; A=3072
  g=lambda n:bisect_right(good,n)/(N-1)
  def parts(length,correct):
   f=g(B-min(length,A))
   d=int(correct and length<=A)
   return [f,d*(1-f),int(correct),int(length<=A),d]
  rows.append((cost,g(B),parts(k[1],answer(k[0])==G), [st.mean(x) for x in zip(*(parts(*r) for h,r in enumerate(fresh) if h!=j))]))
 if rows:by[i]=rows
ids=sorted(by,key=lambda x:hashlib.sha256(('capped-prefix-v1:'+str(x)).encode()).digest())
dev,test=ids[:len(ids)//2],ids[len(ids)//2:]
threshold=st.median(r[0] for i in dev for r in by[i])
print('threshold base tokens',threshold)
for tag,cond in [('short',lambda n:n<threshold),('long',lambda n:n>=threshold)]:
 per=[]
 for i in test:
  k=[(g,*a,*b) for n,g,a,b in by[i] if cond(n)]
  if k: per.append([st.mean(x) for x in zip(*k)])
 a=list(map(st.mean,zip(*per)))
 print(tag,len(per),'fresh uncapped',round(a[0],5),
       'fork fallback,direct,correct,finish,correct&finish',[round(x,5) for x in a[1:6]],
       'fresh same cap parts',[round(x,5) for x in a[6:11]],
       'fork total',round(a[1]+a[2],5),'fresh same cap total',round(a[6]+a[7],5))
