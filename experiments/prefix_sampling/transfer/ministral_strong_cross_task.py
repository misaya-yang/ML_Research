"""No tuning: copy the AIME development rule to two new archive benchmarks."""
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import json,statistics as S,random
D=Path('external_ministral_transfer')
B,C,TH=4096,3072,8478
def value(s):
 try:return int(str(s).strip())
 except (ValueError,TypeError):return None
def read(p):return {a['problem_num']:a for a in map(json.loads,p.read_text().splitlines())}
r=random.Random(267421)
for d in sorted(D.glob('*_jsonl')):
 if not (d/'_rm25pct_full_x1.jsonl').exists():continue
 A,Z=read(d/'_init.jsonl'),read(d/'_rm25pct_full_x1.jsonl')
 out=[]; na=0; aligned=0; total=0; excluded=0; nongrade=0
 for i,a in A.items():
  G=value(a['gold_answer'])
  if G is None:nongrade+=1;continue
  assert i in Z and G==value(Z[i]['gold_answer'])
  X=a['all_answers'];Y=Z[i]['all_answers']
  mapidx=dict(enumerate(range(len(X)))) if X==Y else {}
  if not mapidx:
   numx,numy=Counter(map(tuple,X)),Counter(map(tuple,Y))
   jm={tuple(y):j for j,y in enumerate(Y) if numy[tuple(y)]==1}
   mapidx={j:jm[tuple(x)] for j,x in enumerate(X) if numx[tuple(x)]==1 and tuple(x) in jm}
  N=len(X); Suc=sorted(x[1] for x in X if value(x[0])==G); good=lambda c:bisect_right(Suc,c)/(N-1)
  dist=[(x[1],value(x[0])==G) for x in X]
  row=[]
  for j,(n,v) in enumerate(dist):
   if v:continue
   if j not in mapidx:excluded+=1;continue
   gen=Z[i]['regen_answers'][mapidx[j]][0]
   total+=1; short=n<TH; na+=short; aligned+=len(gen)>0
   def val(t,ok):return (1 if ok and t<=C else good(B-min(t,C)))
   s0=good(B)
   s1=val(gen[1],value(gen[0])==G)
   s2=(sum(val(t,ok) for t,ok in dist)-val(n,False))/(N-1)
   F=[]
   for C0 in (1024,2048,3072):
    vf=lambda t,ok:1 if ok and t<=C0 else good(B-min(t,C0))
    F.append((sum(vf(t,ok) for t,ok in dist)-vf(n,False))/(N-1))
   row.append((s0,*F,s1,s1 if short else s0))
  if row:out.append(tuple(S.mean(c) for c in zip(*row)))
 print(d.name,'archived problems',len(A),'non-integer gold excluded',nongrade,'tested',len(out),'failed originals',total,'treated',na,'unmatched',excluded)
 if not out:continue
 print('full fresh / first cap 1024 / 2048 / 3072 / always prefix / frozen routed',[round(S.mean(c),5) for c in zip(*out)])
 for j in range(4):
  d=[x[-1]-x[j] for x in out]
  b=sorted(S.mean(r.choices(d,k=len(d))) for _ in range(5000))
  print('delta vs',j,round(S.mean(d),5),'descriptive cluster CI',round(b[125],5),round(b[4875],5))
