"""Supervised first-order tag potentials and exact conditional simulation."""
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parent.parent/"data"
def read(split):
    records=[]
    for sent in (ROOT/f'en_ewt-ud-{split}.conllu').read_text().strip().split('\n\n'):
        rows=[line.split('\t') for line in sent.splitlines() if line.split('\t')[0].isdigit()]
        key=next((line.partition(' = ')[2] for line in sent.splitlines() if line.startswith('# sent_id')),'')
        if rows:records.append((key,[r[1].lower() for r in rows],[r[3] for r in rows]))
    return records
class Model:
    def __init__(self,records):
        self.tags=sorted({t for _,_,ts in records for t in ts}); self.ix={x:i for i,x in enumerate(self.tags)}
        freq=Counter(w for _,ws,_ in records for w in ws)
        self.lex={w for w,n in freq.items() if n>=4}
        counts=Counter(); D=len(self.tags); a=np.full((D,D),.5); begin=np.full(D,.5)
        for _,ws,ts in records:
            tags=[self.ix[t] for t in ts];begin[tags[0]]+=1
            for u,v in zip(tags,tags[1:]):a[u,v]+=1
            for w,v in zip(ws,tags):counts[(self.token(w),v)]+=1
        keys=sorted({k for k,_ in counts}|{'[other]'});self.lexid={k:i for i,k in enumerate(keys)}
        e=np.full((len(keys),D),.03)
        for (k,j),n in counts.items():e[self.lexid[k],j]+=n
        self.em=np.log(e/e.sum(axis=0));self.a=np.log(a/a.sum(1)[:,None]);self.b=np.log(begin/begin.sum())
    def token(self,w):
        if w in self.lex:return w
        if any(x.isdigit() for x in w): return '[number]'
        return '[suffix]'+w[-2:]
    def law(self,ws,aa=1.,ee=1.):
        e=self.em[[self.lexid.get(self.token(w),self.lexid['[other]']) for w in ws]]*ee
        return Law(e,self.a*aa,self.b)
class Law:
    def __init__(self,e,a,b):
        self.e=e;self.a=a;self.b=b;fw=[b+e[0]]
        for row in e[1:]:fw.append(row+logsumexp(fw[-1][:,None]+a,axis=0))
        self.fw=np.stack(fw);self.z=logsumexp(fw[-1]);self.n=len(e);self.D=len(b)
        self.last=np.cumsum(np.exp(fw[-1]-self.z))
        self.back=[]
        for row in fw[:-1]:
            v=row[:,None]+a;self.back.append(np.cumsum(np.exp(v-logsumexp(v,axis=0)),axis=0).T.copy())
    def draw(self,rng,n):
        x=np.empty((n,self.n),dtype=np.int16)
        x[:,-1]=np.minimum(np.searchsorted(self.last,rng.random(n)),self.D-1)
        for j in range(self.n-2,-1,-1):
            c=self.back[j][x[:,j+1]]
            x[:,j]=np.minimum((c<rng.random((n,1))).sum(1),self.D-1)
        return x
    def logp(self,x):
        return self.b[x[:,0]]+np.sum(self.e[np.arange(self.n),x],axis=1)+np.sum(self.a[x[:,:-1],x[:,1:]],axis=1)-self.z
    def moments(self):
        beta=np.zeros(self.D);marg=[];pairs=[]
        for j in range(self.n-1,-1,-1):
            marg.append(np.exp(self.fw[j]+beta-self.z))
            if j:
                joint=self.fw[j-1][:,None]+self.a+self.e[j][None,:]+beta[None,:]-self.z
                pairs.append(np.exp(joint));beta=logsumexp(self.a+self.e[j][None,:]+beta[None,:],axis=1)
        return np.array(marg[::-1]),np.array(pairs[::-1])
def kl(p,q):
    m,a=p.moments()
    return float((m*(p.e-q.e)).sum()+(a*(p.a-q.a)).sum()+q.z-p.z)
def values(x,G,nouns):
    return np.stack([(x==G).mean(1),np.isin(x,nouns).mean(1),(x[:,1:]!=x[:,:-1]).mean(1)],axis=1)
