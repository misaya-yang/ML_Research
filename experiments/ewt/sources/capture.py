from pathlib import Path
import os,sys,csv,ctypes,subprocess
import numpy as np
from scipy.special import softmax
HERE=Path(__file__).resolve().parent
from model import Model,read
from runtime_paths import DATA,new_run_dir
import model
model.ROOT=Path(os.environ.get('EWT_DATA', DATA))
SO=Path(os.environ.get('TMPDIR','/tmp'))/'imh_disclosure_ewt_cold_scored.so'
if not SO.exists():subprocess.run(['gcc','-shared','-fPIC','-O3',str(HERE/'draw.c'),'-o',str(SO)],check=True)
fn=ctypes.CDLL(str(SO)).paired; datptr=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');bptr=np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')
fn.restype=None;fn.argtypes=[ctypes.c_int]*3+[datptr]*5+[ctypes.c_int]*2+[datptr,bptr]
def heldout():
    docs=[];seen=set()
    for r in read('test')[::-1]:
        key=r[0].rsplit('-',1)[0]
        if not 24<=len(r[1])<=31 or key in seen:continue
        docs.append(r);seen.add(key)
        if len(docs)==24:break
    return docs
class Draw:
    def __init__(self,p,q,law,pos,gold):
        self.law=law;self.back=np.ascontiguousarray(law.back);self.e=np.ascontiguousarray(p.e-q.e);self.a=np.ascontiguousarray(p.a-q.a);self.pos=pos;self.gold=gold
    def sample(self,rng,n):
        s=np.empty(n);h=np.empty(n,np.uint8);law=self.law
        for k in range(0,n,8192):
            count=min(n-k,8192);unif=rng.random((count,law.n))
            fn(count,law.n,law.D,law.last,self.back,unif,self.e,self.a,self.pos,self.gold,s[k:k+count],h[k:k+count])
        return s,h
def generate():
    out_dir=new_run_dir('ewt-capture')
    rows=[];m=Model(read('train'));docs=heldout();(out_dir/'pools').mkdir(exist_ok=False)
    for d,(doc,ws,gs) in enumerate(docs):
        p=m.law(ws);G=np.array([m.ix[u] for u in gs]);em=softmax(p.e,axis=1)[np.arange(len(G)),G];j=int(np.argmin(abs(em-.5)));mu=p.moments()[0][j,G[j]]
        for z,(aa,ee) in enumerate(((.55,1.),(1.,.6))):
            q=m.law(ws,aa,ee);pdraw=Draw(p,q,p,j,G[j]);qdraw=Draw(p,q,q,j,G[j])
            rng=np.random.default_rng(168877+53*d+z)
            ps,ph=pdraw.sample(rng,1280);qs,qh=qdraw.sample(rng,65536)
            score_rows=(ps,ph,qs,qh);f=out_dir/'pools'/f'{d}-{z}.npz'
            np.savez_compressed(f,ps=ps,ph=ph,qs=qs,qh=qh)
            test=np.random.default_rng(90991+53*d+z);N=4096;L=160
            s0,h0=pdraw.sample(test,N)
            sumq=np.zeros((N,L+1),np.uint16);sump=sumq.copy()
            for r in range(0,N,128):
                s,h=qdraw.sample(test,128*L);s=s.reshape(128,L);h=h.reshape(128,L)
                start=s[:,0].copy();v=h[:,0].copy();stat=s0[r:r+128].copy();vs=h0[r:r+128].copy()
                sumq[r:r+128,1]=v;sump[r:r+128,1]=vs
                for t in range(1,L):
                    u=np.log(test.random(128));take=u<s[:,t]-start
                    start[take]=s[take,t];v[take]=h[take,t];ts=u<s[:,t]-stat
                    stat[ts]=s[ts,t];vs[ts]=h[ts,t]
                    sumq[r:r+128,t+1]=sumq[r:r+128,t]+v;sump[r:r+128,t+1]=sump[r:r+128,t]+vs
            np.savez_compressed(out_dir/'pools'/f'traj-{d}-{z}.npz',q=sumq,p=sump)
            rows.append(dict(d=d,z=z,sent_id=doc,n=len(ws),position=j,gold=gs[j],mu=float(mu),logZ=float(p.z-q.z),qmean=float(q.moments()[0][j,G[j]])))
            with (out_dir/'evaluation-only.csv').open('w',newline='') as out:
                w=csv.DictWriter(out,rows[0]);w.writeheader();w.writerows(rows)
            print(d,z,doc,flush=True)
if __name__=='__main__':generate()
