"""Observable-specific finite-time profiles for IMH, with testable bin error."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
from scipy import fft,signal
from scipy.special import logsumexp,ndtr
ROOT=Path(__file__).resolve().parent
NAMES=('target mode','target surprisal','log weight','token hash')
def outputs(p,lp,lq):
    ids=np.arange(len(p))
    return [np.array(v,dtype=float) for v in (ids==np.argmax(p),-lp,lp-lq,((ids*2654435761)%(2**32))<2**31)]
def standardize(x,p):
    a=x-p@x
    return a/np.sqrt(p@(a*a))
def profile(z,rho):
    a=ndtr(z);b=np.exp(-z*z/2)/np.sqrt(2*np.pi)
    return ndtr(-z)+rho*rho*(z*b+b*b/a) if a>0 else 1.
def filt(logalpha,logt):
    """Return stationary finite average variance for one eigenvalue, Var(pi)=1."""
    la=np.minimum(0.,np.asarray(logalpha));lt=float(logt)
    if lt<1e-10:return np.ones_like(la)
    invt=np.exp(-lt);x=np.exp(np.clip(lt+la,-745,705));a=np.exp(la)
    out=np.empty_like(x);small=x<.002;large=x>55
    # Finite triangular sum expanded in alpha, valid also for small integer T.
    out[small]=(1-(1-invt**2)*x[small]/3+
                (1-invt**2)*(1-2*invt)*x[small]**2/12)
    mid=~(large|small)
    with np.errstate(divide='ignore',invalid='ignore'):
        # log1p(-alpha)/alpha is well-conditioned except at exact zero.
        a1=a[mid];rel=np.divide(np.log1p(-a1),a1,out=-np.ones_like(a1),where=a1>0)
        out[mid]=(2/x[mid]-invt+2*(1-a1)*np.expm1(x[mid]*rel)/x[mid]**2)
    out[large]=(2/x[large]-invt)-2*(1-a[large])/x[large]**2
    return np.clip(out,0,1)
def table(logp,logq,d,step):
    p=np.exp(logp);s=logp-logq
    labels=np.rint(s/step).astype(int);start=labels.min();labels-=start
    n=labels.max()+1;span=d*(n-1)+1;nf=fft.next_fast_len(span,real=True)
    def transform(values):return fft.rfft(np.bincount(labels,weights=p*values,minlength=n),nf)
    def inv(x):return fft.irfft(x,nf)[:span]
    a=transform(np.ones_like(p));prefix=a**(d-2)
    prob=inv(prefix*a*a)
    levels=(d*start+np.arange(span))*step
    # Omit numerically invisible extreme tails; report removed algebraic masses.
    good=prob>1e-15
    err=float(np.abs(prob[~good]).sum())
    pr=np.maximum(0,prob);pr/=pr.sum()
    F=np.cumsum(pr);upper=signal.lfilter([1],[1,-np.exp(-step)],pr[::-1])[::-1]-pr
    logalpha=np.log(np.maximum(F+upper,1e-300))-levels
    out=[]
    for y in outputs(p,logp,logq):
        h=standardize(y,p);b=transform(h);c=transform(h*h)
        m1=inv(np.sqrt(d)*b*prefix*a);m2=inv(c*prefix*a+(d-1)*b*b*prefix)
        cdf=np.cumsum(m2)-np.cumsum(m1)**2/np.maximum(F,1e-290)
        cdf[F<1e-12]=0
        weight=np.diff(np.r_[0.,cdf]);bad=float(-weight[weight<0].sum())
        weight=np.maximum(weight,0);weight[~good]=0
        total=float(weight.sum());weight/=total
        out.append((weight,logalpha,float(abs(1-total)+bad+err)))
    return out
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=float,default=.01)
    parser.add_argument('--input-dir',type=Path,required=True,help='Directory containing archived p.npz and q.npz')
    parser.add_argument('--ids',default='0:32');parser.add_argument('--out',default='gpt-profiles.csv')
    opts=parser.parse_args();i,j=map(int,opts.ids.split(':'))
    if Path(opts.out).name != opts.out or opts.out in ('', '.', '..'):
        parser.error('--out must be a filename, not a path')
    sys.path.insert(0,str(ROOT.parent/'sources'))
    from runtime_paths import new_run_dir
    out_dir=new_run_dir('ewt-legacy-risk-profile')
    lp=np.load(opts.input_dir/'p.npz')['logits'].astype(float);lq=np.load(opts.input_dir/'q.npz')['logits'].astype(float)
    lp-=logsumexp(lp,axis=1)[:,None];lq-=logsumexp(lq,axis=1)[:,None];rows=[]
    for row in range(i,j):
        p=np.exp(lp[row]);s=lp[row]-lq[row];kl=p@s;sd=np.sqrt(p@((s-kl)**2))
        obs=outputs(p,lp[row],lq[row])
        for d in (4,16,64,256):
            block=table(lp[row],lq[row],d,opts.step)
            for obsname,y,(mass,la,noise) in zip(NAMES,obs,block):
                corr=float((p@(standardize(y,p)*(s-kl)))/sd)
                for z in (-1.,0.,1.):
                    logt=max(0,d*kl+z*np.sqrt(d)*sd)
                    if logt<36:logt=np.log(max(1,int(np.rint(np.exp(logt)))))
                    zeff=(logt-d*kl)/(sd*np.sqrt(d));delta=opts.step*d/2
                    exact=float(mass@filt(la,logt))
                    rows.append(dict(doc=900+row,split='dev' if row<8 else 'test',d=d,h=obsname,z=z,
                           logT=logt,kl=kl,sd=sd,rho=corr,estimate=profile(zeff,corr),blind=ndtr(-zeff),
                           ref=exact,quant_lo=float(mass@filt(la+delta,logt)),quant_hi=float(mass@filt(la-delta,logt)),floating_resid=noise))
        print('computed',row,flush=True)
    with (out_dir/opts.out).open('x',newline='') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
