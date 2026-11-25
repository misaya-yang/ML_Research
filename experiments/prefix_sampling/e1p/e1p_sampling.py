"""E1-P native Qwen3 sample collection. Resumable and single model job."""
import argparse,fcntl,hashlib,json,math,os,re,subprocess,time
from collections import deque
from e1p_lattice import advance,fp32_floor
from pathlib import Path

FORMATS={"F1":(False,"Give only the final answer in \\boxed{}.",24,32),
         "F2":(False,"Reason briefly, then put the final answer in \\boxed{}.",1024,32),
         "F3":(True,"Please reason step by step, and put your final answer within \\boxed{}.",8192,24)}
SRC=Path('../native-smc-continuation-20261125/MATH500_test_original.jsonl')
MODEL=Path('../ancestry-budget-20261125/models/qwen3-1.7b-4bit')
REV='3b1b1768f8f8cf8351c712464f906e86c2b8269e'
MASK=2**32-1
def seed(*p):return int.from_bytes(hashlib.sha256(str(p).encode()).digest()[:8],"big")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_frozen(p,data):
    p=Path(p);s=json.dumps(data,ensure_ascii=False,indent=2)+"\n"
    if p.exists():
        if p.read_text()!=s:raise ValueError(f"Frozen file differs: {p}")
    else:
        with p.open("x") as f:f.write(s)
def logs(p):
    if not Path(p).exists():return []
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def boxnum(s):
    box=re.findall(r'\\boxed\s*\{\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*\}',s)
    if not box:return None
    return int(box[-1].replace(",",""))
def lastbox_int(s):
    # Don't reward an older integer box if the last balanced box is non-integer.
    starts=list(re.finditer(r'\\boxed\s*\{',s))
    for b in reversed(starts):
        d=1
        for k in range(b.end(),len(s)):
            d+=(s[k]=='{')-(s[k]=='}')
            if d==0:return boxnum(s[b.start():k+1])
    return None
def preflight():
    table=subprocess.run(['/bin/ps','-ax','-o','pid=,comm=,args='],capture_output=True,text=True,errors='replace',check=True).stdout
    markers=("gate1_round2.py calibrate","gate1_profile_mlx.py profile","gate1_profile_mlx.py heldout",
             "mlx_parallel_calibrate.py calibrate","collect_fork_bank.py --","collect_q3_finalization.py --",
             "collect_q3_empty_final.py --","e1p_sampling.py collect")
    active=[]
    for l in table.splitlines():
        parts=l.split(None,2)
        if len(parts)!=3 or int(parts[0])==os.getpid() or not any(m in parts[2] for m in markers):continue
        if "python" in Path(parts[1]).name.lower() and ("mlx_env" in parts[2] or "envs/" in parts[2]):active.append(parts[0])
    if active:raise SystemExit(f"Existing model job: {active}")
    res=subprocess.run(['/usr/bin/memory_pressure','-Q'],capture_output=True,text=True,check=True).stdout
    pc=re.search(r'System-wide memory free percentage:\s*(\d+)%',res)
    size=subprocess.check_output(['/usr/sbin/sysctl','-n','hw.memsize'],text=True)
    assert pc and int(size)*int(pc[1])/100>=6*2**30,res
    print(res.strip(),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("freeze","collect"))
    p.add_argument("--arm",choices=("iid","lattice"),default="iid")
    p.add_argument("--formats",nargs="+",choices=tuple(FORMATS),default=["F1","F2"])
    p.add_argument("--exactness",action="store_true",help="Separate F1 distribution check on five disjoint questions.")
    p.add_argument("--limit",type=int,help="Explicit smoke sample limit; use a separate output directory.")
    a=p.parse_args()
    problems=sorted((r for r in logs(SRC) if r["level"] in (2,3,4,5) and re.fullmatch(r'-?\d+',str(r['answer']))),
                    key=lambda r:hashlib.sha256(("e1p-v1/"+r["unique_id"]).encode()).hexdigest())[:64]
    assert len(problems)==64 and len({r['unique_id'] for r in problems})==64
    write_frozen("problems.json",{"sha256_source":sha(SRC),"selection":"e1p-v1/ + unique_id","model_revision":REV,
                    "problems":[{"id":r["unique_id"],"level":r["level"],"problem":r["problem"],"gold":int(r["answer"])} for r in problems]})
    if a.action=="freeze":print("frozen",len(problems),sha("problems.json"));return
    lock=Path("model.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.exactness:
        assert a.formats==["F1"] and a.limit is None
        seen={r["unique_id"] for r in problems}
        dq=sorted((r for r in logs(SRC) if r["unique_id"] not in seen and r["level"] in (2,3,4,5)
                   and re.fullmatch(r'-?\d+',str(r["answer"]))),
                  key=lambda r:hashlib.sha256(("e1p-exactness-v1/"+r["unique_id"]).encode()).hexdigest())[:5]
        assert len(dq)==5
        checked=[{"id":r["unique_id"],"level":r["level"],"problem":r["problem"],"gold":int(r["answer"])} for r in dq]
        write_frozen("problems_exactness.json",{"problems":checked,"study_sha256":sha("problems.json")})
    if a.arm=="lattice" and not a.exactness:
        f=Path("predictions_F3.json" if "F3" in a.formats else "predictions_F1F2.json")
        assert f.is_file() and Path("prediction_seals.json").is_file(),"Predictions and their hash must precede lattice sampling."
        seals=json.loads(Path("prediction_seals.json").read_text())
        assert seals.get(f.name)==sha(f),"Prediction file hash mismatch."
        assert json.loads(f.read_text()).get("iid_sha256")=={t:sha(f"iid_{t}.jsonl") for t in a.formats},"IID banks differ from prediction inputs."
    preflight()
    if a.arm=="lattice":assert json.loads(Path("lattice_toy_validation.json").read_text())["passed"]
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import apply_top_p
    model,tok,cfg=load(str(MODEL.resolve()),return_config=True)
    assert cfg.get("quantization",{}).get("bits")==4 and cfg["model_type"]=="qwen3"
    Es=set(tok.eos_token_ids)
    assert Es
    print("model ready",cfg["model_type"],cfg["quantization"],"MLX",mx.default_device(),"EOS",sorted(Es),flush=True)

    @mx.compile
    def cdf(lp):
        ranks=mx.argsort(-lp.astype(mx.float32),axis=-1)
        scores=mx.take_along_axis(lp.astype(mx.float32)/.6,ranks,axis=-1)
        probs=mx.softmax(scores,axis=-1)
        p0=mx.cumsum(probs,axis=-1)
        kept=(p0-probs)<.95
        probs=probs*kept
        cm=mx.cumsum(probs,axis=-1)
        probs=probs/cm[...,-1:]
        cm=mx.cumsum(probs,axis=-1)
        entropy=-mx.sum(mx.where(probs>0,probs*mx.log(mx.maximum(probs,1e-30)),0),axis=-1)
        return ranks,probs,cm,entropy
    @mx.compile
    def detail(ranks,probs,cm,h,idx):
        idx=idx.reshape(-1,1)
        p=mx.take_along_axis(probs,idx,axis=-1).reshape(-1)
        hi=mx.take_along_axis(cm,idx,axis=-1).reshape(-1)
        lo=mx.where(idx.reshape(-1)>0,mx.take_along_axis(cm,mx.maximum(idx-1,0),axis=-1).reshape(-1),0)
        token=mx.take_along_axis(ranks,idx,axis=-1).reshape(-1)
        return token,mx.stack((p,lo,hi,h),axis=-1)
    @mx.compile
    def iid_pick(lp,key):
        ranks,p,cm,h=cdf(lp)
        loc=mx.random.categorical(mx.where(p>0,mx.log(mx.maximum(p,1e-30)),-float("inf")),key=key)
        return detail(ranks,p,cm,h,loc)
    @mx.compile
    def lattice_pick(lp,u):
        ranks,p,cm,h=cdf(lp)
        j=mx.minimum(mx.sum(cm<=u,axis=-1),mx.sum(p>0,axis=-1)-1)
        return detail(ranks,p,cm,h,j)

    import random
    def state(fmt,i,j):
        sd=seed("e1p-iid-v1",fmt,i,j); group=j//8;point=j%8
        delta=random.Random(seed("e1p-shared",fmt,i,group)).random() if a.arm=="lattice" else None
        u=((int(f"{point:03b}"[::-1],2)/8+delta)%1) if delta is not None else None
        rng=random.Random(seed("e1p-refresh",fmt,i,j)); key=mx.random.key(sd)
        cache=deque();bits=0.;refresh=None;calls=0
        def draw(lp):
            nonlocal key,u,bits,refresh,calls
            if a.arm=="iid":
                key,sub=mx.random.split(key); t,y=iid_pick(lp,sub)
                mx.async_eval(t,y)
            else:
                t,y=lattice_pick(lp,mx.array(fp32_floor(u),mx.float32))
                row=y.tolist()[0]
                if row[0]<=0:raise RuntimeError("Zero-probability sampled token")
                u,bits,refreshed=advance(u,row[1],row[2],row[0],bits,rng)
                if refreshed and refresh is None:refresh=calls+1
            calls+=1;cache.append((t,y));return t
        return draw,cache,sd,group,point,delta,lambda:refresh
    def prompt(row,fmt):
        mode,suffix,_,_=FORMATS[fmt]
        msg=[{"role":"user","content":row["problem"]+"\n\n"+suffix}]
        x=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=mode)
        if isinstance(x,dict) or hasattr(x,"keys"):x=x["input_ids"]
        if hasattr(x,"tolist"):x=x.tolist()
        if x and isinstance(x[0],list):x=x[0]
        assert x and all(isinstance(t,int) for t in x)
        return x
    for fmt in a.formats:
        suffix=FORMATS[fmt];cap,n=suffix[-2:]
        picked=json.loads(Path("problems.json").read_text())["problems"]
        if fmt=="F3":
            subset=json.loads(Path("problems_F3.json").read_text())["ids"]
            picked=[x for x in picked if x["id"] in set(subset)]
            assert len(picked)==32
        if a.exactness:picked=checked;n=400
        ns="exactness_" if a.exactness else ""
        target=Path(f"{ns}{a.arm}_{fmt}.jsonl")
        seen={r["sample_id"] for r in logs(target)}
        print(fmt,a.arm,len(picked),cap,n,"completed",len(seen),flush=True)
        todo=deque((r,j) for j in range(n) for r in picked if f'{fmt}:{a.arm}:{r["id"]}:{j}' not in seen)
        if a.limit is not None:todo=deque(list(todo)[:a.limit])
        arraydir=Path("arrays")/(ns+fmt)/a.arm;arraydir.mkdir(parents=True,exist_ok=True)
        arrpath=lambda sid: arraydir/(hashlib.sha256(sid.encode()).hexdigest()+".npz")
        g=BatchGenerator(model,completion_batch_size=2,prefill_batch_size=2,stop_tokens=[[int(x)] for x in sorted(Es)])
        live={}
        with target.open("a",buffering=1) as dest:
          try:
            while todo or live:
                tasks=[]
                while todo and len(live)+len(tasks)<2:
                    r,j=todo.popleft();x=prompt(r,fmt);st=state(fmt,r["id"],j)
                    tasks.append((r,j,x,st))
                if tasks:
                    uu=g.insert([x[2] for x in tasks],[cap]*len(tasks),samplers=[x[3][0] for x in tasks])
                    now=time.perf_counter()
                    for uid,x in zip(uu,tasks):live[uid]=(x,now,[],[])
                events=g.next_generated()
                if not events:raise RuntimeError("No model responses")
                for e in events:
                    task,ts,ids,arr=live[e.uid];r,j,inp,st=task
                    tr,ar=st[1].popleft();v=ar.tolist()[0]
                    assert tr.item()==int(e.token) and v[0]>0 and 0<=v[1]<=v[2]<=1.00001
                    ids.append(int(e.token));arr.append(v)
                    if e.finish_reason is None:continue
                    del live[e.uid]
                    assert len(ids)<=cap
                    sid=f'{fmt}:{a.arm}:{r["id"]}:{j}'
                    fp=arrpath(sid);tmp=fp.with_suffix(".new.npz")
                    b=np.asarray(arr,dtype=np.float32);assert b.shape==(len(ids),4)
                    np.savez_compressed(tmp,prob=b[:,0],cdf_lo=b[:,1],cdf_hi=b[:,2],entropy=b[:,3])
                    os.replace(tmp,fp)
                    answer=lastbox_int(tok.decode(ids,skip_special_tokens=True))
                    row={"sample_id":sid,"id":r["id"],"format":fmt,"arm":a.arm,"j":j,"token_ids":ids,"new_tokens":len(ids),
                         "prefill_tokens":len(inp),"finish_reason":e.finish_reason,"answer":answer,"correct":answer==r["gold"],
                         "capped":e.finish_reason=="length","group":st[3],"point":st[4],"delta":st[5],
                         "seed":st[2],"refresh_step":st[6]() if st[6]() is not None and st[6]()<=len(ids) else None,
                         "seconds":round(time.perf_counter()-ts,3),"npz":str(fp)}
                    print(json.dumps(row,ensure_ascii=False),file=dest,flush=True)
                    seen.add(sid)
                    if len(seen)<=4 or len(seen)%32==0:print(fmt,a.arm,len(seen),"/",len(picked)*n,flush=True)
          finally:g.close()
    print("collection complete",a.arm,a.formats,flush=True)

if __name__=="__main__":main()
