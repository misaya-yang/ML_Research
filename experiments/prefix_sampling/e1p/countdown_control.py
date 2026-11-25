"""Countdown implementation control on the existing checkpoint, separate from frozen E1-P."""
import ast,fcntl,hashlib,json,math,random,re,time
from collections import Counter,deque
from fractions import Fraction
from pathlib import Path
from e1p_sampling import MODEL,preflight,seed,sha,write_frozen
from e1p_lattice import advance,fp32_floor

def grade(s,numbers,target):
    line=s.strip().splitlines()[0].strip().strip('`$`.') if s.strip() else ''
    m=re.fullmatch(r'([0-9+*/() \-]+)(?:\s*=\s*([+-]?\d+))?',line)
    if not m:return False,False,line
    used=Counter()
    def evaluate(t):
        if isinstance(t,ast.Constant) and type(t.value)==int:
            used[t.value]+=1;return Fraction(t.value)
        if isinstance(t,ast.BinOp) and type(t.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div):
            a,b=evaluate(t.left),evaluate(t.right)
            if isinstance(t.op,ast.Add):return a+b
            if isinstance(t.op,ast.Sub):return a-b
            if isinstance(t.op,ast.Mult):return a*b
            return a/b
        raise ValueError("not arithmetic")
    try:val=evaluate(ast.parse(m[1].strip(),mode="eval").body)
    except (SyntaxError,ValueError,ZeroDivisionError,RecursionError):return False,False,line
    return True,used==Counter(numbers) and val==target and (m[2] is None or int(m[2])==target),line

assert grade('43+57-11',[57,43,11],89)[:2]==(True,True)
assert grade('19*8-11 = 141',[19,8,11],141)[:2]==(True,True)
assert not grade('43+46',[57,43,11],89)[1]
assert not grade('(43+57)/(11-11)',[57,43,11],89)[1]
r=random.Random(seed('countdown-fallback-v1'))
tests=[];unique=set()
while len(tests)<32:
    a,b,c=r.sample(range(10,89),3)
    target=a+b-c
    if not 10<=target<=99:continue
    ns=sorted((a,b,c));k=(*ns,target)
    if k in unique:continue
    unique.add(k);r.shuffle(ns)
    j=len(tests)
    tests.append({"id":f"cd{j:02}","numbers":ns,"target":target,"set":"pilot" if j<8 else "confirmation"})
F=Path('countdown_control_frozen.json')
write_frozen(F,{"checkpoint":"cached Qwen3-1.7B 4-bit","not_original_checkpoint":True,"source":"https://arxiv.org/html/2607.01179v1",
   "temperature":1.,"top_p":.95,"cap":32,"data":tests,"contract":"synthetic three-number +/- generation with mixed allowed operations, strict arithmetic multiset; separate local control, not paper's SFT model or original data",
   "gate":"Pilot only: >=40% parsed; .10<= pass@1 <=.80. If passed, fixed confirm 24 questions, 2 independent groups of 8 per arm.",
   "prompt":"two dialogue examples then native nonthinking chat, assistant prefill Equation: form"})
start=time.monotonic()
lk=Path('model.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);preflight()
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
import numpy as np
model,tok,cfg=load(str(MODEL.resolve()),return_config=True)
E=set(tok.eos_token_ids)
@mx.compile
def choose(lp,u):
    order=mx.argsort(-lp,axis=-1)
    p=mx.softmax(mx.take_along_axis(lp,order,axis=-1),axis=-1)
    p=p*((mx.cumsum(p,axis=-1)-p)<.95)
    p=p/mx.sum(p,axis=-1,keepdims=True)
    cm=mx.cumsum(p,axis=-1)
    k=mx.minimum((cm<=u).sum(axis=-1,keepdims=True),(p>0).sum(axis=-1,keepdims=True)-1)
    tok=mx.take_along_axis(order,k,axis=-1).reshape(-1)
    hi=mx.take_along_axis(cm,k,axis=-1).reshape(-1)
    lo=mx.where(k.reshape(-1)>0,mx.take_along_axis(cm,mx.maximum(k-1,0),axis=-1).reshape(-1),0.)
    pro=mx.take_along_axis(p,k,axis=-1).reshape(-1)
    return tok,mx.stack([lo,hi,pro],axis=-1)
RULE="Use each listed number exactly once, using +, -, *, / and parentheses, to make the target. Reply with the arithmetic expression only."
def prompt(row):
    msg=[{"role":"system","content":RULE},
         {"role":"user","content":"Target: 89\nNumbers: [57, 43, 11]"},
         {"role":"assistant","content":"Equation: 43+57-11"},
         {"role":"user","content":"Target: 81\nNumbers: [25, 14, 70]"},
         {"role":"assistant","content":"Equation: 70+25-14"},
         {"role":"user","content":f'Target: {row["target"]}\nNumbers: {row["numbers"]}'}]
    text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return tok.encode(text+'Equation: ',add_special_tokens=False)

def sampler(row,j,arm):
    rng=random.Random(seed("countdown-poscontrol/v1",row["id"],j,arm))
    d=random.Random(seed("countdown-lattice-shift",row["id"],j//8)).random()
    b=int(f"{j%8:03b}"[::-1],2);u=(d+b/8)%1
    bits=0.;pos=0;refresh=None
    def draw(lp):
        nonlocal u,bits,pos,refresh
        pos+=1
        if arm=="iid":t,_=choose(lp,mx.array(fp32_floor(rng.random()),mx.float32)); return t
        t,v=choose(lp,mx.array(fp32_floor(u),mx.float32))
        lo,hi,p=v.tolist()[0]
        u,bits,f=advance(u,lo,hi,p,bits,rng)
        if f and refresh is None:refresh=pos
        return t
    return draw,lambda:refresh
filepath=Path('countdown_control_samples.jsonl')
def go(tests,n,arm):
    rows=[json.loads(l) for l in filepath.read_text().splitlines()] if filepath.exists() else []
    old={(r["id"],r["arm"],r["j"]):r for r in rows}
    tasks=deque((r,j) for j in range(n) for r in tests if (r["id"],arm,j) not in old)
    rmap={}
    g=BatchGenerator(model,completion_batch_size=2,prefill_batch_size=2,stop_tokens=[[int(x)] for x in sorted(E)])
    try:
      with filepath.open("a",buffering=1) as f:
        while tasks or rmap:
            new=[]
            while tasks and len(rmap)+len(new)<2:
                row,j=tasks.popleft();new.append((row,j,sampler(row,j,arm),[]))
            if new:
                ids=g.insert([prompt(x[0]) for x in new],[32]*len(new),samplers=[x[2][0] for x in new])
                rmap.update(zip(ids,new))
            z=g.next_generated()
            if not z:raise RuntimeError("Missing responses")
            for ev in z:
                row,j,s,ids=rmap[ev.uid];ids.append(int(ev.token))
                if ev.finish_reason is None:continue
                del rmap[ev.uid];assert len(ids)<=32
                txt=tok.decode(ids,skip_special_tokens=True)
                fmt,yes,line=grade(txt,row["numbers"],row["target"])
                obj={"id":row["id"],"arm":arm,"j":j,"correct":yes,"parsed":fmt,"text":txt,"expression":line,
                     "n":len(ids),"tokens":ids,"finish":ev.finish_reason,"refresh_step":s[1]()}
                f.write(json.dumps(obj,ensure_ascii=False)+"\n");f.flush()
                old[row["id"],arm,j]=obj
    finally:g.close()
    return [[old[r["id"],arm,j] for j in range(n)] for r in tests]
def summary(raw):
    xs=[x for a in raw for x in a];n=len(xs)
    return {"questions":len(raw),"draws":n,"parsed":sum(x["parsed"] for x in xs)/n,
            "correct":sum(x["correct"] for x in xs)/n,
            "mean_tokens":sum(x["n"] for x in xs)/n,"cap":sum(x["finish"]=="length" for x in xs)/n}
pilot=go(tests[:8],8,"iid");a=summary(pilot);print("pilot",json.dumps(a),flush=True)
out={"pilot":a,"control_not_exact_replication":True,"frozen_sha256":sha(F)}
if .10<=a["correct"]<=.80 and a["parsed"]>=.40:
    A=go(tests[8:],16,"iid");B=go(tests[8:],16,"lattice")
    out["iid"]=summary(A);out["lattice"]=summary(B);out["pass"]={}
    rng=np.random.default_rng(seed("countdown-question-bootstrap"))
    for k in (2,4,8):
        na=16
        def pp(a):
            f=na-sum(x["correct"] for x in a)
            return 1-(math.comb(f,k)/math.comb(na,k) if f>=k else 0)
        q=np.array([pp(x) for x in A])
        p=np.array([sum(any(z["correct"] for z in xs[j:j+k]) for j in range(0,16,k))/(16/k) for xs in B])
        d=p-q;t=rng.integers(len(d),size=(10000,len(d)))
        out["pass"][str(k)]={"iid":float(q.mean()),"lattice":float(p.mean()),"gain":float(d.mean()),
                             "paired_ci":np.quantile(d[t].mean(1),[.025,.975]).tolist()}
else:out["gate_failed"]="Pilot cannot serve as a positive control on this checkpoint; no confirmation sampler run."
out["wall_s"]=time.monotonic()-start
Path('countdown_control_report.json').write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out),flush=True)
