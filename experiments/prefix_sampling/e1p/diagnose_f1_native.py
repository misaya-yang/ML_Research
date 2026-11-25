"""F1 implementation audit only: identical frozen design, no writes to data banks."""
import ast,fcntl,json,time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import apply_top_p,make_sampler
from e1p_sampling import preflight,MODEL,FORMATS,seed,logs,sha,lastbox_int
lock=Path('model.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
preflight()
m,tok,cfg=load(str(MODEL.resolve()),return_config=True)
assert cfg["model_type"]=="qwen3" and cfg["quantization"]["bits"]==4
E=set(tok.eos_token_ids)
doc=ast.parse(Path("e1p_sampling.py").read_text())
main=next(n for n in doc.body if isinstance(n,ast.FunctionDef) and n.name=="main")
fs={"cdf","detail","iid_pick","prompt"}
exec(compile(ast.Module(body=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in fs],type_ignores=[]),
             "e1p_sampling.py","exec"))
assert E
frozen=json.loads(Path("problems.json").read_text())["problems"]
bank={(r["id"],r["j"]):r for r in logs("iid_F1.jsonl")}
cases=[(r,j) for r in frozen[:8] for j in range(2)]
K=FORMATS["F1"][2]
def native(r,j,which):
    x=prompt(r,'F1')
    ref=bank[r["id"],j]
    assert len(x)==ref["prefill_tokens"]
    mx.random.seed(seed("native-distribution-control/v1",r['id'],j))
    key=mx.random.key(ref["seed"])
    stock=make_sampler(temp=1.,top_p=.95)
    def sampler(p):
        nonlocal key
        if which=="same_row_sampler":
            key,sub=mx.random.split(key)
            y,_=iid_pick(p,sub)
            return y
        # MLX-LM's native nucleus implementation, with temperature applied
        # before top-p to match the original F1 materialized probability law.
        return stock(p.astype(mx.float32)/.6)
    out=[];st=time.perf_counter()
    for token,_ in generate_step(mx.array(x),m,max_tokens=K,sampler=sampler):
        out.append(int(token))
        if token in E:break
    mx.synchronize()
    ans=tok.decode(out,skip_special_tokens=True)
    return {"n":len(out),"ended":out[-1] in E,"answer":lastbox_int(ans),"tokens":out,
            "text":ans,"seconds":round(time.perf_counter()-st,3),"same_as_two_slots":out==ref["token_ids"]}

print("native F1 smoke started",len(cases),"prompt cap",K,flush=True)
results=[]
for k,(r,j) in enumerate(cases):
    ref=bank[r["id"],j]
    y={"id":r["id"],"j":j,"two_slots":{"answer":ref["answer"],"n":ref["new_tokens"],"ended":not ref["capped"]},
       "native_same_row_sampler":native(r,j,"same_row_sampler"),
       "native_stock_sampler":native(r,j,"stock")}
    results.append(y)
    print(k+1,r["id"],j,y["native_same_row_sampler"]["same_as_two_slots"],y["native_stock_sampler"]["answer"],flush=True)
def counts(s):
 a=[r[s] for r in results]
 return {"cases":len(a),"boxed":sum(v["answer"] is not None for v in a),
         "capped":sum(v["n"]==K and not v["ended"] for v in a),
         "token_exact_against_two_slots":sum(v.get("same_as_two_slots",False) for v in a)}
out={"question_manifest_sha256":sha("problems.json"),"frozen_iid_sha256":sha("iid_F1.jsonl"),
     "scope":"8 fixed first questions by preregistered sort, 2 samples each; same F1 prompt and cap",
     "summary":{n:counts(n) for n in ("two_slots","native_same_row_sampler","native_stock_sampler")},"rows":results}
with Path("F1_native_implementation_check.json").open("x") as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out["summary"]),flush=True)
