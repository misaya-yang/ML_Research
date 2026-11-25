import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from e1p_sampling import lastbox_int,logs,sha,FORMATS
p=Path("../ancestry-budget-20261125/models/qwen3-1.7b-4bit")
t=AutoTokenizer.from_pretrained(p,local_files_only=True)
rows=logs("iid_F1.jsonl");problems={r["id"]:r for r in json.load(open("problems.json"))["problems"]}
assert len(rows)==2048 and len(problems)==64
src=Path("../native-smc-continuation-20261125/MATH500_test_original.jsonl")
eligible=sorted((x for x in logs(src) if x["level"] in [2,3,4,5] and re.fullmatch(r"-?\d+",str(x["answer"]))),
                  key=lambda x:hashlib.sha256(("e1p-v1/"+x["unique_id"]).encode()).hexdigest())[:64]
assert [x["unique_id"] for x in eligible]==list(problems)
assert all(str(problems[x["unique_id"]]["gold"])==str(x["answer"]) for x in eligible)
for s,a in [(r'\boxed{-15}',-15),(r'\boxed{12,319}',12319),(r'\boxed{42}',42),
      (r'\boxed{42} \boxed{\frac{1}{3}}',None),(r'\boxed{42',None)]:
    assert lastbox_int(s)==a
chars=[]
for i,r in problems.items():
    mode,suffix,cap,n=FORMATS["F1"]
    msgs=[{"role":"user","content":r["problem"]+"\n\n"+suffix}]
    x=t.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=mode)
    assert x.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids=t.encode(x,add_special_tokens=False)
    chars.append((i,len(ids)))
pids=dict(chars)
text=t.batch_decode([r["token_ids"] for r in rows],skip_special_tokens=True)
assert all(r["prefill_tokens"]==pids[r["id"]] and r["finish_reason"]=="length" and r["new_tokens"]==len(r["token_ids"])==24 for r in rows)
assert all(lastbox_int(s)==r["answer"] for r,s in zip(rows,text))
gen=defaultdict(set);nobox=Counter()
for r,s in zip(rows,text):gen[r["id"]].add(tuple(r["token_ids"]))
max_interval_gap=0.
for r in rows:
    with np.load(r["npz"]) as a:
        p=a["prob"];l=a["cdf_lo"];u=a["cdf_hi"];h=a["entropy"]
        assert all(v.shape==(24,) for v in (p,l,u,h))
        assert np.all(np.isfinite(p)) and np.all(p>0) and np.all(np.isfinite(h)) and np.all(h>=-1e-5)
        max_interval_gap=max(max_interval_gap,float(np.max(np.abs((u-l)-p))))
print(json.dumps({"rows":len(rows),"prompt_lengths_matched":True,"nonthinking_template_confirmed":True,
                  "decoded_containing_box_marker":sum("\\boxed" in s for s in text),
                  "decoded_with_any_balanced_box":sum(bool(re.search(r"\\boxed\s*\{[^{}]*\}",s)) for s in text),
                  "unique_sequences_per_question_range":[min(map(len,gen.values())),max(map(len,gen.values()))],
                  "unique_sampling_seeds":len({r["seed"] for r in rows}),
                  "cdf_interval_max_rounding_difference":max_interval_gap,
                  "opening_histogram":Counter(s.split(' ')[0] for s in text).most_common(8),
                  "raw_examples":[text[0],text[1],text[2],text[3]]},ensure_ascii=False))
