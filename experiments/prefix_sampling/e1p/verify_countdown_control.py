"""Regrade completed Countdown controls without loading a model."""
import ast,json,math,re
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
s=Path("countdown_control.py")
exec(compile(ast.Module([x for x in ast.parse(s.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=="grade"],[]),str(s),"exec"))
p=json.loads(Path("countdown_control_frozen.json").read_text())
check={r["id"]:r for r in p["data"]}
rows=[json.loads(l) for l in Path("countdown_control_samples.jsonl").read_text().splitlines()]
report=json.loads(Path("countdown_control_report.json").read_text())
assert len(rows) in (64,832) and len({(r["id"],r["arm"],r["j"]) for r in rows})==len(rows)
for r in rows:
    q=check[r["id"]]
    assert len(r["tokens"])==r["n"]<=32
    fmt,c,_=grade(r["text"],q["numbers"],q["target"])
    assert (fmt,c)==(r["parsed"],r["correct"])
if "pass" in report:
    a=defaultdict(list);b=defaultdict(list)
    for r in rows:
        if check[r["id"]]["set"]=="confirmation":
            (a if r["arm"]=="iid" else b)[r["id"]].append(r)
    assert len(a)==len(b)==24 and all(len(v)==16 for v in [*a.values(),*b.values()])
    for k in (2,4,8):
        s=report["pass"][str(k)]
        ii=sum(1-math.comb(16-sum(r["correct"] for r in v),k)/math.comb(16,k) for v in a.values())/24
        co=sum(sum(any(x["correct"] for x in sorted(v,key=lambda r:r["j"])[j:j+k]) for j in range(0,16,k))/(16/k) for v in b.values())/24
        assert abs(ii-s["iid"])<1e-12 and abs(co-s["lattice"])<1e-12
print("Verified",len(rows),"samples against frozen Countdown labels and saved pass@k.")
