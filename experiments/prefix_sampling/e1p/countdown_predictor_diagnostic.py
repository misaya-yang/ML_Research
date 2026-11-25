"""Retrospective CPU diagnostic, not the preregistered E1-P result."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from e1p_predictor import bank_prediction
from e1p_evaluate import iid_pass
from e1p_sampling import logs,sha,seed
out=defaultdict(lambda:defaultdict(list))
for r in logs("countdown_control_samples.jsonl"):
    if int(r["id"][2:])>=8:out[r["id"]][r["arm"]].append(r)
assert len(out)==24 and all(len(v[x])==16 for v in out.values() for x in ("iid","lattice"))
# Predictor sees only IID. Recovered control outcomes are only accessed after
# every prediction is computed, although the earlier aggregate result was known.
rr={i:bank_prediction([(r["tokens"],int(r["correct"])) for r in a["iid"]]) for i,a in out.items()}
records=[]
for k in (2,4,8):
 for i,a in sorted(out.items()):
    pp=rr[i][str(k)]
    y=[int(r["correct"]) for r in sorted(a["lattice"],key=lambda r:r["j"])]
    g=np.mean([any(y[j:j+k]) for j in range(0,16,k)])-iid_pass([int(r["correct"]) for r in a["iid"]],k)
    records.append({"id":i,"k":k,"pred":pp["predicted_gain"],"observed":float(g),"mu_iid":pp["mu"],"gmax":pp["gmax"],"leaves":pp["num_leaves"]})
res={}
rng=np.random.default_rng(seed("countdown-retrospective-ci"))
for k in (2,4,8):
 x=[r for r in records if r["k"]==k];v=np.array([[r["pred"],r["observed"]] for r in x])
 V=v[[r["gmax"]>=.02 for r in x]];dif=v[:,1]-v[:,0]
 bs=rng.integers(len(v),size=(10000,len(v)))
 res[k]={"n":len(x),"pred_mean":float(v[:,0].mean()),"real_mean":float(v[:,1].mean()),
  "gap_ci":np.quantile(dif[bs].mean(1),[.025,.975]).tolist(),"n_informative":len(V),
  "spearman_informative":float(spearmanr(*V.T).statistic) if len(V)>=5 and np.ptp(V[:,0])>0 and np.ptp(V[:,1])>0 else None}
with Path("countdown_predictor_diagnostic.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
obj={"scope":"retrospective, 24 local controls / 16 IID rows per question, separate from Claude E1-P","raw_sha":sha("countdown_control_samples.jsonl"),
     "predictor_sha":sha("e1p_predictor.py"),"results":res}
Path("countdown_predictor_diagnostic.json").write_text(json.dumps(obj,indent=2)+"\n")
print(json.dumps(obj,indent=2))
