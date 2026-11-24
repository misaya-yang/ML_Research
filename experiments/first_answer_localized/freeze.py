"""Select a fresh dev cohort using public IDs only, not candidate labels/outcomes."""
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
REPO=P.parent.parent
OLD=REPO/'experiments/first_answer/panels/pilot40.json'
POOL=REPO/'results/local_runs/native-firstpass-1qr7b1va/feedback_policy_v2/population_replication56/mbppplus_rows.json'
def read(p): return json.loads(p.read_text())
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=read(OLD)
    excluded=set(range(11,75))
    excluded.update(map(int,(t['task_id'] for t in old['tasks'])))
    for v in old['excluded_ids'].values(): excluded.update(map(int,v))
    excluded.update(int(t['task_id']) for t in old.get('skipped_reference_failures',[]))
    pool=read(POOL); pool=pool.get('rows',[]) if isinstance(pool,dict) else pool
    ids=[int(row.get('row',row)['task_id']) for row in pool]
    fresh=set(i for i in ids if 11<=i<=600 and i not in excluded)
    seed='first-answer-localized-dev24-v1-20261123'
    sel=sorted(fresh,key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).digest())[:24]
    assert len(sel)==24 and len(ids)==len(set(ids)) and not (set(sel)&excluded)
    cfg=dict(protocol='execution_localized_v1',stage='dev24',ordered_ids=sel,eligible_pool_size=len(fresh),
      selection_seed=seed,cap=192,extra_positions=192,fork_ratio=1.3862943611198906,k_alt=2,max_forks=4,
      model_sha256='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee',source_sha256=h(POOL),
      prior_cohort_sha256=h(OLD), sandbox_image='sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b',
      grade_partition='visible first assert only, hidden test grades only offline',
      primary='localized-public vs same-budget full direct retries and margin-localized control, check reachability under hard cap; no pilot/transfer if dev fails')
    with (P/'dev24.json').open('x') as f: json.dump(cfg,f,indent=2,sort_keys=True);f.write('\n')
if __name__=='__main__':main()
