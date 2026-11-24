"""Frozen first-answer controlled development experiment."""
import argparse
import collections
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from freeze import P, REPO, POOL, OLD, h, read
from localization import operation_point, traced_worker

sys.path.insert(0,str(REPO/'experiments/first_answer'))
from common import DockerGrader, Engine, HFBackend, Ledger, VisibleChecker, chash, clean, grade_hidden, mbpp_task, run_decode, sampler_chooser, stable_int
from policies import branch, greedy_trace, list_forks

class Tracer:
    def __init__(self,image):
        self.docker=shutil.which('docker')
        if not self.docker: raise RuntimeError('no isolated Docker grader')
        pinned=subprocess.check_output([self.docker,'image','inspect','python:3.12-slim','--format','{{.Id}}'],text=True).strip()
        if pinned!=image: raise RuntimeError('unapproved local grader image')
        from grade_batch import WORKER, DRIVER
        self.worker=traced_worker(WORKER)
        self.driver=DRIVER
        self.cmd=[self.docker,'run','--rm','--pull=never','--network=none','--read-only','--cap-drop=ALL',
                  '--security-opt=no-new-privileges','--pids-limit=16','--memory=384m','--memory-swap=384m','--user=65534:65534',
                  '--cpus=1','--log-driver=none']
        deps=Path.home()/'miniconda3/envs/aidemo/lib/python3.13/site-packages'
        for k in ('sympy','mpmath'):
            if not (deps/k).is_dir(): raise RuntimeError(f'missing confined dependency {k}')
            self.cmd+=['--mount',f'type=bind,src={deps/k},dst=/opt/mbppdeps/{k},readonly']
        self.cmd+=['-i',image,'python','-I','-S','-B','-c',self.driver]
    def __call__(self,code,task):
        t=time.monotonic()
        tkey='trace'
        p=subprocess.run(self.cmd,input=json.dumps({'worker':self.worker,'jobs':[[tkey,code,task.setup,task.visible[0]]]}),
                         text=True,capture_output=True,timeout=60,check=True)
        rows=list(map(json.loads,p.stdout.splitlines()))
        if len(rows)!=1 or rows[0][0]!=tkey or rows[0][1]=='infra_error':
            raise RuntimeError('confined trace judge did not produce a valid result')
        r=rows[0]
        return (r[1]=='pass',r[1],r[3] if len(r)==5 else [],time.monotonic()-t)

def forks_for(eng,tr,code,lines,cfg):
    whole=eng.text(tr['gen']); start=whole.find(code)
    if start<0 or not code: return [],[]
    ops=[]; loc=[]
    for f in list_forks(eng,tr,cfg['fork_ratio'],cfg['k_alt']):
        before=eng.text(tr['gen'][:f['pos']])
        if not whole.startswith(before): continue
        k=len(before)-start
        if not 0<=k<len(code):continue
        prefix=code[:k]; line=prefix.count('\n')+1; col=len(prefix.rsplit('\n',1)[-1])
        if operation_point(code,line,col):
            q=f|{'line':line,'column':col};ops.append(q)
            if line in lines:loc.append(q)
    # Trace ranks reachable operation sites lexically by dynamic line proximity to failure,
    # not by hidden correctness. Within a line confidence remains a deterministic tiebreak.
    last=lines[-1] if lines else 0
    loc.sort(key=lambda f:(abs(f['line']-last),f['margin'],f['pos']))
    return loc,ops

def run_item(eng,task,cfg,grader,tracer):
    led=Ledger();eng.bind(led,task)
    ids=eng.encode(task.messages());tr=greedy_trace(eng,ids); base=led.as_dict()
    c=clean(eng.text(tr['gen'])); vis,status,lines,t=tracer(c,task)
    base['exec_calls']=1;base['exec_s']=t
    local,unlocal=forks_for(eng,tr,c,lines,cfg)
    methods={};all_codes={chash(c):c}
    cfg_budget=cfg['extra_positions']
    for mode,fs in [('localized',local),('unlocalized',unlocal),('likelihood',local),('direct_retry',[])]:
        l=Ledger();eng.bind(l,task); chosen=c;attempts=[]; options=[]
        ck=VisibleChecker(grader,task,l);ck.cache[chash(c)]=vis
        if not vis and status in ('fail','timeout','resource_limit'):
            if mode=='direct_retry':
                for a in range(1,17):
                    seed=[20261123,stable_int(task.task_id),a]
                    cache=eng.clone(tr['cache'],len(ids))
                    b=run_decode(eng,cache,tr['lp0'],len(ids),[],sampler_chooser({'strategy':'top_p','temperature':1.0,'params':{'p':.99}},seed),budget=cfg_budget)
                    if b['aborted']:break
                    cc=clean(eng.text(b['gen']));ok=ck(cc)
                    attempts.append({'hash':chash(cc),'public':ok,'extra':l.positions})
                    all_codes[chash(cc)]=cc
                    if ok:chosen=cc;break
            else:
                for f in fs[:cfg['max_forks']]:
                    for v,lp in f['alts']:
                        if l.positions>=cfg_budget:break
                        b=branch(eng,tr,f['pos'],v,budget=cfg_budget)
                        if b['aborted']:break
                        cc=clean(eng.text(b['gen']));all_codes[chash(cc)]=cc
                        ok=ck(cc) if mode!='likelihood' else None
                        score=(sum(tr['seq_lp'][:f['pos']])+lp+sum(b['seq_lp']))/max(1,len(b['gen']))
                        options.append((score,cc))
                        attempts.append({'hash':chash(cc),'public':ok,'extra':l.positions,'fork':f['pos'],'line':f['line']})
                        if ok:chosen=cc;break
                    if chosen!=c or l.positions>=cfg_budget:break
                if mode=='likelihood' and options:
                    chosen=max(options,key=lambda x:x[0])[1]
        assert l.positions<=cfg_budget
        methods[mode]={'hash':chash(chosen),'ledger':l.as_dict(),'attempts':attempts}
    labels,grade_seconds=grade_hidden(grader,task,all_codes)
    return dict(task=task.task_id,draft=chash(c),draft_visible=vis,draft_status=status,executed_lines=lines,
                localized_forks=len(local),unlocalized_forks=len(unlocal),base=base,methods=methods,labels=labels,grade_seconds=grade_seconds)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--approve',action='store_true');p.add_argument('--out')
    a=p.parse_args();cfg=read(P/'dev24.json')
    if h(POOL)!=cfg['source_sha256'] or h(OLD)!=cfg['prior_cohort_sha256']:raise SystemExit('frozen inputs changed')
    model=Path('/var/folders/py/yshkz8454g98pq1wc0rnw5pm0000gp/T/icml-m4-native.8t0fjn/qwen15-official')
    if h(model/'model.safetensors')!=cfg['model_sha256']:raise SystemExit('checkpoint mismatch')
    grader=DockerGrader(); tracer=Tracer(cfg['sandbox_image'])
    if a.preflight:
        class Example: setup='';visible=['assert example(2)==4']
        c='def example(n):\n    return n+1\n'
        ok,status,lines,_=tracer(c,Example())
        assert ok is False and status=='fail' and lines==[2],(ok,status,lines)
        print('sandbox trusted fixture pass; no inference')
        return
    if not(a.approve and a.out):raise SystemExit('need approval + new private output path')
    for line in (P/'RUN.sha256').read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if h(P/filename)!=digest:raise SystemExit(f'frozen runner mismatch: {filename}')
    pool=read(POOL);pool=pool.get('rows',[]) if isinstance(pool,dict) else pool
    by={int(r.get('row',r)['task_id']):r.get('row',r) for r in pool}
    out=Path(a.out);out.mkdir(exist_ok=False)
    (out/'provenance.json').write_text(json.dumps({'config_sha256':h(P/'dev24.json'),'code_sha256':{p.name:h(p) for p in P.glob('*.py')}},indent=2))
    eng=Engine(HFBackend(str(model)),cap=cfg['cap'])
    rows=[]
    with (out/'rows.jsonl').open('x') as f:
        for i in cfg['ordered_ids']:
            r=dict(by[i]);r['test_setup_code']='\n'.join(r.get('test_imports') or [])
            z=run_item(eng,mbpp_task(r,family='mbppplus'),cfg,grader,tracer);rows.append(z)
            f.write(json.dumps(z)+'\n');f.flush()
            print(i,z['draft_visible'],z['localized_forks'],flush=True)
    counts=collections.Counter()
    for r in rows:
        for name,m in [('greedy',{'hash':r['draft']}),*r['methods'].items()]:
            counts[name]+=int(r['labels'][m['hash']]['all'])
    print(json.dumps(dict(counts),sort_keys=True))
if __name__=='__main__':main()
