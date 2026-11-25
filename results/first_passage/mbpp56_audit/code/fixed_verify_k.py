"""Exact fixed-order first-verified extraction of K distinct Power-SMC programs.

python fixed_verify_k.py MODEL FOLDER phase 0 4 --particles 64 --k 4
The source population is after all strictly nonterminal adaptive checkpoints
and before any optional resampling at the cap. Final importance weights are
not used. Compare complete generation, last-window lazy generation, and
lookup of only the final-child ancestries requested by the external policy.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from power_race_v6 import Engine
from power_race_v5 import ancestors, exponent, final_bound, weights
from checkpoint_replay import end_intervals
from ess_certificate import release_envelope
from resample_certificate import next_checkpoint_parents
from single_child_probe import possible_grid_parents, which_ess


class OneLaneEngine(Engine):
    """Optional, deliberately slow shape-invariant numerical control (width 1)."""
    def advance(self, cohort, h, s):
        for t in range(h+1,s+1):
            groups={}
            for i in cohort:
                groups.setdefault(self.slots[i].tape,[]).append(i)
            for group in groups.values():
                super().advance(group,t-1,t)


def resampling(engine):
    prob=weights([x.w for x in engine.slots])
    return 1./sum(p*p for p in prob)<engine.cfg.ess*engine.n,prob


def public_verifier(code,row):
    from grade_batch import grade
    data,_=grade([["public",code,row.get("test_setup_code",""),row["test_list"][0]]])
    return data[0][1]


class Scan:
    def __init__(self,nat,row,k,checker):
        self.nat,self.row,self.k,self.checker=nat,row,k,checker
        self.decode={}
        self.seen=set()
        self.probes=[]
        self.scanned=0
        self.accepted=None
        self.code=None
        self.verify_s=0.
        self.clean_s=0.

    def clean(self,tape):
        if tape not in self.decode:
            clock=time.perf_counter()
            text=self.nat.tok.decode(list(tape),skip_special_tokens=True,clean_up_tokenization_spaces=False)
            self.decode[tape]=self.nat.clean(text)
            self.clean_s+=time.perf_counter()-clock
        return self.decode[tape]

    def consume(self,i,code):
        if i!=self.scanned:
            raise AssertionError("Program out of native slot order")
        self.scanned+=1
        encoded=code.encode("utf-8")
        if encoded not in self.seen:
            self.seen.add(encoded)
            clock=time.perf_counter()
            verdict=self.checker(code,self.row)
            elapsed=time.perf_counter()-clock
            self.verify_s+=elapsed
            self.probes.append(dict(slot=i,public=verdict,code=code,
                                     sha256=hashlib.sha256(encoded).hexdigest(),verify_s=elapsed))
            if verdict=="pass":
                self.accepted=i
                self.code=code
        return self.accepted is not None or len(self.probes)==self.k


def simulate_all(engine):
    """Conventional complete-population generation and native intermediate checks."""
    cfg=engine.cfg
    ids=list(range(engine.n));h=0
    for s in [*range(cfg.block,engine.cap,cfg.block),engine.cap]:
        engine.advance(ids,h,s)
        if s<engine.cap:
            resample,prob=resampling(engine)
            if resample:
                parents=ancestors(prob,engine.offset[s-1])
                engine.slots=[replace(engine.slots[i],w=0.) for i in parents]
        h=s


def tail(engine,scan,start,width):
    """Known population, no future resampling. Run requested IDs to the cap."""
    n=engine.n
    while scan.scanned<n:
        j=scan.scanned
        if engine.slots[j].done or start==engine.cap:
            if scan.consume(j,scan.clean(engine.slots[j].tape)):
                return
            continue
        end=min(n,j+width)
        engine.advance(list(range(j,end)),start,engine.cap)
        for i in range(j,end):
            if scan.consume(i,scan.clean(engine.slots[i].tape)):
                return


class LastParents:
    """Deferred parent endpoints before the last nonterminal resampling check.

    All uncertain endpoints share an exact current clock. Already materialized
    endpoints stay at s. A child's program needs its parent and original
    child-slot RNG, but no post-check weight because this is the last check.
    """
    def __init__(self,engine,scan,t,s,events):
        self.e,self.scan,self.t,self.s,self.events=engine,scan,t,s,events
        self.pre=engine.slots
        self.known=set()
        self.parents=None
        self.distribution=None
        self.a=lambda t:exponent(t,self.e.cfg)
        self.mark_absorbed()

    def mark_absorbed(self):
        for i,x in enumerate(self.pre):
            if i not in self.known and (x.done or self.t==self.s):
                x.w=final_bound(x.w,x.lp,self.t,self.s,self.e.cfg)
                self.known.add(i)

    def materialize(self,ids):
        ids=sorted(set(ids)-self.known)
        if ids:
            # The original parent slot's token tape is used through step s.
            self.e.slots=self.pre
            self.e.advance(ids,self.t,self.s)
            self.known.update(ids)
            self.distribution=None

    def bounds(self):
        if self.distribution is not None:
            return self.distribution
        n=self.e.n
        w=[x.w for x in self.pre]
        p=[x.lp for x in self.pre]
        fixed=[i in self.known for i in range(n)]
        # Frozen endpoint rows are overwritten below. Only unknown rows use
        # end_intervals' t-to-s ramp. Native lower floor is shared with v5.
        lo,hi=end_intervals(w,p,[False]*n,self.t,self.s,self.a,self.e.prompt_logits.shape[-1])
        for i in self.known:
            lo[i]=hi[i]=w[i]
        maybe_resample,maybe_identity=which_ess(lo,hi,fixed,self.e.cfg.ess)
        self.distribution=(lo,hi,maybe_resample,maybe_identity)
        return self.distribution

    def exact_map(self):
        if self.parents is None:
            w=weights([x.w for x in self.pre])
            if 1./sum(p*p for p in w)<self.e.cfg.ess*self.e.n:
                self.parents=ancestors(w,self.e.offset[self.s-1])
            else:
                self.parents=list(range(self.e.n))
        return self.parents

    def source_for(self,j):
        while True:
            if len(self.known)==self.e.n:
                return self.exact_map()[j],None
            clock=time.perf_counter()
            lo,hi,resample,identity=self.bounds()
            cand=set(possible_grid_parents(lo,hi,(self.e.offset[self.s-1]+j)/self.e.n)) if resample else set()
            if identity:
                cand.add(j)
            if not cand:
                raise AssertionError("Empty possible-parent set")
            direct=None
            if len(cand)>1 and all(self.pre[i].done for i in cand):
                codes={self.scan.clean(self.pre[i].tape) for i in cand}
                if len(codes)==1:
                    direct=next(iter(codes))
            self.e.stats["certificate_s"]+=time.perf_counter()-clock
            if len(cand)==1:
                parent=next(iter(cand))
                self.events.append(dict(t=self.t,s=self.s,child=j,parent=parent,
                                        unknown_before=self.e.n-len(self.known),identity=identity,resampling=resample))
                self.materialize([parent])
                return parent,None
            if direct is not None:
                self.events.append(dict(t=self.t,s=self.s,child=j,possible_done=sorted(cand),
                                        unknown_before=self.e.n-len(self.known),identity=identity,resampling=resample))
                return None,direct
            # Tighten bounds with one shared CoW forward step on every remaining
            # parent, keeping any previously computed endpoints intact.
            ids=[i for i in range(self.e.n) if i not in self.known]
            self.e.slots=self.pre
            until=min(self.s,self.t+getattr(self.e.cfg,"resolve_stride",1))
            self.e.advance(ids,self.t,until)
            self.t=until
            self.mark_absorbed()
            self.distribution=None


def selective_tail(engine,scan,h,s,width,events):
    resolve=LastParents(engine,scan,h,s,events)
    while scan.scanned<engine.n:
        j=scan.scanned
        parent,code=resolve.source_for(j)
        if code is not None or resolve.pre[parent].done:
            code=code if code is not None else scan.clean(resolve.pre[parent].tape)
            if scan.consume(j,code):return
            continue
        end=min(engine.n,j+width)
        batch={j:(parent,None)}
        for i in range(j+1,end):
            batch[i]=resolve.source_for(i)
        # Projection onto child slots. Parent slots and their frozen KV states
        # remain available to satisfy any later verifier requests.
        child=resolve.pre.copy()
        active=[]
        for i,(p,known_code) in batch.items():
            if known_code is None:
                child[i]=replace(resolve.pre[p],w=0.)
                if not child[i].done:
                    active.append(i)
        engine.slots=child
        engine.advance(active,s,engine.cap)
        for i,(p,known_code) in batch.items():
            code=known_code if known_code is not None else scan.clean(child[i].tape)
            if scan.consume(i,code):return
        engine.slots=resolve.pre


def run_policy(nat,row,cfg,seed,mode,checker=public_verifier):
    import torch
    start=time.perf_counter()
    n=cfg.particles
    ids=list(range(n))
    scan=Scan(nat,row,cfg.k,checker)
    events=[]
    with torch.inference_mode():
        if getattr(cfg,"backbone","plain")=="spec":
            from speculative_power import SpecEngine
            cls=SpecEngine
        else:
            cls=OneLaneEngine if cfg.layout=="one" else Engine
        engine=cls(nat,row,cfg,seed)
        cap=engine.cap
        last=((cap-1)//cfg.block)*cfg.block
        if mode=="full":
            simulate_all(engine)
            for i,x in enumerate(engine.slots):
                if scan.consume(i,scan.clean(x.tape)):break
        elif mode=="terminal":
            prev=0
            for s in range(cfg.block,last+1,cfg.block):
                engine.advance(ids,prev,s)
                do,prob=resampling(engine)
                if do:
                    par=ancestors(prob,engine.offset[s-1])
                    engine.slots=[replace(engine.slots[p],w=0.) for p in par]
                prev=s
            tail(engine,scan,last,cfg.tail_width)
        elif mode=="across":
            h=0
            ended=False
            for s in range(cfg.block,last,cfg.block):
                pending=None
                released_at=None
                def after_step(t,cohort):
                    nonlocal pending,released_at
                    if t==s or pending is not None:
                        return cohort
                    x=engine.slots
                    w=[p.w for p in x];lp=[p.lp for p in x];d=[p.done for p in x]
                    if not any(d):
                        return cohort
                    clock=time.perf_counter();a=lambda j:exponent(j,cfg)
                    released=release_envelope(w,lp,d,t,cap,cfg.block,cfg.ess,a,check_at_cap=False)
                    if released is None:
                        pending=next_checkpoint_parents(w,lp,d,t,s,cfg.ess,engine.offset[s-1],a,
                                                        vocab_size=engine.prompt_logits.shape[-1])
                        if pending is not None:
                            cohort=sorted(set(pending))
                            events.append(dict(t=t,s=s,all_parents=list(pending),remaining=len(cohort)))
                    engine.stats["certificate_s"]+=time.perf_counter()-clock
                    if released is not None:
                        events.append(dict(t=t,no_further_resampling=True))
                        released_at=t
                        return []
                    return cohort
                if getattr(cfg,"backbone","plain")=="spec":
                    # The speculative engine can issue a full draft block yet
                    # check the certificate after each committed target token.
                    # Raw issued positions (including unused drafts) are paid.
                    engine.advance(ids,h,s,after=after_step)
                else:
                    subset=ids
                    for t in range(h+1,s+1):
                        engine.advance(subset,t-1,t)
                        subset=after_step(t,subset)
                        if released_at is not None:break
                if released_at is not None:
                    tail(engine,scan,released_at,cfg.tail_width)
                    ended=True;break
                if pending is not None:
                    par=pending
                else:
                    do,prob=resampling(engine)
                    par=ancestors(prob,engine.offset[s-1]) if do else None
                if par is not None:
                    engine.slots=[replace(engine.slots[p],w=0.) for p in par]
                h=s
            if not ended:
                if last>0:
                    selective_tail(engine,scan,h,last,cfg.tail_width,events)
                else:
                    tail(engine,scan,0,cfg.tail_width)
        else:
            raise ValueError("Unknown evaluation mode")
    total=time.perf_counter()-start
    return dict(mode=mode,code=scan.code,accepted_slot=scan.accepted,scanned_slots=scan.scanned,
                public_probes=scan.probes,distinct_probes=len(scan.probes),public_verify_s=scan.verify_s,
                cleaning_s=scan.clean_s,end_to_end_s=total,inference_and_setup_s=total-scan.verify_s,
                events=events,seed=seed,**engine.stats)


def hidden(row,results):
    from grade_batch import grade
    codes=list(dict.fromkeys(x["code"] for x in results.values() if x["code"] is not None))
    if not codes:
        return {key:"abstain" for key in results},0.,[]
    clock=time.perf_counter()
    tests=row["test_list"][1:]
    if not tests:raise ValueError("Missing held-out tests")
    jobs=[[f"{i}/{j}",code,row.get("test_setup_code",""),test] for i,code in enumerate(codes)
          for j,test in enumerate(tests)]
    rows,_=grade(jobs)
    statuses={k:st for k,st,*_ in rows}
    passed={code:all(statuses[f"{i}/{j}"]=="pass" for j in range(len(tests))) for i,code in enumerate(codes)}
    return {k:("abstain" if v["code"] is None else "correct" if passed[v["code"]] else "wrong")
            for k,v in results.items()},time.perf_counter()-clock,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("modeldir","folder","phase","begin","end"):p.add_argument(name)
    p.add_argument("--particles",type=int,default=64)
    p.add_argument("--k",type=int,default=4)
    p.add_argument("--alpha",type=float,default=4.)
    p.add_argument("--ramp",type=int,default=100)
    p.add_argument("--block",type=int,default=64)
    p.add_argument("--ess",type=float,default=.5)
    p.add_argument("--seed",type=int,default=4728103)
    p.add_argument("--tail-width",type=int,default=1)
    p.add_argument("--layout",choices=("compact","one"),default="compact")
    p.add_argument("--sampler",choices=("device","cpu"),default="device")
    p.add_argument("--backbone",choices=("plain","spec"),default="plain")
    p.add_argument("--depth",type=int,default=4)
    p.add_argument("--min-match",type=int,default=2)
    p.add_argument("--max-match",type=int,default=5)
    p.add_argument("--head-mode",choices=("lazy","eager"),default="lazy")
    p.add_argument("--resolve-stride",type=int,default=1,
                   help="Maximum synchronous tokens per unresolved-parent refinement; 4 permits draft blocks")
    p.add_argument("--methods",nargs="+",choices=("full","terminal","across"),
                   default=["full","terminal","across"])
    cfg=p.parse_args()
    if not(1<=cfg.k and cfg.particles>=1 and cfg.block>=1 and 0<cfg.ess<=1 and cfg.ramp>=0
           and cfg.alpha>=1 and math.isfinite(cfg.alpha) and cfg.tail_width>=1):
        raise ValueError("Invalid policy settings")
    if cfg.resolve_stride<1 or (cfg.backbone=="spec" and
            (cfg.layout!="compact" or not 0<=cfg.depth<=16 or not 1<=cfg.min_match<=cfg.max_match)):
        raise ValueError("Invalid checkpoint stride or speculative configuration")
    if len(cfg.methods)!=len(set(cfg.methods)):
        raise ValueError("Repeated method")
    cfg.capture=False;cfg.shape="compact";cfg.ancestor_cert=True;cfg.early=True
    args={k:v for k,v in vars(cfg).items() if k not in {"modeldir","folder","begin","end"}}
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"independent_mh_screen"))
    sys.argv=sys.argv[:6]
    import native_sampling as nat
    if cfg.backbone=="plain" and cfg.resolve_stride==1:
        for unused in ("backbone","depth","min_match","max_match","head_mode","resolve_stride"):
            args.pop(unused,None)
    args.update(version=1,cap=nat.cap)
    tag=hashlib.sha256(json.dumps(args,sort_keys=True).encode()).hexdigest()[:12]
    path=Path(cfg.folder)/f"fixed_verify_{tag}.jsonl"
    def save(data):
        with path.open("a") as f:f.write(json.dumps(data,ensure_ascii=False)+"\n")
    done={r["task"] for line in (path.read_text().splitlines() if path.exists() else [])
          if (r:=json.loads(line)).get("stage")=="paired"}
    save(dict(stage="loaded",params=args,load_s=nat.load_time))
    for e in nat.tasks:
        row=e["row"];tid=row["task_id"];idx=int(e["row_idx"])
        if tid in done:continue
        seed=cfg.seed+1009*idx
        shift=idx%len(cfg.methods)
        order=cfg.methods[shift:]+cfg.methods[:shift]
        results={m:run_policy(nat,row,cfg,seed,m) for m in order}
        outcomes,offline_s,verdicts=hidden(row,results)
        def signature(r):
            return (r["code"],r["accepted_slot"],
                    [(p["slot"],p["code"],p["public"]) for p in r["public_probes"]])
        first=results[order[0]]
        same=all(signature(x)==signature(first) for x in results.values())
        save(dict(stage="paired",task=tid,order=order,same_probe_path=same,outcomes=outcomes,
                  offline_hidden_s=offline_s,hidden_grades=verdicts,methods=results))
        print("fixed",tid,outcomes,same,{m:(x["executed_decode_positions"],x["decode_calls"],
              round(x["end_to_end_s"],2)) for m,x in results.items()},flush=True)
        if not same and cfg.layout=="one":
            raise RuntimeError("One-lane deterministic coupling mismatch; record saved")


if __name__=="__main__":
    main()

