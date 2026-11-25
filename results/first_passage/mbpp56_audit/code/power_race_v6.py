"""Checkpoint-wave Power-SMC. Primary comparator is compact, CoW, certified v5.

python power_race_v6.py MODEL FOLDER dev 0 2 --particles 64 --wave-size 16
--cohort-reference computes a third matched-wave full trajectory and records
endpoint snapshots for replay. Variable BF16 widths may change machine logits:
selected-program parity vs v5 and vs same-wave reference are recorded separately.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from power_race_v5 import (ancestors, draw_rows, draw_rows_device, exponent, final_bound,
                           random_tape, run as synchronous, verify_pair, weights)
from checkpoint_replay import certify_after_reveal, end_intervals, ranked_cohorts


@dataclass
class Slot:
    tape: tuple = ()
    lp: float = 0.
    w: float = 0.
    done: bool = False
    reason: str = "cap"
    store: object = None
    row: int = 0


@dataclass
class Store:
    cache: object
    size: int


def gather_cache(records, torch, device):
    """New DynamicCache shell; gather only the requested Qwen2 KV tensor rows.

    DynamicLayer.update replaces keys/values by concatenation. A shallow layer
    shell therefore isolates a paused wave without cloning the complete batch.
    Other cache types are rejected rather than mutated by guesswork.
    """
    first=records[0][0]
    one=all(st is first for st,_ in records)
    idx=[i for _,i in records]
    identity=one and idx==list(range(first.size))
    if not hasattr(first.cache,"layers") or not all(type(x).__name__=="DynamicLayer"
                                                     for x in first.cache.layers):
        raise TypeError(f"Expected Qwen DynamicCache with DynamicLayer, got {type(first.cache)} / "
                        f"{[type(x).__name__ for x in getattr(first.cache,'layers',())]}")
    selector=None if not one or identity else torch.tensor(idx,dtype=torch.long,device=device)
    result=copy.copy(first.cache)
    result.layers=[]
    for layer_id, old in enumerate(first.cache.layers):
        layer=copy.copy(old)
        if one and not identity:
            layer.keys=old.keys.index_select(0,selector)
            layer.values=old.values.index_select(0,selector)
        elif not one:
            for name in ("keys","values"):
                shards=[getattr(st.cache.layers[layer_id],name)[i:i+1] for st,i in records]
                setattr(layer,name,torch.cat(shards,dim=0))
        result.layers.append(layer)
    return result


class Engine:
    def __init__(self,nat,row,cfg,seed):
        import torch
        self.torch=torch
        self.nat,self.cfg=nat,cfg
        self.model,self.tok=nat.model,nat.tok
        self.head=self.model.get_output_embeddings()
        self.device=next(self.model.parameters()).device
        self.sync=(torch.mps.synchronize if self.device.type=="mps" else
                   torch.cuda.synchronize if self.device.type=="cuda" else lambda:None)
        self.n,self.cap=cfg.particles,nat.cap
        self.u,self.offset,self.g=random_tape(seed,self.cap,self.n)
        self.stats=dict(prefill_calls=1,prefill_positions=0,decode_calls=0,executed_decode_positions=0,
                        live_decode_positions=0,cow_shared_positions=0,padding_decode_positions=0,
                        sampled_tokens=0,executed_proposal_rows=0,stop_text_decodes=0,prefill_s=0.,
                        decode_forward_s=0.,proposal_s=0.,stop_s=0.,cache_reorder_s=0.,certificate_s=0.)
        self.groups={}
        self.forward_id=0
        t0=time.perf_counter()
        prompt=self.tok.apply_chat_template(nat.prompt(row),tokenize=True,add_generation_prompt=True,
                                            return_tensors="pt",return_dict=True)
        ids=prompt.input_ids.to(self.device)
        feed=self.model.model(input_ids=ids,use_cache=True)
        self.prompt_logits=self.head(feed.last_hidden_state[:,-1,:]).float()
        store=Store(feed.past_key_values,1)
        self.slots=[Slot(store=store) for _ in range(self.n)]
        self.sync()
        self.stats["prefill_s"]=time.perf_counter()-t0
        self.stats["prefill_positions"]=ids.shape[1]

    def advance(self, cohort, h, s):
        """Materialize identities in cohort only from token h+1 through s."""
        stats=self.stats
        for t in range(h+1,s+1):
            a=exponent(t,self.cfg)
            delta=a-exponent(t-1,self.cfg)
            active=[]
            for i in cohort:
                x=self.slots[i]
                x.w+=delta*x.lp
                if not x.done:
                    active.append(i)
            if not active:
                continue
            if t==1:
                logits=self.prompt_logits
                mapping={i:0 for i in active}
                store=self.slots[active[0]].store
            else:
                t0=time.perf_counter()
                destinations={}
                records=[]
                tokens=[]
                mapping={}
                for i in active:
                    x=self.slots[i]
                    if x.tape not in destinations:
                        destinations[x.tape]=len(records)
                        records.append((x.store,x.row))
                        tokens.append(x.tape[-1])
                    mapping[i]=destinations[x.tape]
                cache=gather_cache(records,self.torch,self.device)
                inputs=self.torch.tensor(tokens,dtype=self.torch.long,device=self.device).reshape(-1,1)
                self.sync()
                stats["cache_reorder_s"]+=time.perf_counter()-t0
                t0=time.perf_counter()
                out=self.model.model(input_ids=inputs,past_key_values=cache,use_cache=True)
                logits=self.head(out.last_hidden_state[:,-1,:]).float()
                store=Store(out.past_key_values,len(records))
                del out
                self.sync()
                stats["decode_forward_s"]+=time.perf_counter()-t0
                stats["decode_calls"]+=1
                stats["executed_decode_positions"]+=len(records)
                stats["live_decode_positions"]+=len(active)
                stats["cow_shared_positions"]+=len(active)-len(records)
                if self.cfg.capture:
                    groups=self.groups.setdefault(t,[-1]*self.n)
                    for i,r in mapping.items():
                        groups[i]=self.forward_id+r
                    self.forward_id+=len(records)
            t0=time.perf_counter()
            ix=[mapping[i] for i in active]
            us=[self.u[t-1][i] for i in active]
            if self.cfg.sampler=="device":
                refs=None if ix==list(range(store.size)) else ix
                info=draw_rows_device(logits,us,a,self.torch,refs)
            else:
                selected=logits.detach().cpu().numpy()[ix]
                z,lp,w=draw_rows(selected,us,a)
                info=zip(z,lp,w)
            stats["proposal_s"]+=time.perf_counter()-t0
            stats["executed_proposal_rows"]+=store.size if self.cfg.sampler=="device" else len(active)
            stats["sampled_tokens"]+=len(active)
            t0=time.perf_counter()
            fence={}
            for i,(tok,base,inc) in zip(active,info):
                token=int(tok)
                x=self.slots[i]
                x.store,x.row=store,mapping[i]
                x.lp+=base
                x.w+=inc
                x.tape+=(token,)
                if token in self.nat.stops:
                    x.done,x.reason=True,"eos"
                else:
                    if x.tape not in fence:
                        fence[x.tape]=(self.tok.decode(list(x.tape),skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False).count("```")>=2)
                        stats["stop_text_decodes"]+=1
                    if fence[x.tape]:
                        x.done,x.reason=True,"fence"
                if x.done:
                    x.store=None
            stats["stop_s"]+=time.perf_counter()-t0


def run_waves(nat,row,cfg,seed,prune=True,*,engine_class=Engine):
    import torch
    start=time.perf_counter()
    n,cap=cfg.particles,nat.cap
    all_ids=list(range(n))
    windows=[]
    captures=[]
    resampling=[]
    winner=None
    with torch.inference_mode():
        engine=engine_class(nat,row,cfg,seed)
        prev=0
        for s in [*range(cfg.block,cap,cfg.block),cap]:
            h=prev+min(s-prev, max(0,round((s-prev)*cfg.probe_fraction)))
            engine.advance(all_ids,prev,h)
            slots=engine.slots
            w0=[x.w for x in slots]
            p0=[x.lp for x in slots]
            d0=[x.done for x in slots]
            low,high=end_intervals(w0,p0,d0,h,s,lambda t:exponent(t,cfg),engine.prompt_logits.shape[-1]) if h<s else (w0,w0)
            keys={}
            cohorts=[keys.setdefault(x.tape,len(keys)) for x in slots]
            snapshot=dict(h=h,s=s,cap=cap,alpha=cfg.alpha,ramp=cfg.ramp,ess=cfg.ess,
                          offset=engine.offset[s-1],g=engine.g,w0=w0,lp0=p0,done0=d0,
                          vocab=engine.prompt_logits.shape[-1],cohort_keys=cohorts)
            known={i for i,x in enumerate(slots) if x.done or s==h}
            for i in known:
                slots[i].w=high[i]
            unknown=set(all_ids)-known
            gumbels=engine.g if s==cap else None
            batches=ranked_cohorts(high,sorted(unknown),cohorts,cfg.wave_size,gumbels)
            batch_cursor=0
            # Optionally fuse the tail to protect latency when early waves
            # cannot certify. This policy is the same in both cohort runs.
            if cfg.max_waves and len(batches)>cfg.max_waves:
                tail=[i for b in batches[cfg.max_waves-1:] for i in b]
                batches=batches[:cfg.max_waves-1]+[tail]
            pending=set(unknown)
            proof=None
            processed=[]
            safe_parents=None
            safe_winner=None
            while True:
                if proof is None and pending and known:
                    clock=time.perf_counter()
                    decision=certify_after_reveal(low,high,{i:slots[i].w for i in known},
                              ess_fraction=cfg.ess,offset=engine.offset[s-1],gumbels=gumbels)
                    engine.stats["certificate_s"]+=time.perf_counter()-clock
                    if decision is not None:
                        if s==cap:
                            safe_winner=decision[1]
                        else:
                            safe_parents=list(decision[1])
                            required=sorted(set(safe_parents)-known)
                            if required:
                                # The identity of a needed parent is known.
                                # Its trajectory and cache must still be built.
                                engine.advance(required,h,s)
                                processed.append(len(required))
                                known.update(required)
                                pending.difference_update(required)
                        proof=dict(kind="gumbel" if s==cap else "ancestors",unrevealed=sorted(pending),
                                   reached=len(known),processed_waves=len(processed),parents=safe_parents,
                                   winner=safe_winner)
                if not pending or (prune and proof is not None):
                    break
                group=[i for i in batches[batch_cursor] if i in pending]
                batch_cursor+=1
                if not group:
                    continue
                engine.advance(group,h,s)
                processed.append(len(group))
                pending.difference_update(group)
                known.update(group)
            if not prune and pending:
                raise AssertionError("Full endpoint replay left a deferred slot")
            if s==cap:
                winner=(safe_winner if prune and safe_winner is not None else
                        max(all_ids,key=lambda i:(slots[i].w+engine.g[i],-i)))
                ancestry=None
            elif prune and safe_parents is not None:
                ancestry=safe_parents
            else:
                probs=weights([x.w for x in slots])
                ess=1./sum(p*p for p in probs)
                ancestry=ancestors(probs,engine.offset[s-1]) if ess<cfg.ess*n else None
            if not prune and proof is not None:
                if (s==cap and winner!=safe_winner) or (s!=cap and ancestry!=safe_parents):
                    raise AssertionError("Full end weights contradict an earlier wave certificate")
            if cfg.capture and not pending:
                captures.append(dict(**snapshot,wend=[x.w for x in slots],
                    groups=[engine.groups.get(j,[-1]*n) for j in range(h+1,s+1)],
                    ref_parents=ancestry,ref_winner=winner))
            windows.append(dict(h=h,s=s,at_probe_done=sum(snapshot["done0"]),
                                wave_sizes=processed,proof=proof,actually_deferred=len(pending)))
            if ancestry is not None:
                if not set(ancestry)<=known:
                    raise AssertionError("Unmaterialized resampling parent")
                engine.slots=[replace(slots[i],w=0.) for i in ancestry]
                resampling.append(dict(t=s,parents=ancestry))
            prev=s
        x=engine.slots[winner]
        txt=nat.tok.decode(list(x.tape),skip_special_tokens=True,clean_up_tokenization_spaces=False)
        code=nat.clean(txt)
    return dict(winner=winner,token_ids=list(x.tape),code=code,reason=x.reason,seed=seed,
                seconds=time.perf_counter()-start,windows=windows,snapshots=captures,
                resampling=resampling,**engine.stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("modeldir","folder","phase","begin","end"):
        p.add_argument(key)
    p.add_argument("--particles",type=int,default=64)
    p.add_argument("--alpha",type=float,default=4.)
    p.add_argument("--ramp",type=int,default=100)
    p.add_argument("--block",type=int,default=64)
    p.add_argument("--ess",type=float,default=.5)
    p.add_argument("--seed",type=int,default=4728103)
    p.add_argument("--wave-size",type=int,default=16)
    p.add_argument("--max-waves",type=int,default=0)
    p.add_argument("--probe-fraction",type=float,default=.5)
    p.add_argument("--sampler",choices=["device","cpu"],default="device")
    p.add_argument("--cohort-reference",action="store_true")
    p.add_argument("--capture",action="store_true",help="Archive window scores and physical prefix ids for offline replay")
    cfg=p.parse_args()
    if not(1<=cfg.wave_size and cfg.particles>=1 and cfg.block>=1 and cfg.max_waves>=0 and
           cfg.alpha>=1 and math.isfinite(cfg.alpha) and cfg.ramp>=0 and 0<cfg.ess<=1 and 0<=cfg.probe_fraction<=1):
        raise ValueError("Invalid Power-SMC or cohort settings")
    cfg.shape,cfg.ancestor_cert,cfg.early="compact",True,True
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"independent_mh_screen"))
    sys.argv=sys.argv[:6]
    import native_sampling as nat
    params={k:v for k,v in vars(cfg).items() if k not in {"modeldir","folder","begin","end"}}
    params.update(cap=nat.cap,version=6,revision=2)
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]
    logfile=Path(cfg.folder)/f"power_wave_{key}.jsonl"
    def save(record):
        with logfile.open("a") as f:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
    done={r["task"] for line in (logfile.read_text().splitlines() if logfile.exists() else [])
          if (r:=json.loads(line)).get("stage")=="paired"}
    save(dict(stage="loaded",params=params,load_s=nat.load_time))
    for e in nat.tasks:
        row=e["row"]; tid=row["task_id"]; idx=int(e["row_idx"])
        if tid in done:
            continue
        seed=cfg.seed+1009*idx
        order=("pruned","full") if idx%2 else ("full","pruned")
        result={}
        for name in order:
            result[name]=(synchronous(nat,row,cfg,seed,True) if name=="full" else
                          run_waves(nat,row,cfg,seed,True))
        a,b=result["full"],result["pruned"]
        pair=lambda x,y:(x["winner"],x["token_ids"])==(y["winner"],y["token_ids"])
        cohort=run_waves(nat,row,cfg,seed,False) if cfg.cohort_reference else None
        cohort_match=pair(b,cohort) if cohort else None
        evaluation=verify_pair(row,result,order)
        save(dict(stage="paired",task=tid,order=order,full_role="v5 synchronous certified CoW",
                  pruned_role="v6 checkpoint waves",path_match=pair(a,b),cohort_path_match=cohort_match,
                  saved_executed_positions=a["executed_decode_positions"]-b["executed_decode_positions"],
                  saved_forward_calls=a["decode_calls"]-b["decode_calls"],cohort_reference=cohort,**result,**evaluation))
        print("wave",tid,"match",pair(a,b),"cohort",cohort_match,"positions",a["executed_decode_positions"],
              b["executed_decode_positions"],"calls",a["decode_calls"],b["decode_calls"],
              "seconds",round(a["seconds"],2),round(b["seconds"],2),flush=True)
        if cohort_match is False:
            raise RuntimeError("Cohort reference diverged; saved evidence")


if __name__=="__main__":
    main()
