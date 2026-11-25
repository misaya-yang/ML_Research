"""Paired last-block Power-SMC benchmark for the same local MBPP workload.

python power_race.py MODEL FOLDER phase 0 8 --particles 4 --shape fixed
python power_race.py MODEL FOLDER phase 0 8 --particles 4 --shape compact
Fixed physical slots prevent batch-shape changes; count every padded row.
Compact is the efficient comparator, but a changing BF16 batch shape can alter
logits. Matching tokens are checked per task, never assumed. The two algorithms
share frozen, independent time/slot token uniforms and final Gumbels.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from powersmc_mbpp import ancestors, reorder, weights
from ess_certificate import release_envelope
from resample_certificate import native_logz_floor, next_checkpoint_parents


def exponent(t, cfg):
    if t == 0:
        return 1.
    return 1. + (cfg.alpha-1.) * (min(t/cfg.ramp, 1.) if cfg.ramp else 1.)


def uniform_open(rng):
    value = rng.random()
    while value == 0:
        value = rng.random()
    return value


def random_tape(seed, cap, n):
    token_rng = random.Random(seed ^ 0x985D7E)
    ancestor_rng = random.Random(seed ^ 0x157C63)
    race_rng = random.Random(seed ^ 0xB13AE2)
    token_u = [[uniform_open(token_rng) for _ in range(n)] for _ in range(cap)]
    resample_u = [uniform_open(ancestor_rng) for _ in range(cap)]
    g = [-math.log(-math.log(uniform_open(race_rng))) for _ in range(n)]
    return token_u, resample_u, g


def draw_rows(logits, u, a):
    """FP64 inverse-CDF proposal with row-indexed uniforms; no RNG compaction."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logsum = np.log(np.exp(z).sum(axis=1))
    q = np.exp(a*z)
    cdf = np.cumsum(q, axis=1)
    picked = []
    for row, uniform in zip(cdf, u):
        # Search the unnormalized cumulative mass. The last strictly positive
        # interval handles multiplication rounded up to a finite endpoint.
        threshold = min(float(uniform*row[-1]), np.nextafter(row[-1], 0.))
        picked.append(int(row.searchsorted(threshold, side="right")))
    inc = np.clip(np.log(q.sum(axis=1))-a*logsum,native_logz_floor(a,z.shape[1]),0.)
    lp = np.minimum(0., z[np.arange(len(picked)), picked]-logsum)
    if not (np.isfinite(inc).all() and np.isfinite(lp).all()):
        raise ArithmeticError("Nonfinite token proposal or importance weight")
    return picked, lp.tolist(), inc.tolist()


def draw_rows_device(logits, uniforms, a, torch, physical_rows=None):
    """FP32 device categorical CDF, used identically by both paired paths.

    Normalizing by the last CDF entry guarantees an endpoint of one. Discrete
    FP32 CDFs can merge intervals below machine resolution; the CPU FP64
    sampler is a separately reported reference, not a bit-equivalent backend.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    tempered = a*lp
    z = torch.logsumexp(tempered, dim=-1).clamp(min=native_logz_floor(a,logits.shape[-1]),max=0.)
    q = torch.softmax(tempered, dim=-1)
    cdf = q.cumsum(dim=-1)
    cdf = cdf / cdf[:,-1:]
    u = torch.tensor(uniforms,device=logits.device,dtype=torch.float32).clamp(max=1.-2.**-24)
    # Equivalent to searchsorted(..., right=True) on each row. This works even
    # on MPS versions without a batched searchsorted kernel.
    if physical_rows is not None:
        ix=torch.tensor(physical_rows,device=logits.device,dtype=torch.long)
        selected=cdf.index_select(0,ix)
        ids=(selected <= u[:,None]).sum(dim=-1,dtype=torch.int32).long()
        base=lp[ix,ids].clamp(max=0.)
        z=z.index_select(0,ix)
    else:
        ids=(cdf <= u[:,None]).sum(dim=-1,dtype=torch.int32).long()
        base=lp.gather(1,ids[:,None])[:,0].clamp(max=0.)
    values = torch.stack((ids.float(),base,z),dim=1).cpu().tolist()
    if any(not math.isfinite(base+inc) for _,base,inc in values):
        raise ArithmeticError("Nonfinite device proposal")
    return values


def final_bound(w, prefix_lp, t, cap, cfg):
    """Upper bound for an active row, exact for a row that has already stopped.

    Apply the SAME floating-point ramp additions as full simulation. Future
    prefix_lp only decreases and all future log Z increments are nonpositive.
    """
    for step in range(t+1, min(cap, cfg.ramp)+1):
        w += (exponent(step, cfg)-exponent(step-1, cfg))*prefix_lp
    return w


def run(nat, row, cfg, seed, prune):
    import torch
    model, tok = nat.model, nat.tok
    device = next(model.parameters()).device
    sync = (torch.mps.synchronize if device.type=="mps" else
            torch.cuda.synchronize if device.type=="cuda" else lambda: None)
    n, cap = cfg.particles, nat.cap
    last_boundary = ((cap-1)//cfg.block)*cfg.block
    token_u, resample_u, g = random_tape(seed, cap, n)
    pool = list(range(n))
    rows = pool.copy()
    physical, slot_map = 1, dict.fromkeys(pool, 0)
    histories = [()] * n
    done = [False] * n
    reasons = ["cap"] * n
    lp, w = [0.] * n, [0.] * n
    dropped = set()
    held = set()
    pending = None
    certified_checkpoints = []
    events, resampling, decode_trace = [], [], []
    release_step = None
    release_reason = None
    metrics = dict(prefill_calls=1, prefill_positions=0, decode_calls=0,
                   executed_decode_positions=0, live_decode_positions=0,
                   padding_decode_positions=0, cow_shared_positions=0, sampled_tokens=0, stop_text_decodes=0, executed_proposal_rows=0,
                   terminal_sampled_tokens=0, terminal_decode_calls=0,
                   terminal_executed_positions=0, resample_count=0)
    timings = dict(prefill_s=0., decode_forward_s=0., proposal_s=0., stop_s=0.,
                   cache_reorder_s=0., race_s=0.)
    winner = None
    start = time.perf_counter()
    prompt = tok.apply_chat_template(nat.prompt(row), tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
    ids = prompt.input_ids.to(device)
    metrics["prefill_positions"] = ids.shape[1]
    head = model.get_output_embeddings()
    with torch.inference_mode():
        t0 = time.perf_counter()
        feed = model.model(input_ids=ids, use_cache=True)
        logits = head(feed.last_hidden_state[:, -1, :]).float()
        cache = feed.past_key_values
        del feed
        sync()
        timings["prefill_s"] += time.perf_counter()-t0
        for t in range(1,cap+1):
            a = exponent(t,cfg)
            t0 = time.perf_counter()
            for i in pool:
                if i not in dropped:
                    w[i] += (a-exponent(t-1,cfg))*lp[i]
            if rows:
                # Only rows whose next tokens are needed pay the sampling cost.
                # Physical decoder positions, including padding, are separate.
                if getattr(cfg,"sampler","cpu") == "device":
                    if cfg.shape=="fixed" and physical==n:
                        all_values=draw_rows_device(logits,token_u[t-1],a,torch)
                        info=[all_values[slot_map[i]] for i in rows]
                        proposal_cost=n
                    else:
                        query_rows=[slot_map[i] for i in rows]
                        alias=None if query_rows==list(range(physical)) else query_rows
                        info=draw_rows_device(logits,[token_u[t-1][i] for i in rows],a,torch,alias)
                        proposal_cost=physical
                    tokens=[int(x[0]) for x in info]
                    base_lp=[x[1] for x in info]
                    incr=[x[2] for x in info]
                else:
                    cpu_logits = logits.detach().cpu().numpy()
                    selected = cpu_logits[[slot_map[i] for i in rows]]
                    tokens, base_lp, incr = draw_rows(selected,[token_u[t-1][i] for i in rows],a)
                    proposal_cost=len(rows)
                metrics["executed_proposal_rows"] += proposal_cost
                for i, value, increment in zip(rows,base_lp,incr):
                    lp[i] += value
                    w[i] += increment
                metrics["sampled_tokens"] += len(rows)
                if t > last_boundary:
                    metrics["terminal_sampled_tokens"] += len(rows)
            timings["proposal_s"] += time.perf_counter()-t0
            t0 = time.perf_counter()
            fence_by_tape={}
            for i, token in zip(rows,tokens if rows else ()):
                histories[i] += (token,)
                if token in nat.stops:
                    done[i], reasons[i] = True, "eos"
                else:
                    tape=histories[i]
                    if tape not in fence_by_tape:
                        fence_by_tape[tape]=(tok.decode(list(tape),skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False).count("```")>=2)
                        metrics["stop_text_decodes"]+=1
                    if fence_by_tape[tape]:
                        done[i], reasons[i] = True, "fence"
            timings["stop_s"] += time.perf_counter()-t0
            parents = pool
            do_resample = False
            ess = None
            if pending is not None and t==pending["to"]:
                parents=pending["parents"]
                do_resample=True
                pending=None
                held.clear()
            elif t < cap and t % cfg.block == 0 and release_step is None:
                probs = weights(w)
                ess = 1./sum(p*p for p in probs)
                if ess < cfg.ess*n:
                    parents = ancestors(probs,resample_u[t-1])
                    do_resample = True
            if do_resample:
                histories = [histories[i] for i in parents]
                done = [done[i] for i in parents]
                reasons = [reasons[i] for i in parents]
                lp = [lp[i] for i in parents]
                w = [0.] * n
                resampling.append(dict(t=t,ess=ess,parents=parents))
            # At t==cap, unbiased systematic offspring followed by one uniform
            # child has marginal Cat(w). Both implementations use ONE same
            # Gumbel-max draw before this optional terminal resampling.
            t0 = time.perf_counter()
            if release_step is None and pending is None:
                if t >= last_boundary:
                    release_step, release_reason = t, "terminal_window"
                elif prune and cfg.early and any(done):
                    envelope = release_envelope(w, lp, done, t, cap, cfg.block, cfg.ess,
                                                lambda s: exponent(s,cfg),check_at_cap=False)
                    if envelope is not None:
                        release_step, release_reason = t, "all_future_ess_clear"
                if (prune and getattr(cfg,"ancestor_cert",True) and release_step is None
                        and any(done) and not all(done)):
                    checkpoint=(t//cfg.block+1)*cfg.block
                    if checkpoint<cap:
                        future=next_checkpoint_parents(w,lp,done,t,checkpoint,cfg.ess,
                                                       resample_u[checkpoint-1],lambda s: exponent(s,cfg),
                                                       vocab_size=logits.shape[-1],
                                                       equivalence=[(histories[j],lp[j],reasons[j]) if done[j] else None
                                                                    for j in pool])
                        if future is not None:
                            held={i for i in pool if not done[i] and i not in future}
                            if held:
                                pending=dict(start=t,to=checkpoint,parents=list(future),held=sorted(held))
                                certified_checkpoints.append(pending.copy())
            if release_step is not None:
                finished = [i for i in pool if i not in dropped and (done[i] or t==cap)]
                if finished:
                    bound = {i: final_bound(w[i],lp[i],t,cap,cfg) + g[i]
                             for i in pool if i not in dropped}
                    leader = max(finished,key=lambda i:(bound[i],-i))
                    if prune:
                        for i in pool:
                            if i not in dropped and not done[i] and t<cap and (bound[i],-i)<(bound[leader],-leader):
                                dropped.add(i)
                                events.append(dict(t=t,particle=i,leader=leader,
                                                   upper=bound[i],incumbent=bound[leader]))
                    if t==cap or all(done[i] or i in dropped for i in pool):
                        winner = leader
            timings["race_s"] += time.perf_counter()-t0
            if winner is not None:
                break
            rows = [i for i in pool if not done[i] and i not in dropped and i not in held]
            if not rows:
                continue
            t0 = time.perf_counter()
            if cfg.shape == "fixed":
                present = set(rows)
                take = [slot_map[parents[i]] if i in present else 0 for i in pool]
                next_ids = [histories[i][-1] if i in present else 0 for i in pool]
                cost = n
                next_map = {i:i for i in rows}
            else:
                # One transformer continuation per distinct (prior cache, token).
                # Equal maps imply identical full prefixes by induction from the
                # single shared prompt row. Each logical slot keeps its own RNG.
                groups={}
                next_map={}
                for i in rows:
                    key=(slot_map[parents[i]],histories[i][-1])
                    next_map[i]=groups.setdefault(key,len(groups))
                take=[old for old,token in groups]
                next_ids=[token for old,token in groups]
                cost=len(groups)
            cache = reorder(cache,take,physical,device,torch)
            physical, slot_map = cost, next_map
            token_tensor = torch.tensor(next_ids,device=device,dtype=torch.long).reshape(-1,1)
            sync()
            timings["cache_reorder_s"] += time.perf_counter()-t0
            t0 = time.perf_counter()
            feed = model.model(input_ids=token_tensor,past_key_values=cache,use_cache=True)
            logits = head(feed.last_hidden_state[:,-1,:]).float()
            cache = feed.past_key_values
            del feed
            sync()
            timings["decode_forward_s"] += time.perf_counter()-t0
            decode_trace.append([t+1,cost,sum(1 << i for i in rows)])
            metrics["decode_calls"] += 1
            metrics["executed_decode_positions"] += cost
            metrics["live_decode_positions"] += len(rows)
            metrics["padding_decode_positions"] += max(0,cost-len(rows))
            metrics["cow_shared_positions"] += max(0,len(rows)-cost)
            if t >= last_boundary:
                metrics["terminal_decode_calls"] += 1
                metrics["terminal_executed_positions"] += cost
    if winner is None:
        raise AssertionError("No winner at horizon")
    metrics["resample_count"] = len(resampling)
    tape = histories[winner]
    text = tok.decode(list(tape),skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return dict(winner=winner,token_ids=list(tape),code=nat.clean(text),reason=reasons[winner],
                seed=seed,pruned=events,resampling=resampling,seconds=time.perf_counter()-start,
                last_boundary=last_boundary,release_step=release_step,release_reason=release_reason,
                decode_trace=decode_trace,certified_checkpoints=certified_checkpoints,**metrics,**timings)


def oracle_ceiling(full, release, shape):
    """Clairvoyant, winner-only ceiling *after the certified release*.

    This uses the completed full trajectory for diagnostics only. It presumes
    the winner is known for free and is never used by run() to prune.
    """
    # Trace the selected node backward through every actual full-reference
    # ancestry update. In a fixed-width run even one required live token costs
    # a full physical batch; in compact mode its best possible cost is one.
    checks={x["t"]:x["parents"] for x in full["resampling"]}
    by_step={step:(physical,mask) for step,physical,mask in full["decode_trace"] if step>release}
    slot=full["winner"]
    calls=physical_total=live_total=winner_calls=oracle_physical=0
    for step in range(max([release,*checks,*by_step]),release,-1):
        if step in checks:
            slot=checks[step][slot]
        if step in by_step:
            cost,mask=by_step[step]
            calls+=1
            live_total+=mask.bit_count()
            physical_total+=cost
            if mask & (1<<slot):
                winner_calls+=1
                oracle_physical+=cost if shape=="fixed" else 1
    return dict(after_release=release,positions_at_risk=live_total,
                max_saved_live_positions=live_total-winner_calls,
                max_saved_executed_positions=physical_total-oracle_physical,
                max_saved_forward_calls=calls-winner_calls)


def verify_pair(row, result, order):
    from grade_batch import grade
    tests=row["test_list"]
    if len(tests)<2:
        raise ValueError("At least one public and one held-out test required")
    setup=row.get("test_setup_code","")
    public=[]
    public_wall={}
    for name in order:
        start=time.perf_counter()
        verdict,_=grade([[f"{name}/0",result[name]["code"],setup,tests[0]]])
        public_wall[name]=time.perf_counter()-start
        public.extend(verdict)
    # Reuse held-out test execution only if the candidate program bytes match.
    names=list(order)
    same=result["full"]["code"]==result["pruned"]["code"]
    offline_names=names[:1] if same else names
    jobs=[[f"{name}/{j}",result[name]["code"],setup,test] for name in offline_names
          for j,test in enumerate(tests) if j>0]
    start=time.perf_counter()
    hidden,_=grade(jobs)
    offline_wall=time.perf_counter()-start
    if same:
        hidden += [[r[0].replace(names[0]+"/",names[1]+"/",1),*r[1:]] for r in hidden.copy()]
    gradings=public+hidden
    verdicts={x[0]:x[1] for x in gradings}
    outcome={name:("abstain" if verdicts[f"{name}/0"]!="pass" else
                    "correct" if all(verdicts[f"{name}/{j}"]=="pass" for j in range(1,len(tests))) else "wrong")
             for name in names}
    return dict(gradings=gradings,public_verifier_wall_s=public_wall,offline_hidden_wall_s=offline_wall,
                outcome=outcome,end_to_end_s={name:result[name]["seconds"]+public_wall[name] for name in names})


def verify_log(path):
    path=Path(path)
    source=json.loads((path.parent/"mbpp_source.json").read_text())["rows"]
    tasks={entry["row"]["task_id"]:entry["row"] for entry in source}
    target=path.with_name(path.stem+"_public_cost.jsonl")
    seen={json.loads(x)["task"] for x in target.read_text().splitlines()} if target.exists() else set()
    for line in path.read_text().splitlines():
        record=json.loads(line)
        if record.get("stage")!="paired" or record["task"] in seen:
            continue
        task=record["task"]
        evaluation=verify_pair(tasks[task],record,record["order"])
        with target.open("a") as output:
            output.write(json.dumps(dict(task=task,**evaluation))+"\n")
        seen.add(task)
    print("Individually timed public checks and offline held-out checks:",target,flush=True)


def main():
    if len(sys.argv)==3 and sys.argv[1]=="--verify-log":
        verify_log(sys.argv[2])
        return
    parser=argparse.ArgumentParser(description=__doc__)
    for s in ("modeldir","folder","phase","begin","end"):
        parser.add_argument(s)
    parser.add_argument("--particles",type=int,default=4)
    parser.add_argument("--alpha",type=float,default=4.)
    parser.add_argument("--block",type=int,default=64)
    parser.add_argument("--ess",type=float,default=.5)
    parser.add_argument("--ramp",type=int,default=100)
    parser.add_argument("--seed",type=int,default=4728103)
    parser.add_argument("--shape",choices=["fixed","compact"],default="fixed")
    parser.add_argument("--ancestor-cert",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--sampler",choices=["device","cpu"],default="device")
    parser.add_argument("--early",action=argparse.BooleanOptionalAction,default=True,
                        help="Prove all future nonterminal ESS checks clear before entering the race")
    cfg=parser.parse_args()
    if cfg.particles<1 or cfg.block<1 or cfg.ramp<0 or not 0<cfg.ess<=1 or not math.isfinite(cfg.alpha) or cfg.alpha<1:
        raise ValueError("Invalid parameters")
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"independent_mh_screen"))
    sys.argv=sys.argv[:6]
    import native_sampling as nat
    params=dict(shape=cfg.shape,n=cfg.particles,cap=nat.cap,block=cfg.block,alpha=cfg.alpha,
                ess=cfg.ess,ramp=cfg.ramp,early=cfg.early,ancestor_cert=cfg.ancestor_cert,sampler=cfg.sampler,seed=cfg.seed,phase=cfg.phase,version=5)
    tag=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]
    log=Path(cfg.folder)/f"power_race_{tag}.jsonl"
    def save(x):
        with log.open("a") as handle:
            handle.write(json.dumps(x,ensure_ascii=False)+"\n")
    seen={r["task"] for line in (log.read_text().splitlines() if log.exists() else [])
          if (r:=json.loads(line)).get("stage")=="paired"}
    save(dict(stage="loaded",params=params,load_s=nat.load_time))
    for e in nat.tasks:
        row=e["row"]; tid=row["task_id"]; idx=int(e["row_idx"])
        if tid in seen:
            continue
        seed=cfg.seed+1009*idx
        order=("pruned","full") if idx%2 else ("full","pruned")
        result={name:run(nat,row,cfg,seed,name=="pruned") for name in order}
        full,pruned=result["full"],result["pruned"]
        path_match=(full["winner"]==pruned["winner"] and full["token_ids"]==pruned["token_ids"])
        evaluation=verify_pair(row,result,order)
        save(dict(stage="paired",task=tid,order=order,path_match=path_match,
                  same_code=full["code"]==pruned["code"],oracle_only=oracle_ceiling(full,min([pruned["release_step"],*[e["start"] for e in pruned["certified_checkpoints"]]]),cfg.shape) if path_match else None,
                  saved_executed_positions=full["executed_decode_positions"]-pruned["executed_decode_positions"],
                  saved_forward_calls=full["decode_calls"]-pruned["decode_calls"],
                  saved_live_positions=full["live_decode_positions"]-pruned["live_decode_positions"],
                  **evaluation,**result))
        print("race",tid,"matched",path_match,"positions",full["executed_decode_positions"],
              pruned["executed_decode_positions"],"seconds",round(full["seconds"],2),round(pruned["seconds"],2),flush=True)
        if not path_match and cfg.shape=="fixed":
            raise RuntimeError(f"Fixed-slot numerical/path discrepancy on {tid}; saved both outputs")


if __name__=="__main__":
    main()
