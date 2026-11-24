"""Shared-prefix decoder with left-padded KV state for unequal-depth batches.

Copied from results/local_runs/powermh-native-73eddb4b/code/gpu_depth_two/batched_prefix.py
(the repository's depth-two comparator). Changes: each model call records its own wall time,
the engine tick that issued it, and the labels of the cursors it served; `tick` is set by the engine.
"""
import copy
import math
import time
import numpy as np
import torch


class Store:
    """All prefix nodes on a nonbranching path share a movable reference to its latest KV row."""
    def __init__(self,cache,row,pad,length):
        self.cache,self.row,self.pad,self.length=cache,row,pad,length


class Node:
    def __init__(self,store,length,raw_logits,alpha):
        self.store,self.length,self.children=store,length,{}
        lp=torch.log_softmax(raw_logits.detach().float().cpu().double(),dim=-1).numpy()
        mass=alpha*lp
        m=float(mass.max());z=float(m+np.log(np.exp(mass-m).sum()))
        self.logz=min(0.,z)
        cdf=np.cumsum(np.exp(mass-z))
        distinct=np.flatnonzero(np.r_[True,cdf[1:]!=cdf[:-1]])
        self.ids=distinct.astype(np.int32);self.cdf=cdf[distinct]

    def sample(self,u):
        threshold=min(u*self.cdf[-1],np.nextafter(self.cdf[-1],0.))
        return int(self.ids[np.searchsorted(self.cdf,threshold,side='right')])


class Decoder:
    def __init__(self,model,ids,alpha,physical_min,device):
        self.model,self.alpha,self.physical_min,self.device=model,alpha,physical_min,device
        self.sync=(torch.mps.synchronize if device=='mps' else torch.cuda.synchronize if device.startswith('cuda') else lambda:None)
        self.stats=dict(prefill_calls=1,prefill_inputs=len(ids),decode_calls=0,decode_inputs=0,
            live_unique_decode_inputs=0,dummy_decode_inputs=0,cached_prefix_hits=0,same_batch_shared=0,
            requested_continuations=0,kv_repack_calls=0,kv_fast_reuse_calls=0,kv_left_pad_slots=0,
            prefill_s=0.,decode_s=0.,cache_packing_s=0.,cdf_and_bookkeeping_s=0.)
        self.trace=[];self.tick=0
        self.sync();t=time.perf_counter()
        data=torch.tensor([ids],device=device,dtype=torch.long)
        out=model(input_ids=data,attention_mask=torch.ones_like(data),position_ids=torch.arange(len(ids),device=device)[None,:],use_cache=True,logits_to_keep=1)
        cache=out.past_key_values;self.sync();self.stats['prefill_s']=time.perf_counter()-t
        if not hasattr(cache,'layers') or any(type(x).__name__!='DynamicLayer' for x in cache.layers):
            raise TypeError('Expected a Qwen DynamicCache with DynamicLayer; this runner has no sliding-window cache support.')
        self.root=Node(Store(cache,0,0,len(ids)),len(ids),out.logits[0,-1],alpha)

    def _pack(self,nodes):
        longest=max(x.length for x in nodes);pads=[longest-x.length for x in nodes]
        first=nodes[0].store.cache
        if (all(x.store.cache is first and x.store.row==i and x.store.pad==pads[i] for i,x in enumerate(nodes))
            and first.layers[0].keys.shape[0]==len(nodes) and first.get_seq_length()==longest):
            self.stats['kv_fast_reuse_calls']+=1
            return first,pads
        # All old backing rows remain valid for their recorded prefix lengths. New
        # branch cache shell may share views; DynamicLayer appends with torch.cat.
        self.stats['kv_repack_calls']+=1
        cache=copy.copy(first);cache.layers=[]
        for layer_idx,template in enumerate(first.layers):
            layer=copy.copy(template)
            for kind in ('keys','values'):
                pieces=[]
                for node,new_pad in zip(nodes,pads):
                    st=node.store;arr=getattr(st.cache.layers[layer_idx],kind)
                    piece=arr[st.row:st.row+1,:,st.pad:st.pad+node.length,:]
                    if new_pad:piece=torch.nn.functional.pad(piece,(0,0,new_pad,0))
                    pieces.append(piece)
                setattr(layer,kind,pieces[0] if len(pieces)==1 else torch.cat(pieces,dim=0))
            cache.layers.append(layer)
        return cache,pads

    def children(self,edges,labels=None):
        self.stats['requested_continuations']+=len(edges)
        missing={}
        for node,v in edges:
            if v in node.children:self.stats['cached_prefix_hits']+=1
            elif (node,v) in missing:self.stats['same_batch_shared']+=1
            else:missing[node,v]=len(missing)
        if missing:
            keys=list(missing);n=len(keys);width=max(n,self.physical_min)
            padded=keys+[keys[-1]]*(width-n);nodes=[x[0] for x in padded]
            # Snapshot old store metadata before model forward mutates a fast-reused cache.
            old=[(node.store,node.store.length==node.length and not node.children) for node,v in keys]
            self.sync();t=time.perf_counter()
            cache,pads=self._pack(nodes)
            data=torch.tensor([[v] for _,v in padded],device=self.device,dtype=torch.long)
            positions=torch.tensor([x.length for x in nodes],device=self.device,dtype=torch.long).view(-1,1)
            ticks=torch.arange(max(x.length for x in nodes)+1,device=self.device)
            mask=(ticks[None,:]>=torch.tensor(pads,device=self.device,dtype=torch.long)[:,None]).long()
            self.sync();self.stats['cache_packing_s']+=time.perf_counter()-t;t=time.perf_counter()
            out=self.model(input_ids=data,attention_mask=mask,position_ids=positions,past_key_values=cache,use_cache=True,logits_to_keep=1)
            self.sync();self._last_forward_s=time.perf_counter()-t;self.stats['decode_s']+=self._last_forward_s;t=time.perf_counter()
            used=set()
            for i,(node,v) in enumerate(keys):
                ref,tip=old[i]
                if not tip or ref in used:ref=Store(out.past_key_values,i,pads[i],node.length+1)
                else:
                    used.add(ref);ref.cache,ref.row,ref.pad,ref.length=out.past_key_values,i,pads[i],node.length+1
                node.children[v]=Node(ref,node.length+1,out.logits[i,-1],self.alpha)
            self.stats['cdf_and_bookkeeping_s']+=time.perf_counter()-t
            self.stats['decode_calls']+=1;self.stats['decode_inputs']+=width
            self.stats['live_unique_decode_inputs']+=n;self.stats['dummy_decode_inputs']+=width-n
            self.stats['kv_left_pad_slots']+=sum(pads)
            consumers=[[] for _ in range(n)]
            for edge,label in zip(edges,labels or ['unlabelled']*len(edges)):
                if edge in missing:consumers[missing[edge]].append(label)
            self.trace.append(dict(tick=self.tick,width=width,live=n,real_prefix_lengths=[x.length for x,_ in keys],
                                   real_row_consumers=consumers,cache_left_pad=pads,
                                   forward_s=self._last_forward_s))
        return [node.children[v] for node,v in edges]
