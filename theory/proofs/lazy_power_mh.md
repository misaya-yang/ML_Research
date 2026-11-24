# Exact early rejection for power-sampling MH and its read-probability lower bound

Drafted by Claude on 2026-11-23. Status: proof written here; exact rational check passes ([check_lazy_mh.py](../../experiments/first_passage/check_lazy_mh.py)); the user re-derived the core theorem independently; human line-by-line check of the full note pending. This is the score-cap instance of the general read rule in [certified_thinning](certified_thinning.md).

**Scope in one paragraph.**
- *Exactness.* Under the same proposal and random numbers, early rejection preserves every accept/reject decision and every chain state.
- *Optimality.* It holds only in this access model: units are revealed one token at a time at additive per-position prices, and the exact law of the whole trajectory must be reproduced. Pathwise replay is not required, but the full trajectory law is.
- *Where the optimality claim stops.* It says nothing about methods that only need the final-answer law, cost models that score several positions per call, or white-box access.
- *Prior art.* Early rejection is classical MCMC.
- *What is specific here.* For sharpening proposals, $\log Z_\alpha(h)=-(\alpha-1)H_\alpha(p(\cdot\mid h))\le0$. A language model therefore supplies an exact monotone bound for free, and the MH test becomes a comparison of accumulated Rényi entropies. The open quantity is how much this saves on top of strong exact baselines.

## Policy and access model

Fix a prompt, a base model $p(v\mid h)$, an exponent $\alpha\ge1$ and a logical horizon $T$. Use the fixed-horizon convention: after EOS every position is a deterministic pad with $p(\mathrm{pad}\mid h)=1$, so the chain needs no length correction ([Parallel Power Tempering, §3.3](https://arxiv.org/abs/2609.38104)). The target is $\pi(x)\propto\prod_{s<T}p(x_s\mid h_s)^\alpha$ with $h_s=x_{<s}$. Write
$$
Z(h)=\sum_v p(v\mid h)^\alpha\le1,\qquad q(v\mid h)=p(v\mid h)^\alpha/Z(h).
$$
One MH step from state $x$ draws a cut $c\sim\mathrm{Unif}\{0,\dots,T-1\}$, keeps $x_{<c}$, regenerates $x'_{c:T}$ from $q$, and accepts with probability
$$
A(x,x')=\min\Bigl\{1,\ \prod_{s>c}Z(h'_s)\Big/\prod_{s>c}Z(h_s)\Bigr\}.
$$
Since $\sum_v p^\alpha=\exp((1-\alpha)H_\alpha)$, the ratio equals $\exp(-(\alpha-1)[\sum_{s>c}H_\alpha(h'_s)-\sum_{s>c}H_\alpha(h_s)])$. It is the Metropolis–Hastings ratio for suffix resampling with proposal temperature $1/\alpha$ ([Karan and Du](https://arxiv.org/abs/2510.14901); [PreSTO, eq. 6](https://arxiv.org/abs/2610.13171)): the per-token factor $p^\alpha/q$ equals $Z$, and the $s=c$ factor cancels because $h'_c=h_c$. The same algebra holds for a top-$p$ or top-$k$ truncated proposal with $Z$ replaced by the truncated sum, which is still at most one. The policy runs $K$ steps from $x^{(0)}\sim q$.

**Units and prices.** A forward pass at a new non-padding prefix $h$ reveals $p(\cdot\mid h)$, hence $Z(h)$ and the next-token law. This is the paid unit. Prefixes already read are cached and free; pads are free. Policy randomness (cuts, $U$'s, token uniforms indexed by step and position) is free. **Trajectory contract:** a simulator is exact if, for every admissible model, its output $(x^{(0)},\dots,x^{(K)})$ has the policy's law. Admissible models are all next-token kernels on the vocabulary extended by one fresh token $\star$; the policy's own model need not use $\star$.

## Lazy rule and pathwise identity

Draw $U$ before generating. Let $\Phi=\sum_{s>c}\log Z(h_s)$, known from the current state's own generation. After each paid forward pass at $h'_s$ ($s>c$) update $S_s=\sum_{c<r\le s}\log Z(h'_r)$ and reject as soon as $S_s<\log U+\Phi$.

**Proposition 1.** Under shared $U$ and token uniforms, the lazy and eager steps return the same decision and the same next state.

*Proof.* Each $\log Z\le0$, so $S_s$ is non-increasing and its final value is $\log$ of the ratio in $A$. If lazy stops, the full ratio is below $U$ and eager rejects. If lazy never stops, it has generated the whole proposal and applies the same test. Tokens drawn before the stop are identical because the uniforms are indexed by position. $\square$

**Variant with cuts inside the current response.** Many implementations draw $c\sim\mathrm{Unif}\{0,\dots,n-1\}$ inside the current response of length $n$ and stop each proposal at the native stop rule (EOS, a closing code fence, or a cap). The reverse move then has probability $1/n'$. The exact ratio is
$$
\exp\Bigl(\sum_{s>c}\log Z(h'_s)-\Phi\Bigr)\,\frac{n}{n'}.
$$
After $j$ paid decode forwards the proposal already has $n'\ge c+j+1$ tokens, so
$$
B_j=\exp(S_j-\Phi)\,\frac{n}{c+j+1}
$$
is non-increasing in $j$ and equals the exact ratio when the proposal stops. Proposition 1 and the cost formula carry over with $B_j$ in place of $\exp(S_j-\Phi)$. Theorem 2 carries over when the fresh token $\star$ also terminates the response, so that the altered world attains $n'=c+j+1$. This variant is the one used in the MBPP Instruct pilot ([powermh_mbpp_pilot.py](../../experiments/first_passage/powermh_mbpp_pilot.py)). Omitting the $n/n'$ factor changes the policy, as the truncation-bias analysis of Parallel Power Tempering shows.

The expected number of paid forward passes for one proposal with tape $x'$ is
$$
\sum_{s=c+1}^{e}\min\Bigl\{1,\ \exp\Bigl(\sum_{c<r<s}\log Z(h'_r)-\Phi\Bigr)\Bigr\},
$$
where $e$ is the last non-padding position. The eager count is $e-c$. Both are computable from eager logs, so the lazy saving can be predicted before an accelerator is built.

## Theorem: read probability equals counterfactual influence

**Theorem 2.** Fix an admissible model $p$ and a non-root, non-padding prefix $h$. Let $p^{h\star}$ agree with $p$ off the subtree of $h$ and emit $\star$ deterministically at $h$ and every extension of $h$. Then
$$
\Pr_p(\text{lazy ever pays for }h)=\mathrm{TV}\bigl(\mathcal L_p(x^{(0:K)}),\ \mathcal L_{p^{h\star}}(x^{(0:K)})\bigr),
$$
and every trajectory-exact simulator pays for $h$ under $p$ with at least this probability. Summing over prefixes, lazy minimizes every nonnegative additive cost among trajectory-exact simulators in this access model, instance by instance.

*Proof.* (i) *Lower bound for any simulator.* Couple runs under $p$ and $p^{h\star}$ with the same internal randomness. The two models answer every query identically until $h$ is read, so the transcripts and outputs agree on the event that $h$ is never read. Hence the output TV is at most the probability of reading $h$, for any exact simulator, and in particular for lazy.

(ii) *Lazy attains it.* Run lazy under the same coupling. Before $h$ is first read, the two worlds evolve identically and $\star$ cannot appear. If $h$ is first read during the initial draw, $x^{(0)}$ contains $\star$ under $p^{h\star}$. If it is first read inside a proposal at step $m$, lazy reached $h$ with $U\le\exp(S_{<h}-\Phi)$. Under $p^{h\star}$ every factor from $h$ on equals one, so the full ratio is $\exp(S_{<h}-\Phi)$ and the proposal is accepted on the same event (up to a null tie), placing a $\star$-state in the trajectory. Thus $\Pr_{p^{h\star}}(\star\text{ appears})=\Pr_p(\text{lazy reads }h)$. Under $p$ no trajectory contains $\star$, so the trajectory TV is at least this probability. Combine with (i). $\square$

Without a fresh token, choose a deterministic completion $z$ of $h$ instead of $\star$; the bound holds up to $\inf_z\Pr_p(\text{trajectory contains }(h,z))$, which is tiny for a full-softmax model. The theorem is a per-unit statement in the stated black-box model. It does not price batching, prefill versus decode, shared KV memory, or parallel depth, and it does not cover simulators with white-box access to the network.

## The output contract matters

If only the final state $x^{(K)}$ must have the right law, the same coupling gives the weaker bound $\mathrm{TV}(\mathcal L_p(x^{(K)}),\mathcal L_{p^{h\star}}(x^{(K)}))$. In the exact check this bound is strictly below the lazy read probability for **270/270** units over 33 random models, by up to **63.7%** relatively. A strictly smaller lower bound does not by itself give a cheaper algorithm. Whether some output-only simulator beats lazy, and by how much on real chains, is open. The four-probe Power-SMC witness in [finite_particle_result](../../results/local_runs/native-firstpass-1qr7b1va/feedback_policy_v2/finite_particle_result.md) is a different example where a contract change alters what must be read.

## Certificate tightness sets the exact-speedup ceiling

The theorem's mechanism is generic. Call a decision a **threshold race** when stream $i$ is accepted iff an independent policy key satisfies $\kappa_i\le B_i$, where $B_i$ is a function of the stream's full content, the threshold side is already known or determined by other streams independently of $\kappa_i$, and a certified bound $\bar B_{i,k}\ge B_i$ is available after $k$ units and is non-increasing in $k$. Examples:

| Policy | Key | Certified bound after $k$ units | Note |
|---|---|---|---|
| PowerMH step (this note) | $U$ | $\exp(S_k-\Phi)$ | $\log Z\le0$ |
| Weighted order with dedup and verification budget | $E_i/T_{-i}$ | $u_{i,k}$ | [stopped_weighted_ranking](stopped_weighted_ranking.md) |
| Rejection sampling with envelope $M$ | $U_i$ | partial likelihood ratio over $M$ | classical early rejection |
| Power-SMC terminal selection | Gumbel | replayed ramp on prefix log-probability | [single_output_ancestry](../../results/local_runs/native-firstpass-1qr7b1va/feedback_policy_v2/single_output_ancestry.md) |
| Best-of-$N$ with an outcome reward model, softmax selection | Gumbel | global $r_{\max}/\beta$ only | no prefix certificate |

Lazy evaluation reads unit $k$ of stream $i$ iff $\kappa_i\le\bar B_{i,k-1}$. If the unread part of a stream can be replaced, within the admissible class, so that $B_i$ attains $\bar B_{i,k-1}$ with a new distinguishable output, the coupling argument above makes this read probability necessary. **The exact saving is therefore $\sum_{\text{units}}[\Pr(\text{reach})-\Pr(\kappa\le\bar B_{k-1})]$, governed entirely by how fast certified bounds tighten.** For outcome-reward best-of-$N$ the only certificate is the global maximum, so an exact simulator can stop a candidate only on the key gap; prefix-based pruning such as [speculative rejection](https://arxiv.org/abs/2410.20290) necessarily changes the output law. Likelihood-tempered policies (power sampling, Power-SMC, likelihood-ranked sampling) carry free monotone certificates, which is why exact acceleration is possible there. The general threshold-race theorem with a multi-stream threshold is proved for the ranking case in its own note; a single statement covering all rows is still to be written.

## Relation to prior work

- Early rejection for likelihoods that are products of factors at most one is classical in MCMC ([Solonen et al., 2012](https://doi.org/10.1214/12-BA724)). The new content is the instance-wise read-probability lower bound that it attains, the trajectory/output contract distinction, and the application to token-level LLM samplers.
- Delayed acceptance ([Christen and Fox, 2005](https://doi.org/10.1198/106186005X76983); [Banterle et al.](https://arxiv.org/abs/1503.00996)) preserves the stationary target but changes the finite-step kernel. At $K=10$–$100$ steps PowerMH is far from stationarity, so only a kernel-exact accelerator is guaranteed quality-neutral.
- PreSTO prefetches future proposals and leaves the kernel unchanged, reporting median 1.71× wall-clock on PowerMH. It generates every proposal in full and proves no lower bound ([arXiv 2610.13171](https://arxiv.org/abs/2610.13171)). Early rejection is orthogonal: it reduces work per proposal, and prefetched speculative proposals can also be cut early.
