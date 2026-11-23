# Trimming failure suffixes under finite budgets using native control marks

## Access model and result

Fix the original policy, initial history, and hard step limit $B$. At step $j$, from the actual history $h$, first sample the native control action $C\sim\pi_h$. Then pay to generate a result under the original conditional law, check whether it passes, and update the history. On acceptance, return the identity, payload, original step, and cumulative logical cost at that point.

Let a cheaply available action set $D(h)$ cover every action that could accept at that step. Assume exact access to $s(h)=\pi_h(D(h))$ and conditional sampling from $\pi_h$ on $D(h)$ and its complement. For each $j$, take a predetermined $c_j\in[0,1]$ such that $s(h)\le c_j$ on every continuation history reachable by the original policy at step $j$. The bound must hold before obtaining unknown future outputs. Generate and update using the original policy's conditional law given history and action. If additional controller random state is carried into future histories, retain it under conditional sampling.

Draw independent $M_j\sim\operatorname{Bern}(c_j)$. Let $T$ be the last positive-mark index, or $T=0$ if there is none. At step $j\le T$, draw the action from the complement if $M_j=0$. If $M_j=1$, enter $D(h)$ with probability $s(h)/c_j$, then draw from the corresponding conditional law. Pay for the original generation and update. Stop on the original policy's acceptance or logical stopping condition. Return failure if no result has passed by $T$.

This procedure preserves the exact first-accepted identity, payload, original step, cumulative logical cost at success, and deadline failure probability. Simulation steps on successful paths are unchanged. The work saved is the failure suffix the original policy would execute after $T$.

## Coupling proof

On a common probability space, first construct a reference execution that reaches the hard limit or stops natively. Given the history before step $j$, $M_j$ is independent of that history. If $c_j>0$, the conditional probability of choosing $D(h)$ is exactly $c_j\,s(h)/c_j=s(h)$. The original conditional laws on both sides give $C$ marginal law $\pi_h$. If $c_j=0$, then $s(h)=0$ and the draw comes directly from the complement. If $s(h)=1$, then $c_j=1$ and the draw comes directly from the carrier. No zero-probability branch is invoked.

By induction, the reference execution has the full original process law, including any adaptive updates. Acceptance is possible only when $M_j=1$, so all acceptances occur no later than $T$. The accelerator uses the same reference prefix and preserves the original stop and acceptance rules. For fixed coupled randomness, it returns the same success record if the original policy succeeds, and fails if the original policy fails. The implementation knows $T$ and the unseen marks; the original controller, emission, and update do not read them.

The last mark can be drawn directly from the distribution

$$
\Pr(T=0)=\prod_{j=1}^B(1-c_j),\qquad
\Pr(T=t)=c_t\prod_{j=t+1}^B(1-c_j)\quad(1\le t\le B).
$$

Conditional on $T=t>0$, the mark $M_t=1$, all later marks are zero, and earlier marks remain independent with their original probabilities. Scan backward from $B$ to find the first positive mark, and draw earlier marks only as needed. For constant $0<c<1$, the distance from the end to the first mark is truncated geometric. At $c=0$, the whole segment can be skipped; at $c=1$, the final step always has a mark. Even when $T$ is known in advance, logical steps keep their original indices.

This construction of independent marks dominating all real acceptance times corresponds to classical thinning. Updates between positive marks can still change future success probabilities, so conditional native actions and emissions remain necessary until the last mark. [Lewis and Shedler](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800260304)

## Dynamic original budgets and independent restarts

Within the hard limit $B$, the original policy may stop dynamically based on history-dependent costs, feasibility rules, or resource thresholds. The same coupling applies if stop and acceptance decisions can be computed incrementally from the revealed reference prefix. On success, preserve the full logical cost, including the increment used to decide whether that success is in budget. On failure, the accelerator returns $\bot$. An omitted failure suffix can change the original stopping time, its failed terminal state, and its variable-length total cost ledger; these cannot be inferred from $\bot$. If wall-clock timing in an asynchronous system changes the original policy's decisions, first specify how to reproduce its logical scheduling.

If each restart resets independently to a specified initial state and uses a fixed logical allowance, sample a separate last mark for each restart. A failed restart’s terminal state is not carried forward, so its own suffix can be omitted even if a later restart has a positive mark. For identical independent restarts, zero-mark restarts can also be skipped in batches: the number of consecutive zero-mark restarts is truncated geometric. If the next restart depends on the preceding failed terminal state, or the remaining global allowance uses variable deductions from the previous failure, the block method must restore the original entry state and ledger.

## Call accounting and a lower bound under the same interface

Let $N$ be the number of paid steps when the original execution stops. Under the same coupling, the accelerator pays for $\min(N,T)$ steps. All savings come from paths that ultimately fail under the original policy. With fixed positive costs $a_j$, the expected cost saved is $\sum_j a_j\Pr(T<j\le N)$. If every path fails and runs all $B$ steps, this gives the upper bound

$$
\mathbb E[B-T]=\sum_{i=1}^B\prod_{j=i}^B(1-c_j).
$$

For constant $c>0$, this is $(1-c)[1-(1-c)^B]/c$; for $c=0$, it is $B$. Thus at fixed $c$ and large $B$, the absolute savings are bounded by a suffix of roughly $1/c$ steps. This is the maximum suffix saving under the stated interface; actual success rates and dynamic stopping affect the gain.

The lower bound uses the same interface. Let each native action be a binary mark with probability $c_j$, with the positive action as carrier. The expensive emission at step $i$ reveals a private bit $\theta_i$ only when address $i$ is queried. Any observed positive bit permanently turns on a switch; at each step, read the bit first and then allow execution to pass on a positive action if the switch is on. The initial switch setting, off, is public. The original policy always fails in the all-zero world. In the world that changes only bit $i$ to one, the reference success probability is

$$
p_i=1-\prod_{j=i}^B(1-c_j).
$$

For any algorithm exact on all these instances, couple the all-zero world and the world with only bit $i$ positive using the same internal randomness. Observed data are identical until address $i$ is queried. The all-zero world cannot produce a success, so each success in the single-positive world must be preceded by a query to $i$. The probability of querying $i$ in the all-zero world is therefore at least $p_i$. The accelerator expands step $i$ in the all-zero world exactly when $T\ge i$, with probability exactly $p_i$. Weight each address by any given nonnegative price. This gives an exact address-wise lower bound on the all-zero instance in this fixed access model. Summaries, parallel bulk queries, or other public interfaces revealing multiple private bits at once change the bound. If the switch resets on every restart, the same construction gives the corresponding lower bound independently for each restart.

## Random-scan single-coordinate repair

A state has $n$ coordinates; the original controller chooses a coordinate uniformly and pays to generate a new value. At every failing state, choose one currently violated local constraint whose scope $D(h)$ contains at most $k$ coordinates. A single-coordinate update outside that scope leaves the constraint violated, so the step cannot pass. Take $s(h)=|D(h)|/n$ and $c_j=k/n$. Both conditional control draws are just uniform samples from the relevant coordinate mask: no full-sequence density, sampling from correct outcomes, or reverse kernel is needed. Charge the cost of finding the violated constraint and its mask.

Sequential caching and early rejection can avoid repeated costs or skip verification that is known to fail at the current step. However, an update outside the scope still needs to be computed if it changes the next selected constraint or acceptance probability. The algorithm omits a suffix of updates known not to pass later either. If the baseline already uses the same coupled marks to identify the suffix, or a cheaper state projection can replay the necessary updates, compare against it. The proof gives an exact mechanism and call counts, not a bound on total FLOPs or end-to-end latency.

Run [marked_horizon.py](../../experiments/first_passage/marked_horizon.py) to enumerate rational results for adaptive updates, dynamic resource limits, random-scan masks, and the lower-bound instances.
