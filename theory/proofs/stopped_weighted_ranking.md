# First passage under weighted verification order and the necessary read probability

## Target and forward interface

Condition on the original branch randomness for one search and on the already-paid model prefix. There are \(n\) slots; completing slot \(i\) gives text \(z_i\) and positive weight \(w_i\). Different slots can generate the same text. Using \(E_i\sim\mathrm{Exp}(1)\) independent of branch generation, the consumer reads in ascending \(E_i/w_i\) order, deduplicates by original text, verifies no more than \(B\) distinct identities, and returns the first passing item or abstains. The verifier is deterministic. The output includes the program, winning slot, original ordered verification record, and its length. Equal weights are handled in the same random order. The implementation uses independent standard Gumbels \(g_i=-\log E_i\); ties in floating point are ordered by slot index.

A forward call generates a segment of a slot and returns a currently certified bound \(u_{i,k}\ge w_i\), also returning \(z_i\) upon completion. The bound is positive and nonincreasing. It may come from an explicit original-policy score. For example, in Power-SMC after the final resampling, a single step adds log weight \(\log\sum_v p(v\mid h)^\alpha\le0\) for fixed \(\alpha\ge1\). If \(\alpha\) increases monotonically, also multiply each future sharpening-exponent increment by the current nonpositive prefix log-likelihood and include it in the original implementation order; each term still yields an upper bound. Native intermediate resampling can change the whole population and its weights. It must occur before this algorithm can be applied to the slots, or be ruled out by a separate valid certificate. Exact implementation of a finite-particle policy is not the same as exact sampling from a stationary target.

## Execution and correctness

Maintain a min-heap. Unfinished slots use the optimistic key \(E_i/u_{i,k}\), and completed slots use the true key \(E_i/w_i\). If a popped slot is unfinished, read another segment and reinsert it. Otherwise, its true key is no larger than any as-yet-unprocessed true key; invoke the verifier only for new text. Continue in the original order until acceptance, zero remaining verification budget, or exhaustion.

The loop does not need the success rate, reverse transitions, or a generator conditional on acceptance. Every text submitted to the verifier is the next new text in the original order. Induction gives identical output, verification order, and logical deadline event for each fixed random tape. Integrating over branch randomness gives the distributional result. The coupling requires that reordering computation not change the numerical transition in a slot, for example by using consistent row shapes and independent per-slot random streams.

## Theorem: read probability of each private prefix segment

Fix a complete instance with \(B>0\). Remove slot \(i\) and run the same rule on the other slots in true-key order. Let \(T_{-i}\) be the earlier of the key of the first accepted item and the key of the \(B\)th distinct text. If neither occurs, set it to \(+\infty\). It is independent of \(E_i\). Just before reading slot \(i\)'s \(k\)th private segment, its bound is \(u_{i,k-1}\). For each paid segment that actually exists,

\[
 \Pr(\text{read }i,k)
 =\Pr(E_i/u_{i,k-1}<T_{-i})
 =\mathbb E[1-\exp(-u_{i,k-1}T_{-i})].
\]

If \(B=0\), there is nothing to read. Under a continuous law, there are no ties across slots. Compare discrete machine keys as ordered pairs including the slot index. The read event still holds, but the continuous integral formula does not apply to a discrete random source.

Proof: if the optimistic key precedes the stopping key after removing slot \(i\), earlier completed items from other slots cannot satisfy the stopping rule. Earlier bounds for \(i\) are also no later than this key, so this segment must be read. Conversely, the other slots can finish in the original order through their stopping point, without expanding a lower-priority optimistic key. By independence and the exponential law of \(E_i\), integrating conditional on all other slots gives the formula. The algorithm only needs the heap; it never computes this expectation over future information online.

**Lower bound under the same interface.** Suppose segments are private and must be revealed in order. Holding all revealed information fixed, suppose the interface allows replacing only the unread part of the slot by a completion of weight \(u_{i,k-1}\) whose identity is a new passing text, while keeping all other slots and all free information unchanged. Under this replacement, the original policy returns the new identity exactly on the event above. In the unmodified instance, the probability of returning that identity is zero. Couple any exact-distribution simulator on the two instances using the same private randomness. The simulator's observations agree until it first reads this segment. Thus the output total variation distance is at most the probability of reading this segment in the unmodified instance. The necessary read probability for each private segment therefore equals the expression above. If the maximum weight is approachable but not attainable, take the supremum. If feasible passing completions have a smaller maximum weight than the engineering bound, the lower bound uses only an attainable weight.

If the conditions hold simultaneously for every segment, multiply by the respective segment costs and sum to obtain instance-wise optimality for any nonnegative additive cost. Account separately for parallel wall time, batch widths, shared token prefixes, known aliases, and the global parameter constraints of a fixed network. In particular, do not sum private-slot bounds multiple times over slots sharing a forward call or conclude that the theorem optimizes all possible exact algorithms on a particular Transformer. A free semantic check ruling out any new passing item from a slot eliminates the two-world construction for that slot.

## Relation to prior work and the strongest comparator

Exponential races, Gumbel ranking, and lazy sampling with upper bounds are established techniques; optimistic-key expansion is a direct application. This result concerns the read threshold jointly determined by verification budget, repeated identities, and the first accepted output. The lower bound also applies to exact simulators allowed to recouple their randomness. Repeated calls to existing lazy top-1 while retaining the same heap, keys, and cache implement the same algorithm and cannot be treated as a weaker baseline. [A* Sampling](https://papers.neurips.cc/paper_files/paper/2014/hash/937debc749f041eb5700df7211ac795c-Abstract.html) · [Stochastic Beams and Where to Find Them](https://proceedings.mlr.press/v97/kool19a.html) · [Power-SMC](https://arxiv.org/html/2602.10273v3)

[Reference implementation](../../experiments/first_passage/lazy_rank.py) · [Exhaustive check](../../experiments/first_passage/check_lazy_rank.py) · [Same-population paired native-forward measurement](../../experiments/first_passage/measure_native_rank.py)
