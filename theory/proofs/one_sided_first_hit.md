# First passage under a finite budget: a native forward prefix and one amplified final step

## A chargeable one-sided interface

For a fixed problem, first handle any initial hit or stop under the original policy. For the remaining continuation states, use the original hard step limit \(B\). The original policy may depend arbitrarily on generated text, verification feedback, caches, actions, and cumulative resources. A native step can produce the first verified output or change the future search even when it fails at that step.

For a continuation history \(h\) and original step \(i\), let \(H_{i,h}\) be the subprobability kernel for a valid acceptance record from **one native step**, including the required actual candidate, native resource increments, and check outcome. It is not the probability of success between now and the deadline. Assume a provable ex ante bound \(0\le c_i\le1\) and an available final-step operation: if \(c_i>0\), it returns records with subkernel \(H_{i,h}/c_i\) and returns no result with the remaining probability; if \(c_i=0\), the step cannot pass. Track costs for a native step, final-step preparation, and actual final-step generation separately. The bound may depend on the known problem or initial state, but must be exogenous to subsequent unseen native randomness.

One realizable interface: the native controller first chooses action \(A\) under \(\pi_h\), then generates and verifies. Assume a known necessary action set \(D_i(h)\), its mass \(s_i(h)\le c_i\), and a way to draw **control actions** from \(\pi_h\mid D_i(h)\). Activate the final step with probability \(s_i(h)/c_i\); then draw that control action, generate and verify natively, and apply the original budget rule. This can actually produce incorrect candidates and requires no sampling from correct answers. No other step needs access to \(D_i\). There is no need to sample the complement or know the actual success rate for a step or the whole run. A bound alone is insufficient: if the only available interface is an unknown \(p\)-coin obtained by full generation, outputting a real success with probability \(p/c\) requires at least \(1/c\) native draws in expectation, because the output success probability cannot exceed the expected number of observed successes.

## Algorithm and proof

1. Outside the controller, draw an independent time \(J\) with the law of the last 1 in \(B\) independent \(\mathrm{Ber}(c_i)\) marks, or 0 if there is no 1. Scan backward to the first 1; for constant \(c\), use a truncated geometric draw. Nothing is generated at this point.
2. If \(J=0\), return deadline failure. Otherwise run steps \(1,\dots,J-1\) **exactly under the original policy**. Return immediately upon native first acceptance; return failure if the native policy stops without acceptance or runs out of resources.
3. If step \(J\) is reached, perform the final-step operation above once. Return a success record or failure using the original budget rule, then stop.

Let \(F_i(dh,dz)\) be the submeasure of the original successful prefix and output when the first delivery occurs at step \(i\). Write \(R_i=\prod_{j>i}(1-c_j)\). When \(J>i\), the native prefix contains step \(i\), contributing \((1-R_i)F_i\). When \(J=i\), the first \(i-1\) steps remain native and the final step contributes \(c_iR_i\,F_i/c_i\). Thus every time with \(c_i>0\) and every delivery record satisfy

\[
  (1-R_i)F_i+c_iR_i(F_i/c_i)=F_i.
\]

If \(c_i=0\), then \(F_i=0\). All success submeasures match, and their complement gives the same deadline failure probability. The success record retains the original step and all resources used at success, so it also supports the original nested deadlines. A dynamic resource threshold can be included in the continuation rule and \(H_{i,h}\). The algorithm may deliver along a different random trajectory from the original policy; it does not claim seed-by-seed equality. If the caller also consumes the state after failure, the complete negative verification ledger, or the actual failure costs, those must be recovered separately; the failed final step cannot be passed directly to such a caller. Use the algorithm separately in each independent restart if the restart uses a fixed native allowance and receives only a failure flag.

## Cost identity and optimality

In the control-action implementation above, let \(W_i\ge0\) denote expensive generation or verification work after the native action, including dependence on the action and result. Let \(L_i\) be the event that the original policy reaches step \(i\). Charge conditional-control preparation and additional scoring separately. Then the **expected expensive work** saved is exactly

\[
 \mathbb E C_{\rm ordinary}-\mathbb E C_{\rm new,emission}
 =\sum_{i=1}^B R_i\,\mathbb E_\pi[{\bf1}_{L_i}{\bf1}_{A_i\notin D_i(h)}W_i].
\]

The native cost on the left counts only the same \(W_i\) items. Proof: \(J>i\) contributes \((1-R_i)\mathbb E[{\bf1}_{L_i}W_i]\). For \(J=i\), the final-step activation probability times the conditional action law contributes \(R_i\mathbb E[{\bf1}_{L_i}{\bf1}_{A_i\in D_i}W_i]\). Subtract to get the equation above. Estimate the right-hand side from unfiltered native trajectories; actual gating costs must be subtracted. The formula still accounts for omitted suffixes if the activation probability equals one in every continuation state.

**Matching query lower bound.** Each address \(i\) has a public price \(a_i\ge0\). A native or final-step call requires at most one private emission at that address (treat the one-sided control cost as given or charge it separately). The algorithm's worst-case expected cost is at most

\[
 \sum_{i=1}^B a_i\,\Pr(J\ge i)
  =\sum_{i=1}^B a_i\left[1-\prod_{j=i}^B(1-c_j)\right].
\]

Construct a subfamily with the same interface. A native action is a mark of probability \(c_i\); the expensive operation at step \(i\) reveals a private bit \(\theta_i\). Seeing any positive bit turns on a switch permanently. Each step reads the bit first, and delivers if and only if the switch is then on and the action is marked. The final step forces the positive action but still reads only this address. The all-zero world always fails. The world with only bit \(i\) positive delivers with probability \(1-\prod_{j=i}^B(1-c_j)\). Only a query to \(i\) distinguishes these worlds. Couple any distribution-exact algorithm using the same internal randomness; the probability of reading \(i\) in the all-zero world is at least this difference. Multiplying by the price at each address and summing yields the upper bound exactly. This gives the minimax expected query price in this access model. It does not assert that each actual policy or piece of hardware is equivalent to the worst-case instance. Reprice shared forward passes, batching wall time, and other operations already supplying final-step information under the actual interface.

If final and native steps cost \(b_i,a_i\), respectively, the general upper bound is \(\sum_i[(1-R_i)a_i+c_iR_i b_i]\), plus preparation. A lower call count does not remove the cost of conditional generation.

## Native example and numerical checks

In graph-coloring repair with independent per-vertex update probability \(p\), resolving the problem in the current step requires updating at least one endpoint of any currently conflicting edge \(uv\). Here \(D\) is the event that the mask touches \(u\) or \(v\), and \(s=c=2p-p^2\). Native prefix masks are unchanged; only step \(J\) draws a mask conditional on touching at least one endpoint. Other vertices keep their original independent masks. No step has to guess the correct coloring. In the published long-budget setting with \(p=0.3\), this one-sided bound saves less than one step in expectation at the end. The graph also exposes stronger cheap information; see the [matched-task evidence](../../results/first_passage/forward_search.md).

Run [last_mark.py](../../experiments/first_passage/last_mark.py) to compare full rational distributions, including input-dependent actions, random emissions, early stops, zero and one bounds, and different native resource deadlines. An [explicit coupling with marks throughout](marked_forward_horizon.md) works when the complement can also be sampled conditionally. [Bounded-rate thinning](https://doi.org/10.1002/nav.3800260304) and [Markov kernel splitting](https://doi.org/10.1007/BF00534764) are classical principles. The proof here concerns marked first passage at a finite original budget with only one final-step correction and a matching chargeable boundary under the same one-sided interface.
