# First passage: preserving the original policy’s clock, output, and verification costs

## Exact sampling

Let $K(x,\cdot)=\alpha P(\cdot)+(1-\alpha)R(x,\cdot)$ and start from $P$. Independent refresh marks occur with probability $\alpha$, and $PR=P$. Let $R^*$ be a reverse kernel that can be sampled exactly (it is $R$ when $R$ is reversible). Assume we can sample $P\mid D$ and know the mass $d=P(D)$ of a carrier covering the acceptance event $E\subseteq D$. We need not know $p=P(E)>0$ or be able to sample $P\mid E$. Write $\gamma=p/d$, let $G\sim\mathrm{Geom}(\alpha)$ (positive integer-valued and independent), and define
$$
U=\Pr_{P\mid E,R^*}(G\le\tau_E^+),\qquad
\tau_E^+=\inf\{j\ge1:X_j\in E\}.
$$

Between two refreshes is a stationary residual block starting from $P$, with length $L\sim\mathrm{Geom}(\alpha)$. Given $L$, the subprobability that the first acceptance occurs at step $j\in\{0,\ldots,L-1\}$ and the output falls in $dz$ is
$$
P(dz)\,\mathbf1_E(z)\Pr_z^{R^*}(\tau_E^+>j).
$$
For a block with $dL\le1/2$, activate with probability $dL$, then independently choose a uniform age $j$ and draw $z$ from $P\mid D$. Verify $z\in E$ and use the reverse kernel to check that the preceding $j$ steps did not accept earlier. For all other blocks, use sequential forward checks **without additional thinning**. Count the full length of a block that does not accept. The block subprobability and renewal relation give exactly the same first-acceptance time and output $(\tau,Z)$ as the original policy. A known nonstationary cold prefix must be simulated sequentially from the original initial state until the first acceptance or first refresh; it cannot be treated as stationary. For $d\le\alpha/4$, the expected numbers of short-block conditional carrier draws and initial candidate checks are each at most $1/(\gamma U)$, and the expected number of short-block residual draws is at most $(1-\alpha)/\alpha$. These counts cover short blocks only: long-block fallback, reverse verification, and the actual cost of conditional carrier sampling are separate.

When verification is expensive, use the active set $A=D\setminus\{\text{identities cached as negative}\}$. First jump exactly to the actual first entrance into $A$ and invoke the verifier only for an unseen identity. Cache a negative result and resume with the complete state and clock at arrival. Future refreshes are redrawn by the strong Markov property; do not reset immediately to $P$. With the same deterministic verifier, identity cache, and definition of visit order, this preserves the distribution of the paid verification ledger (identities, order, times, and payloads), not merely the final success rate.

**Limit without the reverse kernel.** A stationary forward witness can recover the marginal first-passage age law but need not recover its joint law with prefix cost. For example, use the uniformly stationary three-cycle $e\to a\to b\to e$, acceptance set $\{e\}$, and state costs $c(a)=0,c(b)=1$. At age 1, the first acceptance must go from $b$ to $e$, so the true subprobability of prefix cost 0 is 0. Instead moving forward from $e$ to $a$ and substituting its cost gives $1/3$.

## Lower bound in the sample-access model

In the reversible family with hidden fast and sticky states, a model query reveals whether a state is fast or slow and an independent verification query reveals whether it passes. Here $P,\alpha,d,\gamma$ are public, and draws from ordinary $P$ and the carrier are free. The query prices are $w_R,w_E$. For one exact output in this family, the worst-case optimal expected price is, up to constants,
$$
\Theta\!\left(\frac{w_R}{\alpha}+\frac{w_E}{\gamma}
      +\frac{\min\{w_R,w_E\}}{\alpha\gamma}\right).
$$
The hidden intersection makes it hard for a short query transcript to obtain both types of information. Two sampling routes attain the family’s upper bound by spending more queries on the cheaper side.

The same obstruction yields a finite-time **95% deadline certification** problem with constant correctness probability. The two instance groups have the same numbers of fast and accepting states, but different intersections. For a residual chain at the safe anchor, let the additional one-step probability of entering the acceptance set be $v/(1-\alpha)$ and let $b=\alpha p$. If $S_t=\Pr(\tau>t)$, then
$$
S_{t+1}=(1-b-v)S_t+v\Pr(\tau>t,X_t\in D),\qquad
0\le v\Pr(\tau>t,X_t\in D)\le vd.
$$
An empty intersection and a balanced positive intersection yield acceptance rates of approximately 90% and 99% at the same deadline (with fixed margins in the small-parameter regime), yet the total variation distance between transcripts of $n$ paid sample queries is $O(\alpha\gamma n)$. Near the balanced intersection in this family, measuring accuracy by a relative intersection difference $\theta$ adds a factor of $\theta^{-2}\log(1/\delta)$ to the query lower bound (the carrier mass must be small enough and its cardinality large enough). These statements are for the specified query interface and instance family. They do not cover solvers that can read the transition matrix directly or arbitrary settings where two resource constraints bind simultaneously.

## Joint certification under label uncertainty

Fix proposals and all nonnegative accumulated costs. For each baseline trajectory that accepts, continue it once for its first accepted identity $x$: relabel that identity as negative in the cache, retain the costs already paid, and resume from the actual state and clock. Deleting any other single label does not affect that already accepted trajectory, and a baseline failure cannot become an acceptance by deleting a positive label. If $s$ is the baseline joint acceptance rate and $\ell_x$ is the probability that the baseline accepts first at $x$ but its continuation after deletion fails, then the worst-case acceptance rate under at most one erroneous positive label is exactly $s-\max_x\ell_x$. Construct simultaneous confidence bounds for $s$ and all $\ell_x$ before bounding the maximum loss; unadjusted per-label confidence bounds do not give a joint guarantee.
