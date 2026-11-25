# Two different questions about prefix sampling

This note preserves Claude's November 25 restart and coupling arguments. It separates statements that follow mathematically from the hypothesis tested by E1-P. No assertion here implies a new decoder improves accuracy.

## A capped branch after a checked failure

Fix an instance and the information available after verifying that an original attempt failed. Let $G(b)$ be the probability that a fresh independent attempt finishes correctly by $b$ generated tokens. A branch has uncapped completion length $T$, answer grade $Y$, and cap $C\le B$. It costs $\min(T,C)$ and succeeds in stage one exactly on $A=\{Y=1,T\le C\}$. On failure, spend the remaining budget on one fresh attempt, independent conditional on the instance and the known history.

$$
P(\text{success by }B)-G(B)=
\underbrace{\mathbb E[\mathbf 1_A(1-G(B-T))]}_{\text{incremental rescue}}
-\underbrace{\mathbb E[G(B)-G(B-\min(T,C))]}_{\text{fallback budget sacrificed}} .
$$

Condition on the branch outcome: success is either immediate or has probability $G(B-\min(T,C))$. Expand and subtract $G(B)$. The first term's integrand is zero off $A$. This identity needs the joint law of correctness and completion time; separate mean accuracy and mean cost are not enough. It measures a checked-failure retry, not verifier-free answer selection. Physical cost also needs the prefill, verification and any other overhead absent from generated-token records.

If cost is fixed at $c$ and attempts are independent with success probability $p$, failure at token budget $B$ is $(1-p)^{\lfloor B/c\rfloor}$. The long-budget exponent per token is $-\log(1-p)/c$ (for $p<1$), not $p/c$ except as a small-$p$ approximation.

## Conditional success profiles do not identify an arithmetic lattice

Consider a uniform depth-two binary tree, whose four arithmetic intervals are ordered $00,01,10,11$. Put successes on $\{00,10\}$ in A and $\{00,11\}$ in B. Both have root and first-token conditional success $1/2$, the same distribution of conditional success at each depth, and identical costs. Couple two uniforms as $U,(U+1/2)\bmod1$.

In A the outcomes coincide, so pass@2 is $1/2$. In B exactly one succeeds, so pass@2 is $1$. IID pass@2 is $3/4$ in both. A depth-only profile cannot even tell the sign of the gain. Theorem is about depth profiles, not about full labeled trees, token orderings or knowledge of success intervals.

More generally, if $A\subset[0,1)$ is the arithmetic code set for a correct completion and $s_1,\dots,s_k$ are the lattice offsets, pass@k for the ideal fixed common shift is the measure of $\bigcup_i(A-s_i)\bmod1$. Each single draw has the right marginal. The union depends on arrangement as well as mass. This exact set argument describes the ideal arithmetic map before a finite-precision or independent-refresh implementation changes the joint law.

## An identity for a different coupling: re-stratify while sharing a prefix

Write $M(x)=P(\mathrm{success}\mid x)$. At each shared node $x$, randomly assign one lane to each half of the ordered inverse CDF, sample independently within the two halves, and use independent sampling from the moment the two prefixes differ. Denote the two conditional means of the child value by $m_0(x),m_1(x)$, and the probability both lanes visit $x$ by $c(x)$. Then
$$
P(S_1=S_2=1)-P(S_1=1)^2
= -\sum_x c(x)\frac{(m_0(x)-m_1(x))^2}{4}.
$$
For a shared node, the immediate product of child values has expectation $m_0m_1$, instead of $[(m_0+m_1)/2]^2$. If the lanes diverge, future success products factor given the distinct prefixes; if they stay together, apply the same recursion at the common child. Terminal nodes add no correction. This proves the identity by backward induction on a finite tree. Sorting by child value can maximize the *local* separation of the half means; the identity does not establish a globally optimal order, since $c(x)$ also changes with ordering. For two draws with fixed marginals, pass@2 gain is the negative of the excess co-success above.

The [toy check](../../experiments/prefix_sampling/archive/check_coupling_identity.py) illustrates this conditional identity. It does not prove that a one-dimensional arithmetic lattice loses its dependence when prefixes diverge. Claude's [E1-P protocol](../../experiments/prefix_sampling/e1p/CLAUDE_PROTOCOL.md) tests the latter as a prediction: estimate a truncated tree from IID draws, freeze forecast before evaluating any lattice groups, then measure the gap to the actual coupled sampler.
