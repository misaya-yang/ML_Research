# Exact marked first-passage sampling with forward probability queries

## Original policy and access model

A block of length $L$ starts from $P$ and then follows $R$, with $PR=P$. The acceptance set is $E\subseteq D$ and the first-acceptance age is $J=\inf\{j\ge0:X_j\in E\}$. The goal is to sample $(J,X_J)$ within the block or return failure $\bot$, counting age from zero.

The mass $d=P(D)>0$ is public. We can sample $P\mid D$ exactly, run a deterministic verifier, and sample forward transitions from $R$. The additional interface supplies pointwise forward density ratios $\rho(x,z)$ for named endpoints $z\in D$ only, with a public bound $M$:

$$
R(x,dz)\mathbf 1_D(z)=\rho(x,z)P(dz)\mathbf 1_D(z),\qquad 0\le\rho(x,z)\le M.
$$

An exact coin with probability $\rho(x,z)/M$ can replace the ratio. Conditional carrier sampling, forward transitions, pointwise probability queries, and verification are charged separately. No reverse kernel can be sampled and the density interface is unavailable for endpoints outside $D$. Ordinary $P$ draws are also required for sequential fallback and the original refresh mechanism.

## A forward witness from the most recent earlier acceptance

Let $s_j$ be the probability, conditional on $X_j=z$, that at least one of $X_0,\ldots,X_{j-1}$ passed. For $P$-almost every $z$ in $E$,

$$
s_j(z)=d\sum_{k=1}^{j}
\mathbb E_{Y_0\sim P\mid D,\,R}
\big[\mathbf 1\{Y_0\in E,\ Y_1,\ldots,Y_{k-1}\notin E\}\rho(Y_{k-1},z)\big].
$$

Partition “an earlier acceptance occurred” into the disjoint events in which the last such acceptance is at $j-k$. By stationarity, that step has marginal $P$, so on $E$ we may use $d(P\mid D)$ and verify. Then advance $k-1$ steps and replace the final transition draw by its density at $dz$. Sum the $j$ disjoint terms and divide by $P(dz)$. The sum is empty at $j=0$.

If $dL\le1$ and $dM(L-1)\le1$, sample one block result as follows.

1. Activate with probability $dL$. If activated, independently draw a uniform age $j\in\{0,\ldots,L-1\}$ and $z\sim P\mid D$. Return $\bot$ if not activated or if $z\notin E$.
2. If $j>0$, attempt to find a disqualifying witness with probability $djM$. Draw $k\in\{1,\ldots,j\}$ uniformly and draw an independent $Y_0\sim P\mid D$. If $Y_0\in E$, run forward to $Y_{k-1}$ or the first return to $E$. Only if none of $Y_1,\ldots,Y_{k-1}$ passes, generate a witness with probability $\rho(Y_{k-1},z)/M$.
3. Return $\bot$ if there is a witness and $(j,z)$ otherwise. The original policy's clock counts the full $L$ for both inactive blocks and blocks rejected by a witness.

Given $j,z$, the probability of a witness is exactly $s_j(z)$. Thus the output subprobability at $(j,dz)$ is $P(dz)\mathbf 1_E(z)(1-s_j(z))$, the original policy's joint first-acceptance subprobability. Normalization determines the mass of $\bot$. If $L$ is outside the applicable range, run the block sequentially.

For the original chain $K=\alpha P+(1-\alpha)R$ with independent refresh marks, retain the sampled length of each block. If the original budget leaves only $b$ states, use $L=\min(b,G)$ where $G\sim\operatorname{Geom}(\alpha)$. Start a new refresh only after a failure and with budget remaining. This preserves first-acceptance time, output, and failure mass at every finite logical step budget. Run any nonstationary cold prefix under the original policy until its first refresh or acceptance. Preserving arbitrary variable costs inside the path additionally requires determining their joint law with the sampled endpoint.

## Query accounting and limits

Let $\gamma=P(E)/d$. Given block length $L$, the following bounds exclude long-block fallback. Witness verification counts include verification of the initial seed.

| Query | Expected upper bound |
| --- | ---: |
| Conditional carrier draw | $dL+d^2\gamma M\binom{L}{2}$ |
| Forward $R$ draw | $d^2\gamma M\binom{L}{3}$ |
| Pointwise probability query | $d^2\gamma M\binom{L}{2}$ |
| Verification | $dL+d^2\gamma M[\binom{L}{2}+\binom{L}{3}]$ |

The probability of a witness attempt at each age is $d\gamma\cdot djM$. Each attempt uses at most one pointwise probability query and at most $(j-1)/2$ transitions on average. Summing gives each bound. With sequential forward execution, the probability of acceptance before transition $i$ is at most $id$, so the mean is at least $(L-1)(1-dL)$ transitions. A verification cache can reduce calls on both sides. If the forward interface still must generate each new state, the witness also reduces state generation. Reprice the comparison if a white-box implementation can recover the necessary states more cheaply.

One structure directly realizes the probability interface: choose a class $c'$ through an observable branching rule $A(c'\mid x)$, then run an expensive class-conditional generator $G_{c'}$. If the invariant distribution $P(dc,dy)=\pi(dc)G_c(dy)$ is known and the carrier $D$ is defined by class, the density ratio at a named endpoint $z=(c',y')$ is just $A(c'\mid x)/\pi(c')$: the generator's sequence probability cancels. A conditional carrier draw still requires one complete generation. Charging such draws and witness transitions gives at most $dL+d^2\gamma M[\binom{L}{2}+\binom{L}{3}]$ complete generations per short block in expectation. If the next branch depends on the current generated result, the sequential policy must obtain that result to choose the next step even when its class makes acceptance impossible. If the original branch state can be obtained before full generation or integrated out by a white-box implementation, use that projected interface as the baseline. This is a realizable interface structure; invariance, the carrier, and comparative costs still need verification under the original policy.

For untruncated geometric blocks, let $q=1-\alpha$ and $L_0=\lfloor1/(2dM)\rfloor$ with $L_0\ge1$. Since $M\ge1$, the above procedure applies for $G\le L_0$. Including long-block fallback, the expected number of $R$ queries per reached block is at most

$$
d^2\gamma M\frac{q^2}{\alpha^3}
+q^{L_0}\left(L_0+\frac q\alpha\right).
$$

The first term uses $\mathbb E\binom G3=q^2/\alpha^3$; the second is $\mathbb E[(G-1)\mathbf 1\{G>L_0\}]$. Charge $P\mid D$ and pointwise probability queries separately when estimating real cost.

If the kernel has a public holding term $r(x)\delta_x$, only the remaining part needs the bound $M$. When $j>0$ and $r(z)+djM\le1$, replace the witness stage by two mutually exclusive activation branches. The branch with probability $r(z)$ rejects directly; the branch with probability $djM$ uses the witness above and queries the non-holding part. The removed diagonal mass can occur only at $k=1,Y_0=z$ in the forward decomposition. In a continuous state space, unknown holding mass cannot be included in the density ratio. The short-block algorithm additionally needs a public uniform holding-rate bound guaranteeing the condition, or another exact route for each age.

## Direct forward method when the successful prefix must be preserved

If $d[1+M(L-1)]\le1$, activate with that probability, choose age $j$ with weight $1$ at zero and $M$ at each other age, and draw $z\sim P\mid D$. After verifying $z\in E$, output immediately when $j=0$. For $j\ge1$, draw another forward prefix $X_0,\ldots,X_{j-1}$ starting from $P$ and return $\bot$ on any earlier acceptance. Otherwise return $(j,X_0,\ldots,X_{j-1},z)$ with probability $\rho(X_{j-1},z)/M$ and $\bot$ on rejection.

For any successful prefix, the activation and choice of $j$ contribute $dM$, the conditional endpoint contributes $P(dz)/d$, and the final acceptance probability contributes $\rho/M$. Their product is the original final transition $R(X_{j-1},dz)$. Any deterministic cumulative cost of the successful prefix can be returned. Failure blocks return only $\bot$; if their variable cost still changes future refresh budgets, its correct conditional distribution given failure is also needed.

This method uses at most $d\gamma M\binom{L-1}{2}$ forward $R$ queries per block in expectation, and ordinary $P$ draws and pointwise probability queries are each bounded by $d\gamma M(L-1)$. For pointwise probability queries, this bound is $2/(dL)$ times the bound for outputting only age and endpoint. For forward $R$ queries with $L\ge3$, the ratio between the bounds is $3/(dL)$.

## Lower bounds under the same access model

**Two-step blocks.** For $0<d\le1/4$ and $2\le M\le 1/(2d)$, two instances share all free information, yet any algorithm exact on both must purchase a forward or pointwise query with probability at least $Md^2/16$. Take $D=E=\{a,b\}$, $P(a)=P(b)=d/2$, and $P(c)=1-d$. In instance zero, $R_0(a,a)=R_0(a,b)=Md/4$, $R_0(a,c)=1-Md/2$, and $R_0(b,\cdot)=P$. Set $R_0(c,\cdot)=[P-P(a)R_0(a,\cdot)-P(b)P]/P(c)$ to preserve stationarity. In instance one, add and subtract $\delta=Md/8$ in the $a,b$ columns of row $a$, respectively, and subtract and add $P(a)\delta/P(c)$ in the corresponding columns of row $c$. The stated ranges ensure nonnegative rows summing to one, density ratios into $D$ bounded by $M$, and stationarity by the column sums.

Both instances have the same age-zero mass. At age one the first-acceptance subprobability at $a$ decreases by $P(a)\delta$, the mass at $b$ increases by the same amount, and failure mass is unchanged. Thus the total variation distance between correct outputs is $Md^2/16$. Under shared free randomness, observations and outputs are identical up to the first paid query, so this query occurs with probability at least that distance. The witness method requires only $O(Md^2)$ expected paid queries on these instances, matching the two-step order. Optimal costs for $L>2$ remain open.

**Successful-prefix costs can be more expensive.** Let $L=2$, $D=E=\{e\}$, $P(e)=d\le1/3$, and $P(a)=P(b)=(1-d)/2$. In instance zero, every row equals $P$. In instance one, add $d$ to $R(a,e)$ and subtract $d$ from $R(b,e)$, with opposite compensating changes in column $a$. Both instances have $M=2$ and identical invariant distributions and $(J,X_J)$ laws. If $a,b$ have distinct known constant costs, the total variation distance for successful outputs carrying the age-one prefix cost is $P(a)d=\Theta(d)$. The same coupling argument requires $\Omega(d)$ paid queries, matching the direct method. Preserving only age and endpoint costs just $O(d^2)$ with this access. The distinction also applies to finite-budget outputs where a cost threshold separates the two types of successful prefixes.

**Only an unknown success coin.** Fix budget $B$. If the only information about the unknown parameter comes from independent $p$-coins and $p\downarrow0$ is allowed, exactly preserving the deadline success rate $1-(1-p)^B$ requires

$$
\mathbb E_p N\ge \frac{1-(1-p)^B}{p}\qquad(p>0),
$$

Here $N$ is the number of expensive coin calls. If the algorithm can output success on any finite all-tails transcript, its success probability stays positive as $p\downarrow0$, a contradiction. Thus it must observe at least one head before each success. The expected head count is $p\mathbb E_p N$, at least the required success probability. Sequential simulation until the first head or the deadline attains equality, even when a positive upper bound on $p$ is supplied. If, after early rejection and caching, real generation provides only this coin, a Bernoulli factory cannot skip a string of misses to buy fewer flips. The forward witness instead uses bounded pointwise probabilities and conditional carrier draws.

## Relation to prior results

- Classical stationary first-passage and return identities recover the unmarked age marginal. Here the disjoint decomposition by the most recent acceptance and pointwise endpoint densities allow sampling finite-block marked outputs. The pointwise query and its bound are additional assumptions. [Aldous and Fill, Lemma 2.23](https://stat-www.berkeley.edu/pub/users/aldous/RWG/Book_Ralph/Ch2.S5.html) · [Project reverse-kernel sampler](first_passage_core.md#exact-sampling)
- If the exit probability and exit state can already be obtained cheaply, a standard jump chain recovers the clock using geometric holding times. The unknown-coin lower bound applies to access by ordinary sampling alone. [Rosenthal et al.](https://arxiv.org/abs/1910.13316)
