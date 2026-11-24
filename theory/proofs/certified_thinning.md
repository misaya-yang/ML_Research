# Exact first passage by certified thinning: one read rule, one lower bound

Drafted by Claude on 2026-11-23, after the user's review of [lazy_power_mh](lazy_power_mh.md). Status: statement and proofs written. The PowerMH case is machine-checked ([check_lazy_mh.py](../../experiments/first_passage/check_lazy_mh.py)). The ranking, last-mark and one-sided cases are checked by their own scripts. Human check pending. This note keeps the main line of [RESEARCH_QUESTION.md](../../RESEARCH_QUESTION.md): it is a theorem about preserving a frozen policy's first verified output, logical index and deadline failure, not a change of topic.

## 1. Setting

A frozen policy runs logical steps $t=1,\dots,B$. At step $t$, with history $H_{t-1}$, it runs one **trial**:

- a **decision coin** $\xi_t\sim\nu_t(\cdot\mid H_{t-1})$: policy randomness, independent of the hidden content given $H_{t-1}$, and so drawable before any work;
- hidden **content** $Y_t=(Y_{t,1},Y_{t,2},\dots)$, revealed one paid unit at a time in the policy's order, with price $a(u)\ge0$ for unit $u$;
- **delivery** $D_t=\mathbf 1\{\xi_t\in A_t(H_{t-1},Y_t)\}$.

The **ledger** is the list of delivering steps with their full content, plus deadline failure. First-hit policies stop at the first delivery; chains such as MH continue, and every accepted move is a delivery. This is the contract of the research question: first verified output, its logical index, and failure by the deadline. For MH it is the whole trajectory.

Two update regimes:

- **(M0) miss-oblivious.** After a miss, $H_t$ depends on $(H_{t-1},\xi_t)$ but not on the unread content. Examples: an MH rejection leaves the state unchanged; an IMH proposal outside the carrier; a slot whose key falls behind the stopping point.
- **(M1) miss-dependent.** A miss's content changes the future, as in adaptive repair.

**Certified cap.** In (M0), $\bar A_{t,k}(H_{t-1},Y_{t,1:k})$ contains $A_t(H_{t-1},y)$ for every admissible completion $y$ of the first $k$ units, and is non-increasing in $k$. In (M1), $\bar G_t$ is a set of coin sequences $(\xi_t,\dots,\xi_B)$ containing every sequence under which some delivery at a step $\ge t$ is possible, for all continuations. Write $\Gamma_u$ for the cap event in force just before unit $u$.

**Read rule $R^*$.** Pay for unit $u$ iff $u$ is reached and the pre-drawn coins lie in $\Gamma_u$. Let $E_u$ denote this event.

## 2. Exactness and the cost identity

**Theorem 1.** Suppose deliveries are contained in cap events, and an implementation pays for exactly the units selected by $R^*$, computes everything else as the policy does, and records a miss for every trial it abandons. In (M1) it must stop only when $\bar G_t$ fails. Then, under shared coins and position-indexed content randomness, its ledger equals the policy's ledger pathwise. Its expected cost is
$$
\sum_u a(u)\Pr(E_u),\qquad\text{saving}=\sum_u a(u)\Pr(u\text{ reached},\ \text{coins}\notin\Gamma_u).
$$

*Proof.* Induct over steps.

- *(M0).* If the coin stays in every cap, all units are read and the decision is the policy's. If it leaves $\bar A_{t,k}$, then it lies outside $A_t(H_{t-1},Y_t)$, so the policy misses too, and the next history does not depend on the unread units.
- *(M1).* Once $\bar G_t$ fails, no later delivery is possible, so the ledger is deadline failure whatever the remaining steps do.
- *Races.* For races, where the threshold $T_{-i}$ of stream $i$ comes from the other streams, the optimistic-key heap realizes $R^*$ ([stopped_weighted_ranking](stopped_weighted_ranking.md)).

The cost identity is linearity of expectation over units. $\square$

The saving term is exactly what a **replay of complete eager logs** measures. It also measures what remains on top of a baseline that already applies some caps: the remainder is $\sum_u a(u)[\Pr(\text{baseline pays }u)-\Pr(E_u)]$.

## 3. Necessity

**Access model.**

- (a) Units are revealed one at a time, at their prices.
- (b) A simulator is exact if its ledger has the policy's law for every instance in the admissible class $\mathcal I$. It need not be pathwise and may recouple all randomness.
- (c) Cost is additive over paid units.

**Attainability.** The cap is attainable at unit $u$ of instance $I$ if some $I'\in\mathcal I$ satisfies two conditions:

- it answers every query identically to $I$, except at $u$ and at units that can only be queried after $u$;
- on $E_u$ it produces a ledger value in a set $N$ with $\Pr_I(\text{ledger}\in N)=0$.

**Theorem 2.** If the cap is attainable at $u$, every exact simulator pays for $u$ under $I$ with probability at least $\Pr_I(E_u)$. $R^*$ attains this, so $\mathrm{TV}(\mathcal L_I,\mathcal L_{I'})=\Pr_I(E_u)$.

- If attainability holds at every unit of $I$, then $R^*$ minimizes every nonnegative additive cost **at that instance**.
- If it holds on a subfamily, then $R^*$ is minimax on that subfamily.

*Proof.* Run any exact simulator on $I$ and $I'$ with the same internal randomness. The transcripts agree until it pays for $u$, so
$$
\mathrm{TV}\le\Pr_I(\text{pays for }u).
$$
Conversely,
$$
\mathrm{TV}\ge\Pr_{I'}(N)-\Pr_I(N)\ge\Pr_I(E_u),
$$
because $E_u$ is determined before $u$ and has the same probability in both worlds. $R^*$ is itself an exact simulator, so the first inequality applied to it gives equality. $\square$

**Scope, stated precisely.**

- The bound binds every simulator that is exact for the **ledger law**, including ones that recouple randomness. In particular, for PowerMH it binds every simulator exact for the trajectory law, not only seed-by-seed replay.
- It does not bind simulators that only need the **final-answer law**. There the coupling bound is strictly weaker: 270/270 checked units, by up to 63.7%.
- It does not bind cost models where one call reveals several units at the price of one, such as batched scoring or speculative verification measured in calls or latency.
- It does not bind white-box access to the network.
- It is a statement about paid units, not wall-clock time.

## 4. Where caps come from

| Source | Cap | Why it is free and exact |
|---|---|---|
| **Score monotonicity of sharpening samplers** | PowerMH: $\bar A_{t,k}=\{U\le\exp(S_k-\Phi)\}$. Weighted order and SMC terminal selection: replayed weight ceiling $u_{i,k}$ | For $\alpha\ge1$, $\log Z_\alpha(h)=\log\sum_v p(v\mid h)^\alpha=-(\alpha-1)H_\alpha(p(\cdot\mid h))\le0$, with $H_\alpha$ the Rényi entropy. Prefix log-probabilities are $\le0$. Each new token can only add uncertainty, so the bound only tightens. No learned predictor is involved. It fails for flattening ($\alpha<1$), where $Z>1$ |
| **Syntax and action carriers** | IMH with a necessary first token, mass $q(D)$; repair with a necessary action set $D$, cap $c=\pi(D)$; stationary blocks, union bound $dL$ | The carrier mass is a single softmax or mask probability. Conditional draws from $D$, plus either complement draws or one amplified terminal step, couple the coin to delivery |

For PowerMH the test reads: reject as soon as the proposal's accumulated Rényi-$\alpha$ entropy exceeds the current suffix's total by the random margin $|\log U|/(\alpha-1)$. The saving is therefore governed by where uncertainty accumulates along the proposal. Early high-entropy forks decide rejections early.

## 5. The repository's results as instances

| Result | Regime | Cap | Realization of $R^*$ | Lower bound | Measured, identical ledger |
|---|---|---|---|---|---|
| PowerMH early rejection ([lazy_power_mh](lazy_power_mh.md)) | M0 | Rényi score | Direct check after each token | Instance-wise (fresh-token replacement) | Pending (replay R0) |
| Stopped weighted ranking with dedup and a verification budget | M0 race | Weight ceiling | Optimistic-key heap | Instance-wise (new passing text at the ceiling) | 27 positions on 4 dev tasks |
| Power-SMC terminal selection | M0 race | Replayed ramp | Gumbel race with bounds | Instance-wise when the ramp bound is attainable | Part of 15.43% on MBPP-56 (35/6/15 in all arms) |
| Power-SMC across-checkpoint certificates | **Outside M0**: particles interact through resampling normalization | Weight boxes for ESS and parent maps | Interval certificates | **No matching bound.** The 27.62% witness shows a non-delivering particle can change the winner's survival | Rest of 15.43% |
| Last-mark suffix trimming ([marked_forward_horizon](marked_forward_horizon.md)) | M1 | Carrier marks, $\bar G_t=\{\exists j\ge t: M_j=1\}$ | Marks coupled into native actions (needs complement draws) | Address-wise on the all-zero private-bit instance: $1-\prod_{j\ge i}(1-c_j)$ | Coloring screen: cap 0.961 calls |
| One-sided terminal step ([one_sided_first_hit](one_sided_first_hit.md)) | M1 | Same | Last mark, then one amplified terminal step (no complement draws) | Same, minimax | Same |
| IMH with a syntax carrier (Qwen year task) | M0 | $q(D)=0.010$–$0.032$ | Geometric jumps over carrier misses | Not yet stated; carrier attainability to check | 14.90% over the strong exact baseline, CI [9.63, 20.20]; 60/60 identical; mapping to `native_clock.py` to confirm, as that source is not in this repo |
| Forward witness and reverse-kernel blocks ([forward_first_hit_candidate](forward_first_hit_candidate.md), [first_passage_core](first_passage_core.md)) | M0 blocks | Union bound $dL$ plus a witness | Activation coin | Order-matching for two-step blocks only (union bound not attained) | — |

The joint certification under label errors is a statistical result about a different estimand. It stays outside this theorem.

## 6. What is new and what is not

- **Prior art.**
  - Thinning by a dominating rate ([Lewis and Shedler](https://doi.org/10.1002/nav.3800260304)).
  - Early rejection in MCMC ([Solonen et al. 2012](https://doi.org/10.1214/12-BA724)).
  - Lazy Gumbel and A* sampling.
  - Threshold algorithms with instance optimality (Fagin, Lotem and Naor), which is deterministic, not distributional.

  None of these primitives is claimed.
- **New.**
  1. A single read rule and a single coupling lower bound covering carrier caps and score caps, both miss regimes, and races. The bound is instance-wise exactly when caps are attainable, and minimax or order-matching otherwise.
  2. The identity $\log Z_\alpha=-(\alpha-1)H_\alpha$. It explains why sharpening LLM samplers carry free exact certificates and flattening ones do not.
  3. Explicit boundaries: the output-only contract, normalization-coupled streams, and batched or white-box cost models.
  4. The cost identity, which turns "how much is left on top of a strong baseline" into a replay measurement.

## 7. Open items

- Prove carrier attainability for the IMH syntax-carrier instance, or state the order of the gap.
- Give an output-only simulator strictly cheaper than $R^*$ on some instance, or prove output optimality for one MH step.
- Determine whether the across-checkpoint SMC certificates admit any matching bound within a restricted family.
