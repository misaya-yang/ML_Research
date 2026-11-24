# Speculative first passage: what cheap information can skip, and what it cannot

Drafted by Claude on 2026-11-23. Status: statements and proofs written. The algebra is machine-checked by `experiments/speculative_mh/tests/test_spec.py`: identities on random arrays, and pathwise parity on 40 random tapes through the real HF KV-cache path. The online assertion in `spec_engine.py` checks the cap at run time. Human check pending.

**Native check, MATH-500 rows 0–9, Qwen2.5-Math-1.5B.**
- 50/50 runs are pathwise identical.
- C1 holds on every step: commit ticks are identical.
- C2 holds exactly: 13 rows saved, and 13 rows issued after the cancel tick.
- C4 holds exactly: all 3,213 ticks saved by certificate-time decisions come from units containing a realized rejection, and accept–accept pairs save 0.
- C3's prediction holds: 3 cap events among 11 accept children and 37 shared children.

Paths and the verdict are in `paper/research_notes/speculative_first_passage_20261123.md`.

This note extends [certified thinning](certified_thinning.md) from paid work to **speculative execution**: work done ahead of a decision. Certified thinning says which work any exact simulator must pay for. It says nothing about work spent before a decision on branches that may be discarded, which is the source of PreSTO's speedup and its waste. Notation follows [lazy_power_mh](lazy_power_mh.md).

## 1. Setting

**Frozen policy $\mathsf P$.** This is single-block, fixed-horizon, uniform-cut PowerMH: the [Karan–Du](https://arxiv.org/abs/2510.14901) and [PreSTO eq. 6](https://arxiv.org/abs/2610.13171) kernel with absorbing EOS.

- **Inputs.** A prompt, a horizon $T$, a base next-token law $p$, and $\alpha\ge1$. The proposal law is $q_\alpha(v\mid h)=p(v\mid h)^\alpha/Z(h)$, with $\zeta(h):=\log Z(h)=-(\alpha-1)H_\alpha(p(\cdot\mid h))\le 0$.
- **Tape.** The tape holds independent uniforms $\xi_{k,j}$ for the token at position $j$ of step $k$, cuts $c_k\sim\mathrm{Unif}\{0,\dots,T-1\}$, and coins $U_k\sim\mathrm{Unif}(0,1)$. It is addressed by (step, position), never by schedule.
- **Initial state.** $x^{(0)}$ draws tokens by inverse CDF from $q_\alpha$ using $\xi_{0,\cdot}$ until EOS or $T$.
- **Step $k=1,\dots,K$.**
  - If $c_k\ge|x|$ the step is a no-op.
  - Otherwise the proposal $y$ keeps $x_{<c_k}$ and draws $y_j$ from $q_\alpha(\cdot\mid y_{<j})$ with $\xi_{k,j}$ for $j\ge c_k$.
  - Write $S(w;c)=\sum_{c<j<|w|}\zeta(w_{<j})$; the $j=c$ row is common to both states and cancels. The step accepts iff
$$\log U_k< S(y;c_k)-S(x;c_k)=:\log A_k(y\mid x).$$
  - Cuts do not depend on the state, so the cut choice is symmetric, and this is the exact MH ratio for $\pi\propto p^\alpha$ (lazy_power_mh §1).
- **Verifier and ledgers.** A deterministic verifier $V$ acts on states (here, the public assert on the extracted program).
  - **First-hit ledger:** $L=(\tau,x^{(\tau)})$, where $\tau=\min\{k\le K:V(x^{(k)})=1\}$, or FAIL.
  - **Trajectory ledger:** $\Lambda=(x^{(0)},\dots,x^{(K)})$. $L$ is a function of $\Lambda$.

**Executions and costs.**

- **Reads and cursors.** An execution *reads* rows: the next-token law, and hence $\zeta$, of a prefix. A row can only be read after its parent prefix's row. A *cursor* is a partial proposal.
- **Ticks.** In one *tick* every active cursor advances one position. A *model call* is issued for the rows not already in the prefix trie.
- **Four costs, kept apart:**
  - commit tick (the logical critical path);
  - model calls;
  - live rows (paid work);
  - wall time.
- **Depth-two schedule.** Consider steps $k,k+1$ with cuts $c,d$, from current state $x$.
  - $A$ is step $k$'s proposal.
  - $R$ is step $k{+}1$'s proposal from $x$: the *reject child*.
  - $P$ is step $k{+}1$'s proposal from $y$: the *accept child*. It is launched once $A$ has passed position $d$.
  - If $d\le c$, then $P$ and $R$ have the same prefix and tape addresses and are one cursor: the *shared child*.

## 2. Exactness of any certificate-respecting schedule

**Theorem 1.** Let an execution satisfy three conditions:

- **(E1)** Every token is drawn by inverse CDF from the uniform at its address, given its realized prefix.
- **(E2)** A step is committed with decision $D$ only when $D$ holds for every completion of the revealed rows, given the pre-drawn tape.
- **(E3)** The committed state of step $k{+}1$ is the cursor whose prefix is the committed state of step $k$.

Then $\Lambda$ equals the serial policy's trajectory on every tape. Hence the joint law of $(\tau,x^{(\tau)},\mathbf 1\{\text{FAIL}\})$ and of the whole trajectory are preserved for every verifier. Discarding a cursor is allowed whenever (E2) can still be met for the steps that would have used it.

*Proof.* Induct on $k$, assuming $x^{(k-1)}$ agrees.

- By (E1) and (E3), the realized step-$k$ cursor reads the same tokens as the serial proposal, for as far as it reads, because each token is a deterministic function of its prefix and its address.
- By (E2), its decision is the serial decision.
- The state is either $x^{(k-1)}$ or the proposal. If the proposal, the step was read to EOS, since acceptance needs the complete score (Theorem 2(O1)).

No schedule decision enters any token or decision. $\square$

## 3. The information bottleneck: two obstructions

**Free extension.** Take any finite set of revealed rows and any unrevealed prefix $h$ with $r\ge1$ positions of horizon left. For every $s\in[-r(\alpha-1)\log|\mathcal V|,0]$, some next-token law agrees with all revealed rows and gives *every* continuation of $h$ the suffix score $s$.

- Choose the first position's law with $\zeta=\max(s,-(\alpha-1)\log|\mathcal V|)$ and spread any remainder over later positions.
- Make every other position a point mass ending in EOS.
- No extension of $h$ is revealed, so the revealed rows are untouched.

Call a target score *reachable* if it lies in this range. In code, the range per position is about $-35.8$ nats at $\alpha=4$ with 151,936 tokens, so every bound used below is reachable, except within a position or two of $T$.

**Theorem 2 (obstructions).** Suppose step $k$'s cursor $A$ has read $y$ through position $m<|y|$ and is not rejection-certified: $S_m(y)-S(x;c)\ge\log U_k$.

- **(O1) No acceptance certificate.**
  - There are reachable instances that agree with all revealed rows and make step $k$ accept: deterministic remainder.
  - There are others that make it reject: remaining score below $\log U_k+S(x;c)-S_m(y)$.
  - Hence no exact execution commits an accepted step before reading it to EOS.
  - Hence, when $d>c$ and $R\ne P$, a reject child cannot be irrevocably discarded before $A$ resolves unless its own rejection is certified. An execution that discards it pays to re-read it on the instances where $A$ rejects.
  - In particular, the speculative work on reject children of accepted parents is **irreducible** in this information model.
- **(O2) Coin-blind accept children.** Suppose an execution does not use $U_k$ before $A$ is complete. This is the standard MH loop, and PreSTO, which "generates every proposal in full". Then for every revealed prefix of $P$, some reachable instance agrees with all revealed rows, accepts step $k$ and accepts $P$ at step $k{+}1$. So a coin-blind execution cannot irrevocably discard $P$ before $A$ completes.

*Proof.*

- **(O1)** This is free extension applied to the unrevealed continuation of $y$.
- **(O2)** Fix any $u_k\in(0,1)$ with positive density. Use free extension below $y_{\le m}$ with score $s_A$, low enough that
$$S(y;d)=S(y;c)-S(y;(c,d])\ \le\ S_q(z;d)-\log U_{k+1}-1.$$
  Choose deterministic continuations for $P$, so its final score is $S_q(z;d)$.
  - Step $k{+}1$ then accepts on the accept branch.
  - Step $k$ accepts whenever $u_k<\exp(S(y;c)-S(x;c))$, an event of positive probability whose size the execution cannot see without the coin.
  - On that event the execution's output has to contain $P$'s completion, so $P$ is read. $\square$

Rejection is the only decision that can be certified early, and it arrives only when the cursor's own entropy exceeds its slack. The waste that PreSTO removes by "completed-score cancellation" is exactly the work these two obstructions prevent anyone from removing earlier, except for the part addressed next.

## 4. Escape: the parent's coin is the purchased information

**Lemma 1 (telescoping).** Let $d\le c$, so $z$ is the shared child. Then
$$\log A_{k+1}(z\mid y)=\log A_{k+1}(z\mid x)-\log A_k(y\mid x).$$

*Proof.*

- On $(d,c]$ the rows of $x$ and $y$ coincide, so $S(y;d)-S(x;d)=S(y;c)-S(x;c)$.
- Subtract this from $S(z;d)-S(x;d)$ to get $S(z;d)-S(y;d)$. $\square$

**Lemma 2 (coin-conditioned cap).** Assume step $k$ accepts. Let $S_q(z;d)\ge S(z;d)$ be the partial score of $P$ through any revealed position. This partial score only decreases, because every $\zeta\le0$.

- **If $d\le c$:**
$$\log A_{k+1}(z\mid y)<S_q(z;d)-S(x;d)-\log U_k.$$
- **If $d>c$** (the rows of $y$ on $(c,d]$ are read before $P$ exists):
$$\log A_{k+1}(z\mid y)<S_q(z;d)-\log U_k-S(x;c)+S(y;(c,d]).$$

*Proof.*

- Acceptance means $S(y;c)>\log U_k+S(x;c)$.
- If $d\le c$, then $S(y;d)=S(x;(d,c])+S(y;c)>S(x;d)+\log U_k$.
- If $d>c$, then $S(y;d)=S(y;c)-S(y;(c,d])>\log U_k+S(x;c)-S(y;(c,d])$.
- Finally use $\log A_{k+1}(z\mid y)=S(z;d)-S(y;d)$ and $S(z;d)\le S_q(z;d)$. $\square$

**Rule C.** Discard $P$, or the shared child, at the first tick where the right-hand side of Lemma 2 is at most $\log U_{k+1}$. In code there is a floating-point guard of $10^{-9}$ times the magnitude of the terms.

**Theorem 3.**

- **(a) Exactness.** An execution that adds Rule C to a certificate-respecting schedule satisfies (E1)–(E3).
- **(b) Latency neutrality.** Every commit tick equals that of the same schedule without Rule C, which keeps every needed cursor running: `tree_safe`.
- **(c) Optimality.** If the right-hand side exceeds $\log U_{k+1}$ at some tick, some reachable instance agrees with all revealed rows, accepts step $k$, and accepts $P$. Any execution that discarded $P$ at that tick must re-read it on that instance. So Rule C is the earliest irrevocable discard of an accept-branch child.

*Proof.*

- **(a)** If $A$ is rejected, $P$ is never used. If $A$ is accepted, Lemma 2 gives $\log A_{k+1}<\log U_{k+1}$, so step $k{+}1$ rejects whatever $P$'s unread suffix is. The commit uses the realized denominator $S(y;d)$, which the engine checks to reject at once (online assertion).
- **(b)** In `tree_safe` the same commit happens at the same tick: $A$'s resolution tick, or later if the child is still undecided. In `tree_coin` the realized ordinary bound fires no later, by the same inequality.
- **(c)** Use free extension of $y$ beyond the revealed rows. Make $A$'s final ratio exceed $U_k$ by an arbitrarily small amount, so the inequality in the proof of Lemma 2 is nearly tight, and make $P$'s continuation deterministic. $\square$

**What is purchased, and when.**

- The parent's coin $U_k$, the child's coin and both cuts. These are tape entries drawn before any model work, at zero model cost.
- The rows of $P$ and of $y$ on $(c,d]$, which the schedule has already paid for.
- Nothing about $A$'s unread suffix.

The obstruction (O2) binds coin-blind executions. Rule C escapes it only because, on the one event where $P$ matters, the coin lower-bounds that suffix.

**Depth $r$.** Consider a descendant whose prefix is shared through accepted ancestors with nested cuts. Iterating Lemma 1 divides its ratio by the product of the ancestors' realized ratios, each at least that ancestor's coin on acceptance. So the cap is the root-branch bound divided by the product of the coins on the path. The cap weakens geometrically with depth. This is not implemented; the experiment is depth two.

## 5. Corollary: where any gain must come from (falsifiable)

Compare `tree_coin` with `tree_safe` on the same tape.

- **(C1) Commit ticks.** Commit ticks are identical step by step.
  - Any change in logical latency falsifies Theorem 3(b).
  - A change in wall time can only come from per-call cost (fewer live rows per call), or from calls in which the discarded child was the only uncached row.
- **(C2) Paid work.** The live-row saving equals the rows `tree_safe` issued for discarded children after the discard tick. Cache sharing is the only source of discrepancy. The saving is zero whenever $A$ resolves first.
- **(C3) Margins decide.** Rule C can fire only if the child's rejection margin on the accept branch is at least the parent's acceptance margin, $m_A=S(y;c)-S(x;c)-\log U_k$.
  - *Proof.* At firing, $\log U_{k+1}\ge S_q(z;d)-\log U_k-S(x;c)+S(y;(c,d])$. With $S(z;d)\le S_q(z;d)$ and $S(y;d)=\log U_k+S(x;c)+m_A-S(y;(c,d])$, this gives $\log U_{k+1}-\log A_{k+1}(z\mid y)\ge m_A$. The case $d\le c$ is analogous.
  - So on workloads where accepted proposals clear the threshold by wide margins, Rule C is almost never active, whatever the scheduler.
- **(C4) Fewer ticks only from early rejections.** Moving from completed-score cancellation (`tree_eager`) to certificate-time decisions (`tree_safe`) removes commit ticks only on steps whose realized proposal was rejected before EOS. Theorem 2(O1) says accepted steps are always read to EOS.
- **(C5) The paid-work floor.** By certified thinning Theorem 2, no exact execution reads fewer live rows than serial early rejection on attainable instances. Speculation only adds rows; it trades work for ticks. The best possible paid-work speedup of any exact implementation is the ratio of serial-eager to serial-lazy rows.

## 6. What this establishes and what it does not

**Established.**

- An exact, checkable account of which speculative work can be skipped using information available at decision time:
  - rejection tails (certified thinning);
  - accept-branch children via the parent's coin (Rule C, optimal among irrevocable discards);
  - nothing else (O1, O2).
- The account predicts that gains vanish when acceptance margins are large (C3). It also predicts that the dominant waste, reject children of accepted parents, cannot be removed (O1).

**Not established.**

- Optimality for revocable schedules, which pause and resume: these trade latency against work, and `tree_lazy` is one such heuristic.
- Bounds for batched cost models, where one call reads several rows at the price of one.
- Any statement about final-answer-only contracts. There, certified thinning's bound does not bind, by up to 63.7% on checked units.
- Depth $>2$ beyond the stated product cap.
