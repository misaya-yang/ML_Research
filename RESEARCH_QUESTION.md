# ICML 2027: Research Question and Main Track for First-Hit Search

**Decision date: 2026-11-23.** This document defines the project's main track. The implementation plan and auxiliary experiments follow it. Changing the research question requires stating what new evidence makes the old claim untenable; a blocker alone is not a reason to switch tasks silently.

## The research question

**Given a useful but prohibitively expensive verifier-guided search, can we use only information that is actually available and affordable at runtime to skip parts of the search where a hit might occur, while exactly preserving the first verified answer, logical stopping time, and failure at the deadline? On the same real task and under the same physical budget, can this deliver correct answers the strongest cheaper alternative misses?**

- Original policy: Freeze generation, acceptance and rejection, verification, stopping, and failure rules for a real task. Do not retrofit the policy to fit the theorem and then call it native.
- Exact fidelity: Fix the original policy's logical horizon $T$. Preserve the joint law of the first passing output, its logical index, and failure by the deadline. If the application requires an ordered verification transcript or imposes a call budget, specify those too. Logical steps and physical costs are distinct.
- Practical value: At a matched physical budget, count setup, conditional sampling, model forwards and tokens, verification, caching, and parallelism. Report which additional tasks are solved and at what budget. Do not present virtual steps, archived paths, or isolated query counts as wall-clock gains.

## Three gates, in order

1. **Value on the same workload.** Confirm that a real, frozen policy solves tasks missed by strong direct-sampling or shallow-search baselines on the same workload, with the same verifier and budget. Establish the use case and how much work remains after the strongest baseline's optimizations. Do not build an accelerator for a workload with no marginal value.
2. **What can and cannot be skipped.** On the same workload, specify the success event $E$, prove coverage $E\subseteq D$, quantify the probability of $D$ or bound it, price conditional draws from $D$ and any required complement, and specify a native interface for checking earlier hits. A cheap coupling or random map may replace a backward check only with a proof of equivalence. Do not treat the true success probability, target samples, a full reverse kernel, or unverified conditional generation as free inputs. Give a failure case or lower bound under the same information model, and compare against the strongest baselines with known bounds, early rejection, caching, conditional forward passes, and exact prefetching.
3. **A budget decision changes on the same real task.** Freeze the held-out tasks, original policy, verifier, accounting rules, and physical budget tiers. Independently check the original per-task records. At a matched logical deadline, compare the joint law with the strongest exact implementation. At a matched physical budget, test whether the method delivers more correct answers than the strongest practical alternative. Until the first two gates pass, do not enlarge the sample to hide a bottleneck or retune the threshold to fit the result.

## Current evidence and next step

In the narrow Qwen task, 30 tasks and 60 paired runs matched on outputs and verification transcripts. The strong baseline used 2,134 additional token-logit rows and the jump method 1,816, a 14.90% reduction with reported interval [9.63%, 20.20%]. This falls short of the prespecified 15% gate. It is evidence of reduced computation, not production latency or more solved tasks under the same budget, and the criterion remains unchanged. Next, account for constructing the covering event, conditional draws, and checking earlier hits in the real policy; subtract work the strong baseline already avoids. Do not expand the same narrow workload.

## Keeping the scope fixed

Single-shot decoding, cold-start risk auditing, earlier simulations, and OProver are supporting material or archives, not alternate main tracks or evidence for fixed-policy first-hit performance. When blocked, record the specific gate and missing evidence and address that blocker. Change the main track only if evidence invalidates it or the user explicitly redirects the project.
