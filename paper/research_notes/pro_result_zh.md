# Pro Research Results: Archived Snapshot

Source: Pro results forwarded by the user. Private conversation and Library URLs have been omitted from this public translation. See [source inventory](../sources/SOURCE_LINKS.md) for the archive reference.

## 1. Main theorem

For a fixed independent Metropolis–Hastings chain starting from \(Q\), under finite weight moments and specified window conditions, the information needed to certify mean squared error of the ordinary time average is governed by \(p^{1-r}q^r\). Here \(p\) is the number of independent target draws and \(q\) is the proposal scoring budget. It gives both an executable sufficient condition and a necessary condition for any auditor, up to constants. **CPU experiments support the worst-case frontier but refute the practical prediction that this allocation increases certification at small quotas.**

Let \(W=dP/dQ>0\). The auditor observes only scores \(S=ZW\) with unknown common multiplier and an observable \(0\le h\le1\). Start \(X_0\sim Q\); discard \(b\) steps and average \(t\) states to obtain \(A\). The quantity to certify is
\[
G=\mathbb E[(A-P h)^2]-\tfrac12\operatorname{Var}_P(h).
\]
Require false certification probability at most \(\delta\) for each \(G\ge0\), and certification probability at least \(1-\delta\) for each \(G\le-\varepsilon\). This excludes an auditor that always abstains.

**Exact scope.** Write \(L=b+t,\ K=q/p\). Suppose \(t\ge4,\ 0<r<1,\ \delta\le0.1\), and
\[
\mathbb E_Q\min(1,W)\ge\tfrac18,\qquad
\mathbb E_Q W^{1+r}\le M,\qquad
2(32L)^r\le M\le K^r/2,\qquad
0<\varepsilon\le \frac{M}{2000K^r}.
\]
In this regime there are absolute constants \(0<c<C\). An auditor satisfies both guarantees when \(p^{1-r}q^r\ge CM\varepsilon^{-2}\log(1/\delta)\), and every auditor meeting them must satisfy
\[
\boxed{p^{1-r}q^r\ge cM\varepsilon^{-2}\log(1/\delta).}
\]
The upper-bound auditor does not take \(M,r,Z\) as inputs. The lower bound allows \(Z,h\) to be public and permits adaptive pointwise score queries. When \(r=1/2\), quartering the target quota in this regime requires four times as many proposal scores to maintain the same worst-case order. The proof also supplies instances where allocations with **the same total query budget** differ categorically: one suffices, the other cannot. [Complete theorem and proof](../../theory/proofs/pro_moment_frontier_original.md)

Two steps establish the result.

- **Information recoverable from \(Q\).** Use independent proposal draws to choose a cutoff \(H\). Put \(v=\min(1,S/H)\), \(g=\min(1,H/S)\), and \(\theta=Q(vh)/Qv\). The identity \(Wg=(H/Z)v\) implies
  \[
  Ph=\theta+P[(1-g)(h-\theta)].
  \]
  Estimate the first term using proposal scores only and use the target draws for the leftover residual. Choosing \(H/Z\) on the order of \(q/p\) gives both stochastic errors order \(\sqrt{M\log(1/\delta)/(p^{1-r}q^r)}\). Apply the same draws to \(h^2\), then use independent ordinary cold trajectories for \(\mathbb EA,\mathbb EA^2\) to bound \(G\) directly. One target draw provides both observable moments; a joint confidence bound is used rather than counting two independent draws.
- **Why arbitrarily many trajectories do not substitute for \(P\).** The lower bound uses three groups of weights: low, \(32L\), and \(K\). The highest group has target mass \(m\asymp M/K^r\) and proposal mass \(m/K\). Altering its label proportion while compensating in the low group leaves public \(Q(h=1)\) unchanged. First calibrate the middle group to the **true cold-chain risk threshold**. The perturbation then produces worlds with \(G<-\varepsilon\) and \(G>\varepsilon\). Each target query has at most \(O(\varepsilon^2/m)\) KL information and each proposal score loses a factor \(K\). Summing along adaptive transcripts gives the lower bound. With true \(Ph\) disclosed, ordinary proposal trajectories can estimate \(G\) in these binary instances. The difficulty is **target centering**, not computing cold-chain transitions.

Scope of novelty: Smith–Tierney's Theorem 1 already gives the exact IMH multistep kernel. Hesterberg §6 and Veach Theorem 9.2 imply order-matched variance bounds for normalized static means using mixture importance sampling. Meng–Wong Theorem 1 analyzes bridge sampling for normalizing-constant ratios. Zhang et al. (ICLR 2025), Theorem 1, give minimax moment-estimation bounds with known source and target distributions. The contribution here is an **executable finite-sample certificate at unknown scale and a matching lower bound at the specified cold-chain risk threshold under the same pair of budgets**. No Wiener inversion is needed. [Eigenvalues And Eigenvectors](https://www.scribd.com/document/877458738/document?utm_source=chatgpt.com)

## 2. CPU experiments

For 32 old EWT failures, hold the window, random draws, and quotas fixed. In order of neither oracle, target mean only, normalizer only, and both oracles, virtual jumping certifies **0, 0, 2, 7** and ordinary replay **24, 32, 24, 32**. Even with both oracles, virtual jumping abstains on 25. The old long-window reference remains the original estimate from 8,192 paths; new finite-state and sequence panels have separately computed reference values. The raw old-failure ablation is `dual_round/panel/old-ablation.csv` in the source archive.

Each cell is **safe certifications / false certifications / abstentions**. All methods receive the same query ceilings. Ordinary replay uses eight conditional-moment orderings, collectively one statistical observation. The highest-mark method uses two distinct independently drawn target anchors per observation.

| CPU comparison | Safe / unsafe cases | Truncated two-source certificate | Ordinary trajectory replay | Real-proposal highest-mark method |
|---|---:|---:|---:|---:|
| Finite-state family, full frozen quota grid | 1,920 / 1,920 | 116 / 0 / 3,724 | 181 / 2 / 3,657 | 104 / 80 / 3,656 |
| Theorem regime, \(\sqrt{pq}=2^{23}\) | 96 / 96 | 0 / 0 / 192 | 2 / 1 / 189 | 3 / 3 / 186 |
| Same family, \(\sqrt{pq}=2^{32}\) | 96 / 96 | 96 / 0 / 96 | 81 / 0 / 111 | Not run |
| Same family, \(\sqrt{pq}=2^{36}\) | 96 / 96 | 96 / 0 / 96 | 96 / 0 / 96 | Not run |
| Held-out complete GUM sequences, \(p=32,\ q=65{,}536\) | 16 / 0 | 0 / 0 / 16 | 2 / 0 / 14 | 0 / 0 / 16 |

In the finite family with \(r=1/2\) and \(\sqrt{pq}=2^{23}\) fixed, set \(q/p\) to 4,096, 16,384, and 65,536. The least informative layer is \(q/p\) in all three, and minimum total KL is **3.592, 3.592, 3.594**, respectively. The likelihood ratio test with both candidate worlds known certifies 599, 607, and 603 safe instances out of 1,024 each. The executable auditor abstains on all 96 safe tests in that region. At \(2^{32}\) truncation first certifies 15 more than replay. The largest weight group at that quota uses **\(2^{24}\) target draws and \(2^{40}\) proposal scores** per instance. Lossless multinomial-count simulation implements these large independent draw counts on CPU. This supplementary in-regime analysis was run after the first results and is not a second holdout. See `dual_round/finite/blind-band-results.csv` in the archive and the [local consolidated ledger](../../results/pro/combined_experiment_ledger.csv).

**The most important real failure is on an almost matched proposal.** GUM uses eight unused documents with two proposals each. Models come from EWT; complete labeling sequences are enumerated. Outward-rounded intervals for all 16 true normalized risks lie below \(0.5\). For the mild proposal on `GUM_news_nasa-40`, maximum weight is only **1.011861**, true normalized risk **0.064149**, and the true tail mass after truncation exactly zero. Even with exact path moments, the target mean interval must lie within \([0.548738,0.973581]\) to certify. The new method obtains \([0.533822,0.963096]\), ordinary replay \([0.580522,0.949134]\). The **lower endpoint** flips the decision. See `dual_round/panel/gating-diagnostic.csv` for raw risks and intervals.

Holding each empirical cutoff fixed and supplying only exact proposal-side \(\theta\) while retaining the measured target residual interval makes **16/16** centering intervals pass the exact-path-moment gate. Supplying only the exact target residual makes **1/16** pass. Thus the prototype's principal loss on this panel is estimation of the ratio of two small bounded means. As the cutoff grows with \(q/p\), a small denominator amplifies additive error from generic confidence bounds. With fixed-time path testing the main-quota comparison remains zero certifications versus two for replay. Here \(L=16,K=2048\) is outside the theorem's power guarantee. See `dual_round/panel/component-check.csv` for the component ablation.

The eight target banks at the GUM main quota contain **256 independent target samples** in total, shared by the two proposals. The new method, replay, and highest-mark method use **1,048,576 / 937,984 / 2,861** proposal scores, and **512 / 0 / 512** additional candidate-specific scores on target draws. All GUM windows are safe, so they cannot estimate false certification on unsafe instances. A separate unpack-and-rerun matched the per-case records and 16 risk intervals. The source archive contains the proof, rational auditor, locked inputs, CPU commands, and two query ledgers: Cold-IMH-Two-Resource-Working-Files.zip (private archive).

## 3. Assessment

**Potentially suitable as a theory main result for ICML; current evidence does not establish an Oral-level practical algorithmic advantage.** The paper answers a question a total-query bound cannot: under which distribution class and risk margin must reducing one source's independent observations be offset by the other, by how much, and why that tradeoff really applies to the deployed cold chain. The practical missing lemma is a sufficiently narrow paid-observation confidence interval for proposal-side ratio \(\theta\) near \(P=Q\), while retaining false-certification control for as-yet-unseen high-weight states. The NASA case gives concrete endpoints for testing it. The results do not support explaining this abstention as inherently insufficient target sample count.
