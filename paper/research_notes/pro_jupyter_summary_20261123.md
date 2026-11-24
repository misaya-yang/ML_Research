# Pro Jupyter Work: What to Keep

## Main result

On a separate 56-task MBPP panel, Qwen2.5-1.5B used a 64-particle Power-SMC population. The policy checked up to four distinct programs in slot order and returned the first passing the public test. All three implementations (full, terminal-only pruning, and pruning across checkpoints) agreed on the recorded verification path on all 56 tasks. Independent hidden checks graded each arm at **35 correct, 6 wrong, and 15 abstentions**. The full and across-checkpoint implementations executed 128,459 and 108,641 model positions, respectively, including the same 6,117 prefill positions (15.43% fewer overall). This establishes savings for the same policy on this panel, not superiority over a different policy.

Later completed same-machine controls closed the missing direct comparison; see [the matched deadline results](../../results/first_passage/forward_search.md#an-independent-56-task-replication-at-real-service-deadlines). At eight seconds cross-checkpoint pruning delivered 32 correct versus 24 for the unaccelerated population and 32 for the best direct policy. There is no sustained joint win over both at preset wall-time budgets or any target-model position budget. One case crosses the eight-second deadline relative to both controls, but the case was identified retrospectively. Same-machine direct runs were not a randomized cross-policy latency trial.

The post-hoc stress check examined all five correctness disagreements against greedy plus **three** retries, not eight. Two programs returned only by the particle policy passed 48/48 and 49/49 additional inputs; another particle program passed the original hidden tests but only 53/61 additional inputs. Some disagreements did not survive these extra checks. The inputs are exploratory, and the local MBPP+ data snapshot is not evidence of a completed cohort-wide regrading.

## Earlier notebook screen

The separate cold-start independent-MH screen found no exclusive correct rescue on any of 64 new MBPP proposal tapes at a three-proposal cap. Exact integration over acceptance uniforms gave zero equivalent rescues and 0.1439 equivalent losses. A source-text rule also solved all six WikiText year development items with no model forwards. These controls argue against promoting those workloads to the main experiment; they do not rule out feedback-dependent proposals or longer budgets.

## Theory worth retaining

The finite-policy argument uses prefix-derived upper bounds on future weights. A weight box can certify that an intermediate ESS trigger is impossible, or that systematic resampling with the same offset has an invariant parent map. Terminal selection and the ordered four-probe verifier require separate certificates: a coupling that preserves a single final winner need not preserve multiple ordered probes. The four-probe counterexample shows why a known losing particle cannot always be skipped: its weight changes whether the accepting particle survives resampling. In the stated 64-particle prefix-query model, the two output laws differ by `57344/207617`, giving a 27.62% query-probability lower bound in that witness, not a demand measured on real tasks.

## Local evidence

The receipts are kept untracked under the existing `results/local_runs/` ignore rule. In the local `native-firstpass-1qr7b1va/feedback_policy_v2/` directory, see `population_replication56/population_replication.jsonl`, `population_replication56/fixed_verify_22005f2708c3.jsonl`, `holdout/iid_frontier_report.json`, `semantic_check.log`, and `finite_particle_result.md`. Do not add the notebooks, logs, or intermediate files to Git.


## Semantic decoding feasibility

The separate 24-item MBPP development test found no rescue from execution-line localization: 15/24 correct for greedy, localized, and nonlocalized generation, versus 16/24 for direct retry. Even a selector with hidden labels cannot improve the localized pool. In the 56-item replay, permissive early submission harmed two tasks that needed a second return branch. A return-structure guard matched existing post-generation syntax cleanup at 35/56 correct at the nine-attempt cap, saving only 180 generated tokens while adding 57 public checks. This is a replay, not an online latency measurement.

On 244 miniF2F tasks, permitting completion of a first draft added one theorem beyond direct answers plus tactic automation (157 to 158). Subset ablation showed that two generated equalities enabled that proof. An after-the-fact no-model algebraic control also proved it, and the same intermediate-hint procedure gave 0/42 on a fixed hard subset. The positive example identifies a causal mechanism, not a growing capability advantage.

## First-proof study: archive evidence, not a completed device rerun

The current first-proof notebook reconstructs first verified proof under an eight-attempt policy from 992 archived Goedel-Pset requests: 508 succeeded on the first attempt and 577 by eight. This is a serving policy reconstructed from a generation campaign, not the original campaign's objective. Attribution must use common public proof lookup and Lean automation in both arms; seven attempts after that common front end preserve the 577 successful archive items in their recorded order, but this is not a distributional guarantee. The same notebook reports 82 formal refutations from existing SMT/SlimCheck, and 101 with the additional checks. The added 19 are all archive failures after eight attempts. This can improve delivery cost only after charging the useful front end, proof searches, and verification in the same service budget. Archive L4 allocation, CPU timing, and in-progress M4 jobs are separate measurements; do not join them or count active jobs as finished.

Local source notebooks: `semantic-decoding-feasibility-8ds62rrm/Findings.ipynb` and `first-proof-study-20261123/ICML2027_First_Proof_Current.ipynb` under the ignored run directory. Raw notebooks, attached receipts, and live job directories remain on the device.
