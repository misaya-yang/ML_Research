# Pro Jupyter Work: What to Keep

## Main result

On a separate 56-task MBPP panel, Qwen2.5-1.5B used a 64-particle Power-SMC population. The policy checked up to four distinct programs in slot order and returned the first passing the public test. All three implementations (full, terminal-only pruning, and pruning across checkpoints) agreed on the recorded verification path on all 56 tasks. Independent hidden checks graded each arm at **35 correct, 6 wrong, and 15 abstentions**. The full and across-checkpoint implementations executed 128,459 and 108,641 model positions, respectively, including the same 6,117 prefill positions (15.43% fewer overall). This establishes savings for the same policy on this panel, not superiority over a different policy.

The strongest completed direct comparison, greedy plus seven or eight IID retries, also got **35 correct**, with **9 wrong and 12 abstentions**. The eight-retry comparator used 39,996 model positions in the archived run; the wall-time figures in that comparison are reconstructed, not matched serving measurements. The shared-prefill rerun was incomplete at this review, and no completed matched parallel-IID comparison was in the local receipts. Do not claim a matched-cost or wall-time win until those runs finish and hidden correctness, public acceptance, and abstention are accounted for separately. This 56-task panel, which excludes the earlier 24 pilot tasks, is distinct from the four-particle 24-task winner-only analysis and the 30-item paired jump experiment.

The post-hoc stress check examined all five correctness disagreements against greedy plus **three** retries, not eight. Two programs returned only by the particle policy passed 48/48 and 49/49 additional inputs; another particle program passed the original hidden tests but only 53/61 additional inputs. Some disagreements did not survive these extra checks. The inputs are exploratory, and the local MBPP+ data snapshot is not evidence of a completed cohort-wide regrading.

## Earlier notebook screen

The separate cold-start independent-MH screen found no exclusive correct rescue on any of 64 new MBPP proposal tapes at a three-proposal cap. Exact integration over acceptance uniforms gave zero equivalent rescues and 0.1439 equivalent losses. A source-text rule also solved all six WikiText year development items with no model forwards. These controls argue against promoting those workloads to the main experiment; they do not rule out feedback-dependent proposals or longer budgets.

## Theory worth retaining

The finite-policy argument uses prefix-derived upper bounds on future weights. A weight box can certify that an intermediate ESS trigger is impossible, or that systematic resampling with the same offset has an invariant parent map. Terminal selection and the ordered four-probe verifier require separate certificates: a coupling that preserves a single final winner need not preserve multiple ordered probes. The four-probe counterexample shows why a known losing particle cannot always be skipped: its weight changes whether the accepting particle survives resampling. In the stated 64-particle prefix-query model, the two output laws differ by `57344/207617`, giving a 27.62% query-probability lower bound in that witness, not a demand measured on real tasks.

## Local evidence

The receipts are kept untracked under the existing `results/local_runs/` ignore rule. In the local `native-firstpass-1qr7b1va/feedback_policy_v2/` directory, see `population_replication56/population_replication.jsonl`, `population_replication56/fixed_verify_22005f2708c3.jsonl`, `holdout/iid_frontier_report.json`, `semantic_check.log`, and `finite_particle_result.md`. Do not add the notebooks, logs, or intermediate files to Git.
