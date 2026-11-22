# Reproduce the cold-start two-resource audit on CPU

Run in the extracted archive's top-level directory (the directory containing `dual_round/` and `MANIFEST.sha256`). All model, corpus, C source, and frozen random seeds are included. No account, GPU, neural model download, or target normalizer is required. Historical banks can be regenerated locally; their large uncompressed arrays are deliberately omitted. Work in a writable directory.

## Software

The validation environment was Linux x86-64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, python-flint 0.9.0 and GCC 14.2.0. The first three Python packages and GCC cover the main Monte Carlo and GUM audit; python-flint is used only for its independent outward-rounded numerical reference. For a preprovisioned Python environment the experiments themselves work offline. The archive does not include Python, libraries or a compiler.

~~~bash
sha256sum --check MANIFEST.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p .replay-tmp
export TMPDIR="$PWD/.replay-tmp"
unset EWT_DATA
~~~

## Finite controls, including unsafe cold chains

~~~bash
sha256sum --check dual_round/finite/PROTOCOL.sha256
python dual_round/finite/run.py
sha256sum --check dual_round/finite/BAND-CHECK.sha256
python dual_round/finite/band.py
python dual_round/math/verify.py
python dual_round/math/rational_gate.py
~~~

`finite/truth.csv` gives exact rational risks and margins for eight finite worlds. `finite/results.csv` summarizes 15,360 replicate-by-method records, available individually as `finite/replicates.csv.gz`. This last file has a time stamp in the gzip wrapper; compare its decompressed contents across replays. The 48 replicate experiments per setting share source information across methods. Proposal rows are generated in distribution using multinomial counts of ordered length-four blocks, with explicit prefixes for the signed competing method. A multinomial count of one million draws does not imply that one million model forward passes were executed. The later in-band least-information check has its own status and frozen settings in `finite/BAND-CHECK.md`; `band-truth.csv`, `band-envelope.csv`, and `band-results.csv` in the same folder are the outputs. The two-world likelihood-ratio calculation receives the two world specifications and serves only as an information diagnostic. The theoretical algorithms, fixed-budget gate and proof are under `dual_round/math/`.

## Blind full-sequence transfer

~~~bash
sha256sum --check dual_round/panel/decision-freeze.sha256
python dual_round/panel/freeze_ids.py dual_round/panel/source/en_gum-ud-test.conllu
sha256sum --check dual_round/panel/input-digest.sha256
python dual_round/panel/prepare.py
python dual_round/panel/audit.py
python dual_round/panel/evaluate.py
python dual_round/panel/score.py
python dual_round/panel/reference_interval.py
~~~

The source-only audit writes `panel/issues.csv` before the evaluator runs. The evaluator enumerates the entire 17³- or 17⁴-state posterior and computes the cold-chain risk for each whole sentence and proposal. Exact finite-state transitions and zero Monte Carlo error do not eliminate floating-point error. An independent interval computation encloses the risk of the fitted binary64 model, with one lower/upper pair per window in `panel/certified-reference.csv`. The joined 192-row ledger `panel/runs.csv` records target draws, cross-scores, proposal draws, abstention and truth for every method/budget/candidate. The main comparison uses eight documents, each with two proposals, at 32 target rows and 65,536 proposal rows. The other quotas were preregistered secondary comparisons; no cross-method or cross-budget selection guarantee is claimed.

The diagnostics below use only already-frozen cases. `diagnose.py` and `component_check.py` access truths exclusively for the labeled oracle diagnostics. `terminal_check.py` applies both auditors' identical fixed-time confidence gate, imports it directly from the finite code, and retains the original data partition.

~~~bash
python dual_round/panel/diagnose.py
python dual_round/panel/terminal_check.py
python dual_round/panel/component_check.py
~~~

These produce `gating-diagnostic.csv`, `terminal-check.csv` and `component-check.csv` under `dual_round/panel/`. For query accounting and individual failures see `dual_round/panel/README.md`.

## Historical failure ablation

~~~bash
python price_round/banks/make_banks.py
python dual_round/panel/old_ablation.py
~~~

This rebuilds approximately 147 MiB of P/Q arrays from the EWT train/test corpora, original capture code, deterministic seeds, document exclusion ledgers and the original first-pilot checks. The 32 per-candidate P-identity/Q-row hashes and original pilot compatibility checks are in `price_round/banks/sources.csv`; the independent historical evaluations used as approximate truth are in `cold_round/ewt/final-raw/truth.csv`. To regenerate those older Monte Carlo references at their original seeds, run `python cold_round/ewt/final_raw.py` before bank generation. The oracle gift calculations for the archived fixed window produce 128 rows in `dual_round/panel/old-ablation.csv`. These older references have Monte Carlo uncertainty.

## Source files and attribution

The complete unmodified Universal Dependencies GUM r2.15 test corpus is included so that document selection and exclusions can be checked without network access. Its SHA-256 is `99deb5746618d8ab83e5cafc2d07851e76389b2c8babf5ad9e5ef176a4619114`. Original document authors, titles and source URLs are present in the corpus; the eight used sources are separately listed in `dual_round/panel/GUM-attributions.txt`. GUM terms are in `dual_round/panel/LICENSE-GUM.txt`. [Source](https://github.com/UniversalDependencies/UD_English-GUM/tree/r2.15)

The archived EWT train and test SHA-256 digests are respectively `a049fc40822e21593ff89ccd0a4b081443f09fae71b61f813c624aa598f6c566` and `0612e45914359e4b36d187ef2bfeaf66fc8933210925be985c5fbf6a9e311036`. EWT corpus attribution, separate rights of the source texts and license are in `round3/sequence/README-EWT.txt` and `round3/sequence/LICENSE.txt`. [Source](https://github.com/UniversalDependencies/UD_English-EWT)

Full source texts, trained-model inputs and evaluators are packaged for auditing reproducibility. Access to evaluation-only objects is excluded in the statistical observation model.

## Same-family in-band power and main table

Run the following from the archive root. The final finite check applies the frozen auditors without revealing the pair of possible distributions. Its settings were written after the first panel result. See `dual_round/finite/BLIND-BAND.md` for the distinction between this check and the original holdout. Only the two path methods were run at the two largest in-band caps. Independent rows at those scales are simulated through sufficient multinomial counts, with every method's oracle exposures charged.

~~~bash
sha256sum --check dual_round/finite/BLIND-BAND.sha256
python dual_round/finite/blind_band.py
python dual_round/main_table.py
~~~

`dual_round/main-table.csv` reports the primary finite and GUM panels including paid scores on P, score access to Q, paid independent P draws, and abstentions. The simulation prepared 1,024 unique independent GUM target sequences and 1,048,576 unique proposal sequences once for all caps, methods and diagnostic comparisons. At the primary cap only 256 of these target sequences could enter any auditor. Two evaluated proposals reuse the same sequence per document and take separate candidate-specific cross-scores. Offline enumeration of all labelings is performed by the evaluator only. Per-method Q access is actual prefix consumption; generating the common full banks once is a separate operation.
