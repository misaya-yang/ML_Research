# First-answer steering experiment

The frozen pilot and follow-up are described in [the research note](../../paper/research_notes/first_answer_steering_20261123.md). The development gate failed; the reserved benchmark configurations are not run authorizations.

The source, configs, and manifests are versioned, but original panel JSON is deliberately device-local under `panels/`. A clean checkout needs the frozen local panels (or byte-identical reconstruction from their original benchmark versions) to run the frozen checks. Run `verify_frozen.py` before using these inputs. Do not substitute a different holdout or change a manifest to pass a check.
