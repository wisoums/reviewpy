# ReviewPy File Guide

Quick reference for the main files in the project.

## Data

`data/metadata/dataset_selection.md`: Documents why we selected the GitHub code-review dataset. It records the pinned dataset revision, dataset statistics, audit findings, risks, and licensing notes.

`data/metadata/cleaning_summary.md`: Summarizes how the raw dataset is cleaned before training. It records the filters applied and the final train, validation, and test counts.

`data/metadata/split_manifest.json`: Freezes the final dataset splits used by ReviewPy. It stores split counts, SHA256 hashes, class balance, and leakage-check results so we can verify later that the test set has not changed.

`data/processed/`: Contains the generated `train.jsonl`, `validation.jsonl`, and `test.jsonl` files. These files are produced locally by the preprocessing pipeline and are not committed to Git.

`data/metadata/base_evaluation.json` : Stores the official base-model benchmark on the frozen test set. It records the pinned Qwen revision, test-set hash, environment, and baseline metrics used for later comparison with the fine-tuned model.

## Scripts

`scripts/inspect_dataset.py`: A simple exploration script used to understand the original Hugging Face dataset. It shows available splits, Python example counts, and sample fields.

`scripts/audit_dataset.py`: Audits the original dataset before cleaning. It checks things such as missing data, comment types, positive/negative balance, duplicate comments, diff sizes, and potentially context-dependent reviews.

`scripts/prepare_dataset.py`: Converts the original Hugging Face dataset into the cleaned ReviewPy training dataset. It filters unusable examples, formats the data for Qwen, removes duplicate/leaking examples, and generates the processed train, validation, and test files.

`scripts/validate_splits.py`: Checks whether the cleaned train, validation, and test sets are safely separated. It looks for exact examples, repeated diffs, meaningful normalized-diff matches, and pull requests appearing across multiple splits. It exits with an error if any overlap is found or if the splits no longer match `data/metadata/split_manifest.json`. It only rewrites the manifest when run with `--write-manifest`, and it will not write one while any overlap remains.

## Configuration

`configs/base.yaml`: Stores the main reusable project configuration. It defines the ReviewPy domain, model, training defaults, MLflow experiment, evaluation settings, and serving configuration.

`pyproject.toml`: Defines the Python package and project dependencies. It also contains development configuration such as supported Python versions, Ruff, pytest, and optional training/data/serving dependencies.

## Project Documentation

`README.md`: The public introduction to ReviewPy. It explains what the project does, the architecture, planned CLI usage, setup, and current project status.

`SECURITY.md`: Defines repository safety rules. It is especially important for dataset/model licensing, secrets, private data, and artifacts that should not be committed.

`docs/FILES.md`: This file. It provides a short explanation of what each important project file is responsible for.