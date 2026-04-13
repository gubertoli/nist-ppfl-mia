# NIST Genomics PPFL — Stacked Generalization MIA

This repository contains the implementation and evaluation code for the paper:

> **Evaluating Differential Privacy Against Membership Inference in Federated Learning: Insights from the NIST Genomics Red Team Challenge**  

The methodology achieved **first place** on the [2025 NIST Genomics Privacy-Preserving Federated Learning Red Teaming Event](https://pages.nist.gov/genomics_ppfl/) (Problem 1 — Membership Inference) under the No DP and Low Privacy tiers.

---

## Repository Structure

```
.
├── run_ensemble_attack.py     # Main attack pipeline (stacked generalization MIA)
├── compare_ensemble_lira.py   # TPR@low-FPR evaluation: ensemble vs. LiRA baseline
├── run_fl_experiments.sh      # FL simulation across privacy tiers and epoch settings
├── setup.sh                   # Automated environment setup
├── utils/
│   └── utils.py               # Shared utilities: data loading, feature extraction, seeding
├── fl/                        # Flower FL simulation (client/server apps)
│   ├── client_app.py
│   ├── server_app.py
│   └── task.py
└── output/                    # Generated submissions, logs, and evaluation results
```

> **Note:** The `problem1/` directory (NIST dataset) is excluded from the repository via `.gitignore`. It is downloaded automatically by `setup.sh`.

---

## Setup

Tested on Ubuntu Linux 24.04.

```bash
bash setup.sh
```

This will download the NIST dataset, unzip it, install `uv` if not present, and synchronise the Python environment.

---

## Membership Inference Attack

The stacked generalization attack trains seven ART black-box base estimators (NN, RF, DT, GB, KNN, SVM, LR) on Client 4's ground-truth membership labels, then applies a partial-fit domain adaptation strategy to generalise to Clients 1–3.

```bash
uv run run_ensemble_attack.py
```

Outputs are saved to `output/` as `{tier}_ensemble_detailed_submission.csv`, containing per-client membership probabilities and final predictions for all 73 challenge records across three privacy tiers (No DP, Low DP ε=200, High DP ε=10).

---

## TPR @ Low FPR Evaluation

Compares the ensemble against a variance-based LiRA baseline on Client 4's 116 labeled records (13 members, 103 non-members) using 5-fold stratified cross-validation.

```bash
uv run compare_ensemble_lira.py
```

Reports TPR at ≤1% and ≤3% FPR thresholds for each privacy tier.

---

## Federated Learning Utility Benchmark

Simulates the full FL pipeline using [Flower](https://flower.ai) across three privacy tiers and local epoch settings {1, 5, 20} over 50 communication rounds.

```bash
bash run_fl_experiments.sh
```

Logs are saved to `output/logs/run_{tier}_R50_E{epochs}.log`.

---

## Reproducibility

All random seeds are fixed via `utils.set_seed(42)`. For full CPU determinism, set the following environment variables **before** running any script:

```bash
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## Cite this work

```bibtex
```

