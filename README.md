# DREAM 2025 – Team TREATS

**Source code for the DREAM 2025 Challenge**
🔗 [Challenge Page](https://www.synapse.org/Synapse:syn66496696/wiki/632412)

---

## Overview

This repository contains the complete workflow, code, and model artifacts developed by **Team TREATS** for the **DREAM 2025 Challenge**.
It includes data preprocessing, deviance-based feature selection, predictive model training, and generation of final submission files.

The project integrates both **Python** and **R** code and provides **Dockerfiles** for fully reproducible environments.

---

## Repository Structure

### 🔧 Key Scripts

| File                                          | Description                                                                                                                                                                                                |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`chunking.py`**                             | Reads large single-cell datasets (e.g. `.h5ad`) and splits them into manageable chunks for downstream processing or cross-validation.                                                                      |
| **`deviant_genes.py`**                        | Implements deviance-based ranking and feature selection, identifying genes most deviant across pathological conditions.
|
| **`run_task1.py` / `run_task2.py`**           | High-level pipeline scripts coordinating preprocessing, feature selection, model training, and submission file generation for Task 1 and Task 2 respectively. Each accepts CLI arguments and config files. |
| **`Analysis.R`**                              | R script for exploratory analysis, evaluation, and visualization. Depends on R packages such as `caret`, `Metrics`, `xgboost`, `scry`, etc.                                                                |
| **`Submission_model*.json` / `.pkl`**         | Packaged model configurations and final trained models used for DREAM challenge submissions.
|
| **`reg_model_*.json` / `class_model_*.json`** | JSON configuration files specifying model architectures and hyperparameters for regression and classification tasks.                                                                                       |

---

## 🐳 Docker Usage

Two Dockerfiles are provided for reproducibility:

* `Dockerfile_Task1`
* `Dockerfile_Task2`

### Build the Image (Example: Task 1)

```bash
docker build -f Dockerfile_Task1 -t dream_task1:latest .
```

### Run the Container (Example: Task 1)

```bash
docker run --rm --network none \
  --volume /input:/input:ro \
  --volume $PWD/output:/output:rw \
  dream_task1:latest
```

---

## 🧬 Notes

* Input data should be preprocessed according to the DREAM 2025 challenge data structure.
* Scripts are modular and can be adapted for new datasets or model configurations.
* The repository mixes **Python (machine learning / pipeline orchestration)** and **R (analysis / visualization)** components.

---

**Team TREATS — DREAM 2025 Challenge**
