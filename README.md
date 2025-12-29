# Evaluating the Robustness of Multimodal Large Language Models in Facial Expression Recognition: A Human-Comparative Study on Composite and Partial Faces

This repository contains materials and code accompanying our Pattern Recognition submission on evaluating **multimodal large language models (MLLMs)** for facial emotion understanding under two diagnostic settings:

1) **Semantic Conflict (Composite Faces)** — emotion perception under **cross-region semantic conflict** (e.g., top vs. bottom facial halves).
2) **Information Scarcity (Eye vs Eyebrow+Eye vs Whole face)** — fine-grained reasoning focused on **eye-region cues** under controlled presentation.

The repo is organized into two main folders, each providing:
- **Stimulus/material construction pipeline**
- **MLLM evaluation scripts** (prompting + parsing + aggregation)

> **Note on data licensing:** Due to third-party dataset licenses and privacy constraints, we may not redistribute some raw images/videos. This repo provides **reproducible construction recipes**, metadata templates, and scripts to rebuild stimuli from the original sources.

---

## Repository Structure

```text
.
├── semantic_conflict/          # Composite-face semantic conflict task
│   ├── materials/              # Stimulus construction docs/templates
│   ├── prompts/                # Prompt templates used for MLLM evaluation
│   ├── scripts/                # Stimulus generation + QC utilities
│   └── run_mllm_eval.py        # Main entry for model testing (example)
│
├── eye_local_region/           # Eye-region diagnostic task
│   ├── materials/
│   ├── prompts/
│   ├── scripts/
│   └── run_mllm_eval.py
│
├── analysis/                   # Optional: metrics + plotting to reproduce figures
│   ├── compute_metrics.py
│   └── plot_figures.py
│
├── configs/                    # Model + runtime configs (recommended)
│   ├── model_zoo.yaml
│   └── default_eval.yaml
│
├── results/                    # Output directory (created after running)
└── README.md

```text
---

## Prompting & Parsing

Prompt templates live in each task folder under `prompts/`.
We recommend keeping:

- a single canonical prompt per task
- strict output schema (e.g., JSON) for reliable parsing
- a clear retry policy for invalid outputs (logged transparently)

---

## Ethical / Responsible Use

This repository evaluates facial emotion recognition and perception under controlled research settings.
Please follow:

- dataset licenses and consent requirements
- provider policies for API-based models
- local regulations regarding biometric data


