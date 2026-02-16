# Time Will Tell: Rethinking Temporal Pattern of Contamination 

<div align="center">

#### [📄 Paper](https://arxiv.org/)
</div>

## Overview

This repository contains code and supplementary materials for reproducing three related experiments analyzing model memorization vs. reasoning ability:

- **Main experiment**: Download arXiv papers, extract theorem-like content from LaTeX, generate QA pairs, and evaluate models.
- **Validation 1 (CLOZE)**: Generate fill-in-the-blank items from paper abstracts and measure model predictions.
- **Validation 2 (Perturbed LiveCodeBench)**: Apply code transformations and compare model robustness.

---

## Quick Start

### 1. Set up environment

```powershell
# Install dependencies (uv recommended)
uv sync

# Activate virtual environment
.venv\Scripts\activate

# Or with pip
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. Configure API keys

```powershell
$env:OPENAI_API_KEY = "your-key-here"
$env:OPENROUTER_API_KEY = "your-key-here"
```

Or create a `.env` file in the root directory.

---

## Experiments

### MainExp: QA Generation & Evaluation

**Scripts**: `MainExp/` folder + `helpers/` utilities

Key files:
- `MainExp/monthly_qa_pipeline.py` — orchestrate the full pipeline
- `MainExp/eval.py` — evaluate models on QA datasets
- `MainExp/datacollate.py` — merge per-month outputs
- `helpers/arxiv_retriever.py` — fetch papers from arXiv
- `helpers/extract_latex_text.py` — parse LaTeX
- `helpers/extract_theorems.py` — identify theorem statements
- `helpers/generate_qa.py` — generate QA pairs via LLM

**Example**:
```powershell
# Run pipeline for May 2024
python MainExp/monthly_qa_pipeline.py --year 2024 --start 5 --end 5 --categories math --papers-step 50

# Collate results
python MainExp/datacollate.py --input output --output data

# Evaluate on a model
python MainExp/eval.py --dataset data --output results --model o4-mini
```

**Output**: 
- `output/<category>/<topic>/<YYYY>/<MM>/` — papers, LaTeX, theorems, QA pairs
- `data/` — consolidated datasets for evaluation
- `results/` — model outputs and metrics

---

### ValidationExp1: CLOZE Questions from Abstracts

**Scripts**: `ValidationExp1-CLOZEusingAbstracts/`

Key files:
- `generate_and_evaluate_cloze_abstracts.py` — generate and evaluate CLOZE questions
- `CLOZEonRealMathPapers/` — example datasets and results
- Notebooks for exploratory analysis

**Example**:
```powershell
cd ValidationExp1-CLOZEusingAbstracts
python generate_and_evaluate_cloze_abstracts.py
```

**Output**: `evaluation_*_results.json` and `evaluation_*_scores.json` with metrics (ROUGE, BLEU, F1, exact match)

---

### ValidationExp2: Perturbed LiveCodeBench

**Scripts**: `ValidationExp2-PerturbedLiveCodeBench/`

Key files:
- `livecodebench.py` — apply transformations and run evaluations
- `analysis.py` — aggregate results and compute metrics
- `plotting.py` — generate visualizations
- `README.md` — script-only run instructions for this folder

**Example**:
```powershell
cd ValidationExp2-PerturbedLiveCodeBench
python livecodebench.py
python analysis.py
python plotting.py
```

**Output**: `evaluation_results.json`, printed monthly stats, and `accuracy_plot.png`

---

## Environment Setup Details

All experiments use a **unified environment** configured via `pyproject.toml` at the repository root.

### Dependencies

- **ML/AI**: openai, anthropic, datasets, huggingface-hub
- **Data**: pandas, numpy, scipy, nltk
- **Visualization**: matplotlib, seaborn, plotly
- **Document processing**: arxiv, PyMuPDF, beautifulsoup4
- **Web**: aiohttp, httpx, requests, selenium
- **Utilities**: tqdm, pydantic, click, PyYAML

### Python version

Requires **Python ≥ 3.11** (due to numpy compatibility)

### Install methods

**With uv (recommended)**:
```powershell
uv sync
.venv\Scripts\activate
```

**With pip**:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

**With dev dependencies**:
```powershell
uv sync --all-extras
# or
pip install -e ".[dev]"
```

---

## File Structure

```
├── MainExp/                          # Main QA generation & evaluation
│   ├── monthly_qa_pipeline.py
│   ├── eval.py
│   ├── datacollate.py
│   └── count_qa_pairs.py
├── helpers/                          # Shared utilities
│   ├── arxiv_retriever.py
│   ├── extract_latex_text.py
│   ├── extract_theorems.py
│   ├── generate_qa.py
│   └── prompts.py
├── ValidationExp1-CLOZEusingAbstracts/  # CLOZE validation
│   ├── generate_and_evaluate_cloze_abstracts.py
│   ├── CLOZEonRealMathPapers/
│   └── *.ipynb
├── ValidationExp2-PerturbedLiveCodeBench/  # Code robustness validation
│   ├── livecodebench.py
│   ├── analysis.py
│   ├── plotting.py
│   └── README.md
├── pyproject.toml                   # Unified dependency configuration
├── uv.lock                          # Locked dependencies
└── README.md                        # This file
```

---

## Troubleshooting

- **Import errors**: Ensure environment is activated and all dependencies installed (`uv sync`)
- **API key errors**: Check `.env` file or environment variables are set correctly
- **Script path errors**: Run scripts from repository root, not from subdirectories
- **Logs**: Check `output/*/logs/` for per-month pipeline logs

---

## Citation

If using this work in research, please cite appropriately. Paper links and datasets will be available on publication.

---


