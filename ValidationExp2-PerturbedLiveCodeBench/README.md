# ValidationExp2 - Perturbed LiveCodeBench (Script-Only)

This folder is self-contained and runnable without notebooks.

## Files

- `livecodebench.py` - main evaluation pipeline (load benchmark, perturb prompts, evaluate models)
- `analysis.py` - extract month-level accuracy arrays from `evaluation_results.json`
- `plotting.py` - generate accuracy plot from extracted arrays

## Run end-to-end

From repository root:

```bash
cd ValidationExp2-PerturbedLiveCodeBench

# 1) Run evaluation
python livecodebench.py

# 2) Summarize monthly accuracies
python analysis.py

# 3) Plot results
python plotting.py

```

## Environment

- Requires `OPENAI_API_KEY` for perturbation and model calls.
- Uses dependencies declared in the repository `pyproject.toml`.

## Outputs

- `evaluation_results.json` (from `livecodebench.py`)
- extracted monthly arrays printed by `analysis.py`
- `accuracy_plot.png` (from `plotting.py`)
