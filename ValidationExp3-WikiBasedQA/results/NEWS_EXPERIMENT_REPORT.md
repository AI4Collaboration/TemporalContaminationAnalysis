# News Temporal Decay Additional Experiment

## Setup

- Dataset: `ValidationExp3-WikiBasedQA/data/news_mcq_dataset.json`
- Questions: 12 dated news MCQs (5 pre-cutoff, 7 post-cutoff)
- Cutoff date: `2023-10-01`
- Variants evaluated:
  - `original` questions
  - `transformed` semantically equivalent rewrites
- Models:
  - `gpt-3.5-turbo`
  - `gpt-4`
  - `gpt-4o-mini`

## Core Results

| Model | Original Pre | Original Post | Original Decay (Pre-Post) | Transformed Pre | Transformed Post | Transformed Decay (Pre-Post) | Decay Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-3.5-turbo | 60.0% | 42.9% | 17.1 | 60.0% | 57.1% | 2.9 | **14.3** |
| gpt-4 | 80.0% | 57.1% | 22.9 | 80.0% | 57.1% | 22.9 | **0.0** |
| gpt-4o-mini | 100.0% | 85.7% | 14.3 | 100.0% | 100.0% | 0.0 | **14.3** |

## Interpretation

- For `gpt-3.5-turbo` and `gpt-4o-mini`, temporal decay decreased substantially after transformation.
- For `gpt-4`, decay remained unchanged in this sample.
- This news-domain extension shows that format sensitivity can appear outside coding tasks, but effect size is model-dependent.

## Output Artifacts

- `ValidationExp3-WikiBasedQA/results/transformed_questions.json`
- `ValidationExp3-WikiBasedQA/results/predictions.csv`
- `ValidationExp3-WikiBasedQA/results/summary_by_model.csv`
- `ValidationExp3-WikiBasedQA/results/summary_by_model.json`

## Repro Command

```bash
python ValidationExp3-WikiBasedQA/run_news_temporal_experiment.py \
  --models gpt-3.5-turbo,gpt-4,gpt-4o-mini \
  --cutoff-date 2023-10-01
```
