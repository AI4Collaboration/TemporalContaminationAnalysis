# ValidationExp3-WikiBasedQA

Additional experiment for reviewer request: test whether temporal decay behavior changes under semantically equivalent question transformations on **dated news questions**.

## What this experiment does

1. Loads dated news MCQ questions (`data/news_mcq_dataset.json`).
2. Generates transformed (paraphrased) versions while preserving answer/options.
3. Evaluates selected OpenAI models on both original and transformed questions.
4. Splits accuracy into pre/post cutoff windows and reports decay deltas.

## Build 200+ dated news questions

From repo root:

```bash
python ValidationExp3-WikiBasedQA/build_news_dataset_from_wikipedia.py \
  --start-date 2023-05-01 \
  --end-date 2024-06-30 \
  --target-size 220 \
  --output ValidationExp3-WikiBasedQA/data/news_mcq_dataset_200plus.json
```

## Run

From repo root:

```bash
python ValidationExp3-WikiBasedQA/run_news_temporal_experiment.py \
  --dataset ValidationExp3-WikiBasedQA/data/news_mcq_dataset_200plus.json \
  --models gpt-3.5-turbo,gpt-4,gpt-4o-mini,gpt-4o,gpt-4.1-mini,o4-mini \
  --cutoff-date 2023-10-01 \
  --batch-size 20
```

Optional:

```bash
python ValidationExp3-WikiBasedQA/run_news_temporal_experiment.py \
  --dataset ValidationExp3-WikiBasedQA/data/news_mcq_dataset_200plus.json \
  --output-dir ValidationExp3-WikiBasedQA/results \
  --transform-model gpt-4o-mini \
  --max-questions 200 \
  --batch-size 15
```

## Outputs

- `results/transformed_questions.json`
- `results/predictions.csv`
- `results/summary_by_model.csv`
- `results/summary_by_model.json`

## Notes

- Requires `OPENAI_API_KEY` in environment.
- If a model is unavailable for your account, the script records errors and continues with other models.
