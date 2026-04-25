# Data directory

This folder is intentionally kept in git **without bundled dataset JSONs**.

For release hygiene, generate the news MCQ dataset locally:

```bash
python ValidationExp3-WikiBasedQA/build_news_dataset_from_wikipedia.py \
  --start-date 2023-05-01 \
  --end-date 2024-06-30 \
  --target-size 220 \
  --output ValidationExp3-WikiBasedQA/data/news_mcq_dataset_200plus.json
```

The generated JSON files are local artifacts and should not be committed.
