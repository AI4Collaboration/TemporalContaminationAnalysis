#!/usr/bin/env python3
import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


def chunked(items: List[dict], chunk_size: int) -> List[List[dict]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            candidate = text[start:end].strip()
            if candidate.startswith("[") and candidate.endswith("]"):
                return candidate
    l = text.find("[")
    r = text.rfind("]")
    if l != -1 and r != -1 and r > l:
        return text[l : r + 1]
    return None


@dataclass
class QuestionItem:
    id: str
    date: str
    question: str
    options: Dict[str, str]
    answer: str
    topic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="News temporal decay experiment with question transformations"
    )
    parser.add_argument(
        "--dataset",
        default="ValidationExp3-WikiBasedQA/data/news_mcq_dataset.json",
        help="Path to dated news MCQ dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="ValidationExp3-WikiBasedQA/results",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--models",
        default="gpt-3.5-turbo,gpt-4,gpt-4o-mini",
        help="Comma-separated model names to evaluate",
    )
    parser.add_argument(
        "--transform-model",
        default="gpt-4o-mini",
        help="Model used to paraphrase questions",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2023-10-01",
        help="Cutoff date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Optional limit for quick runs (0 means all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of questions per batched API request",
    )
    return parser.parse_args()


def load_dataset(path: Path, max_questions: int = 0) -> List[QuestionItem]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = [QuestionItem(**obj) for obj in raw]
    items.sort(key=lambda x: x.date)
    if max_questions > 0:
        return items[:max_questions]
    return items


def format_options(options: Dict[str, str]) -> str:
    ordered = []
    for letter in ["A", "B", "C", "D"]:
        ordered.append(f"{letter}. {options[letter]}")
    return "\n".join(ordered)


def transform_question(client: OpenAI, model: str, item: QuestionItem) -> str:
    prompt = (
        "Rewrite the following multiple-choice news question into a semantically equivalent variant. "
        "Keep exactly the same factual target and difficulty. Do not reveal the answer. "
        "Return only the rewritten question text.\n\n"
        f"Original question: {item.question}\n"
        f"Options:\n{format_options(item.options)}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce concise, semantically equivalent rewrites."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=180,
    )
    text = response.choices[0].message.content.strip()
    if not text:
        return item.question
    return text


def transform_questions_batched(
    client: OpenAI,
    model: str,
    items: List[QuestionItem],
    batch_size: int,
) -> Dict[str, str]:
    out = {item.id: item.question for item in items}

    for batch in chunked(
        [
            {
                "id": item.id,
                "question": item.question,
                "options": item.options,
            }
            for item in items
        ],
        batch_size,
    ):
        prompt = (
            "Rewrite each question into a semantically equivalent news-question variant. "
            "Keep factual target and difficulty the same. Do not reveal any answer letter. "
            "Return only JSON array with objects: {\"id\":..., \"transformed_question\":...}.\n\n"
            f"Items:\n{json.dumps(batch, ensure_ascii=False)}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You produce concise semantic rewrites and strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2200,
            )
            text = (response.choices[0].message.content or "").strip()
            json_block = extract_json_block(text)
            if not json_block:
                continue

            parsed = json.loads(json_block)
            if not isinstance(parsed, list):
                continue
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                qid = row.get("id")
                tq = (row.get("transformed_question") or "").strip()
                if qid in out and tq:
                    out[qid] = tq
        except Exception:
            continue

    return out


def ask_model_mcq(client: OpenAI, model: str, question_text: str, options: Dict[str, str]) -> str:
    prompt = (
        "Answer this multiple-choice question. "
        "Return exactly one letter: A, B, C, or D. No explanation.\n\n"
        f"Question: {question_text}\n"
        f"Options:\n{format_options(options)}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer MCQs with a single letter only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    raw = (response.choices[0].message.content or "").strip().upper()
    for ch in raw:
        if ch in {"A", "B", "C", "D"}:
            return ch
    return "INVALID"


def ask_model_mcq_batched(
    client: OpenAI,
    model: str,
    rows: List[Dict],
    batch_size: int,
) -> Dict[str, str]:
    answers = {row["id"]: "ERROR" for row in rows}

    for batch in chunked(rows, batch_size):
        payload = [
            {
                "id": row["id"],
                "question": row["question_text"],
                "options": row["options"],
            }
            for row in batch
        ]
        prompt = (
            "Answer each multiple-choice question with exactly one letter A/B/C/D. "
            "Return only JSON array: [{\"id\":...,\"answer\":\"A\"}, ...].\n\n"
            f"Questions:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You answer MCQs and return strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1800,
            )

            text = (response.choices[0].message.content or "").strip()
            json_block = extract_json_block(text)
            if not json_block:
                continue

            parsed = json.loads(json_block)
            if not isinstance(parsed, list):
                continue
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                qid = row.get("id")
                pred = str(row.get("answer", "")).strip().upper()
                if pred not in {"A", "B", "C", "D"}:
                    pred = "INVALID"
                if qid in answers:
                    answers[qid] = pred
        except Exception:
            continue

    return answers


def parse_date(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d")


def safe_call_transform(client: OpenAI, model: str, item: QuestionItem) -> str:
    try:
        return transform_question(client, model, item)
    except Exception:
        return item.question


def safe_call_answer(client: OpenAI, model: str, question_text: str, options: Dict[str, str]) -> str:
    try:
        return ask_model_mcq(client, model, question_text, options)
    except Exception:
        return "ERROR"


def compute_summary(pred_rows: List[Dict], cutoff: datetime) -> List[Dict]:
    by_model_variant = {}
    for row in pred_rows:
        key = (row["model"], row["variant"])
        if key not in by_model_variant:
            by_model_variant[key] = {
                "pre_total": 0,
                "pre_correct": 0,
                "post_total": 0,
                "post_correct": 0,
                "errors": 0,
            }
        bucket = by_model_variant[key]
        if row["prediction"] == "ERROR":
            bucket["errors"] += 1
            continue

        is_pre = parse_date(row["date"]) < cutoff
        bucket_key_total = "pre_total" if is_pre else "post_total"
        bucket_key_correct = "pre_correct" if is_pre else "post_correct"

        bucket[bucket_key_total] += 1
        if row["is_correct"]:
            bucket[bucket_key_correct] += 1

    summary_rows = []
    for (model, variant), stats in by_model_variant.items():
        pre_acc = (100.0 * stats["pre_correct"] / stats["pre_total"]) if stats["pre_total"] else None
        post_acc = (100.0 * stats["post_correct"] / stats["post_total"]) if stats["post_total"] else None
        decay = (pre_acc - post_acc) if (pre_acc is not None and post_acc is not None) else None

        summary_rows.append(
            {
                "model": model,
                "variant": variant,
                "pre_total": stats["pre_total"],
                "pre_correct": stats["pre_correct"],
                "pre_accuracy": pre_acc,
                "post_total": stats["post_total"],
                "post_correct": stats["post_correct"],
                "post_accuracy": post_acc,
                "decay_pre_minus_post": decay,
                "errors": stats["errors"],
            }
        )

    summary_rows.sort(key=lambda x: (x["model"], x["variant"]))
    return summary_rows


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.strptime(args.cutoff_date, "%Y-%m-%d")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    client = OpenAI(api_key=api_key)
    items = load_dataset(dataset_path, max_questions=args.max_questions)

    transformed_map = transform_questions_batched(
        client=client,
        model=args.transform_model,
        items=items,
        batch_size=max(1, args.batch_size),
    )

    transformed = []
    for item in items:
        transformed_question = transformed_map.get(item.id, item.question)
        transformed.append(
            {
                "id": item.id,
                "date": item.date,
                "topic": item.topic,
                "question_original": item.question,
                "question_transformed": transformed_question,
                "options": item.options,
                "answer": item.answer,
            }
        )

    transformed_path = output_dir / "transformed_questions.json"
    with transformed_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2)

    pred_rows = []
    variants = [
        ("question_original", "original"),
        ("question_transformed", "transformed"),
    ]
    for model in models:
        for variant_key, variant_name in variants:
            batch_rows = [
                {
                    "id": row["id"],
                    "question_text": row[variant_key],
                    "options": row["options"],
                }
                for row in transformed
            ]
            answer_map = ask_model_mcq_batched(
                client=client,
                model=model,
                rows=batch_rows,
                batch_size=max(1, args.batch_size),
            )

            for row in transformed:
                pred = answer_map.get(row["id"], "ERROR")
                pred_rows.append(
                    {
                        "model": model,
                        "variant": variant_name,
                        "id": row["id"],
                        "date": row["date"],
                        "topic": row["topic"],
                        "answer": row["answer"],
                        "prediction": pred,
                        "is_correct": pred == row["answer"],
                    }
                )

    predictions_path = output_dir / "predictions.csv"
    save_csv(
        predictions_path,
        pred_rows,
        fieldnames=[
            "model",
            "variant",
            "id",
            "date",
            "topic",
            "answer",
            "prediction",
            "is_correct",
        ],
    )

    summary_rows = compute_summary(pred_rows, cutoff)
    summary_csv_path = output_dir / "summary_by_model.csv"
    save_csv(
        summary_csv_path,
        summary_rows,
        fieldnames=[
            "model",
            "variant",
            "pre_total",
            "pre_correct",
            "pre_accuracy",
            "post_total",
            "post_correct",
            "post_accuracy",
            "decay_pre_minus_post",
            "errors",
        ],
    )

    summary_json_path = output_dir / "summary_by_model.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cutoff_date": args.cutoff_date,
                "models": models,
                "dataset_size": len(items),
                "summary": summary_rows,
            },
            f,
            indent=2,
        )

    print("=" * 72)
    print("News temporal experiment finished")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Cutoff date: {args.cutoff_date}")
    print(f"Questions: {len(items)}")
    print(f"Models: {', '.join(models)}")
    print(f"Saved: {transformed_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_json_path}")
    print()

    for row in summary_rows:
        pre = f"{row['pre_accuracy']:.1f}%" if row["pre_accuracy"] is not None else "N/A"
        post = f"{row['post_accuracy']:.1f}%" if row["post_accuracy"] is not None else "N/A"
        decay = f"{row['decay_pre_minus_post']:.1f}" if row["decay_pre_minus_post"] is not None else "N/A"
        print(
            f"{row['model']:16s} | {row['variant']:11s} | "
            f"pre={pre:6s} post={post:6s} decay={decay:>5s} | errors={row['errors']}"
        )


if __name__ == "__main__":
    main()
