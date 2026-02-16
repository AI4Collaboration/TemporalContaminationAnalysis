#!/usr/bin/env python3
import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import requests


USER_AGENT = "Mozilla/5.0 (TemporalContaminationAnalysis/1.0)"
GENERIC_ENTITY_BLACKLIST = {
    "war",
    "government",
    "president",
    "prime minister",
    "city",
    "country",
    "state",
    "people",
    "police",
    "military",
}


@dataclass
class EventExample:
    event_date: str
    clue_text: str
    answer_entity: str
    topic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dated news MCQ dataset from Wikipedia Current Events")
    parser.add_argument("--start-date", default="2023-05-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2024-06-30", help="YYYY-MM-DD")
    parser.add_argument("--target-size", type=int, default=220, help="Target number of questions")
    parser.add_argument(
        "--output",
        default="ValidationExp3-WikiBasedQA/data/news_mcq_dataset_200plus.json",
        help="Output dataset path",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_raw_day_page(day: date) -> Optional[str]:
    title = f"Portal:Current events/{day.year}_{day.strftime('%B')}_{day.day}"
    url = f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}&action=raw"

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if response.status_code != 200:
            return None
        text = response.text
        if "{{Current events" not in text:
            return None
        return text
    except Exception:
        return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_references(text: str) -> str:
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


def wikilink_to_display(text: str) -> str:
    def repl(match):
        inner = match.group(1)
        if "|" in inner:
            return inner.split("|")[-1]
        return inner

    return re.sub(r"\[\[([^\]]+)\]\]", repl, text)


def extract_link_candidates(raw_line: str) -> List[str]:
    candidates = []
    for inner in re.findall(r"\[\[([^\]]+)\]\]", raw_line):
        display = inner.split("|")[-1].strip()
        display = re.sub(r"\s+", " ", display)
        if len(display) < 3:
            continue
        if display.lower() in GENERIC_ENTITY_BLACKLIST:
            continue
        if re.match(r"^[0-9\W_]+$", display):
            continue
        candidates.append(display)
    return candidates


def category_from_line(raw_line: str) -> str:
    l = raw_line.lower()
    if "election" in l or "parliament" in l or "minister" in l or "president" in l:
        return "politics"
    if "earthquake" in l or "flood" in l or "wildfire" in l:
        return "disaster"
    if "stock" in l or "bank" in l or "econom" in l or "inflation" in l:
        return "economy"
    if "football" in l or "olympic" in l or "championship" in l:
        return "sports"
    if "space" in l or "nasa" in l or "launch" in l or "science" in l:
        return "science"
    return "world-news"


def mask_first_entity(text: str, entity: str) -> str:
    escaped = re.escape(entity)
    masked, count = re.subn(escaped, "[BLANK]", text, count=1)
    if count == 0:
        return text
    return masked


def extract_examples_from_raw(day: date, raw_text: str) -> List[EventExample]:
    examples = []
    lines = raw_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("**"):
            continue
        if stripped.startswith("***"):
            pass

        content = stripped.lstrip("*").strip()
        if len(content) < 40:
            continue

        candidates = extract_link_candidates(content)
        if not candidates:
            continue

        answer = candidates[0]
        cleaned = wikilink_to_display(strip_references(content))
        cleaned = normalize_whitespace(cleaned)
        if len(cleaned) < 35:
            continue

        clue = mask_first_entity(cleaned, answer)
        if clue == cleaned:
            continue
        if "[BLANK]" not in clue:
            continue

        examples.append(
            EventExample(
                event_date=day.strftime("%Y-%m-%d"),
                clue_text=clue,
                answer_entity=answer,
                topic=category_from_line(content),
            )
        )

    return examples


def to_mcq_dataset(examples: List[EventExample], target_size: int, seed: int) -> List[dict]:
    random.seed(seed)

    if len(examples) < target_size:
        raise ValueError(f"Not enough extracted examples ({len(examples)}) for target size {target_size}")

    random.shuffle(examples)
    selected = examples[: target_size * 2]

    pool_entities = [ex.answer_entity for ex in selected]
    unique_entities = list(dict.fromkeys(pool_entities))

    dataset = []
    letters = ["A", "B", "C", "D"]

    for idx, ex in enumerate(selected):
        distractor_pool = [e for e in unique_entities if e != ex.answer_entity and e.lower() != ex.answer_entity.lower()]
        if len(distractor_pool) < 3:
            continue

        distractors = random.sample(distractor_pool, 3)
        option_values = distractors + [ex.answer_entity]
        random.shuffle(option_values)

        options = {letter: option_values[i] for i, letter in enumerate(letters)}
        answer_letter = next(letter for letter in letters if options[letter] == ex.answer_entity)

        question = (
            f"According to news reports on {ex.event_date}, which entity best fills the blank in this statement?\n"
            f"{ex.clue_text}"
        )

        dataset.append(
            {
                "id": f"wiki-news-{idx+1:04d}",
                "date": ex.event_date,
                "question": question,
                "options": options,
                "answer": answer_letter,
                "topic": ex.topic,
            }
        )

        if len(dataset) >= target_size:
            break

    if len(dataset) < target_size:
        raise ValueError(f"Could only build {len(dataset)} questions; target was {target_size}")

    dataset.sort(key=lambda x: x["date"])
    return dataset


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    all_examples: List[EventExample] = []
    fetched_days = 0

    for day in iter_dates(start, end):
        raw = fetch_raw_day_page(day)
        if raw is None:
            continue
        fetched_days += 1
        all_examples.extend(extract_examples_from_raw(day, raw))

    dataset = to_mcq_dataset(all_examples, args.target_size, args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("=" * 72)
    print("Wikipedia news dataset build complete")
    print("=" * 72)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Days fetched: {fetched_days}")
    print(f"Raw extracted examples: {len(all_examples)}")
    print(f"Final dataset size: {len(dataset)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
