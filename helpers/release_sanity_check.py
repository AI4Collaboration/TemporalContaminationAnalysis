#!/usr/bin/env python3
"""Quick OSS release sanity checks for this repository.

Checks include:
- Potential hardcoded secrets in tracked text files
- Tracked files that look like local/editor residue
- Large tracked files that may be accidental artifacts
- Inventory of tracked data files and basic PII pattern scan
"""

from __future__ import annotations

import re
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Generic api_key assignment": re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*[\"'][^\"']{12,}[\"']"
    ),
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".cfg",
    ".ini",
    ".env",
    ".lock",
}

DATA_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".xlsx",
    ".parquet",
    ".pkl",
}

RESIDUE_PATTERNS = (
    re.compile(r"^\.vscode/"),
    re.compile(r"^\.idea/"),
    re.compile(r"\.DS_Store$"),
    re.compile(r"^output/"),
    re.compile(r"^results/"),
)

PII_PATTERNS = {
    "Email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "US phone": re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b"),
    "US SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    return path.name in {".env", "Dockerfile", "Makefile"}


def scan_secrets(paths: list[Path]) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for path in paths:
        if not is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        rel = str(path.relative_to(ROOT))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append((rel, lineno, label))
    return findings


def scan_residue(paths: list[Path]) -> list[str]:
    flagged: list[str] = []
    for path in paths:
        rel = str(path.relative_to(ROOT))
        if any(p.search(rel) for p in RESIDUE_PATTERNS):
            flagged.append(rel)
    return flagged


def scan_large_files(paths: list[Path], threshold_mb: int = 5) -> list[tuple[str, float]]:
    large: list[tuple[str, float]] = []
    threshold = threshold_mb * 1024 * 1024
    for path in paths:
        size = path.stat().st_size
        if size >= threshold:
            large.append((str(path.relative_to(ROOT)), size / (1024 * 1024)))
    return sorted(large, key=lambda x: x[1], reverse=True)


def data_file_summary(path: Path) -> str:
    if path.suffix.lower() != ".json":
        return ""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "json parse failed"

    if isinstance(content, list):
        sample_keys = sorted(content[0].keys()) if content and isinstance(content[0], dict) else []
        return f"rows={len(content)}, sample_keys={sample_keys[:8]}"
    if isinstance(content, dict):
        return f"top_level_keys={sorted(content.keys())[:8]}"
    return f"type={type(content).__name__}"


def scan_data_files(paths: list[Path]) -> list[tuple[str, float, str]]:
    results: list[tuple[str, float, str]] = []
    for path in paths:
        if path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        summary = data_file_summary(path)
        results.append((str(path.relative_to(ROOT)), size_mb, summary))
    return sorted(results, key=lambda x: x[1], reverse=True)


def scan_pii_patterns(paths: list[Path]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for path in paths:
        if path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for label, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                hits.append((str(path.relative_to(ROOT)), label))
                break
    return hits


def main() -> int:
    files = git_tracked_files()

    secret_hits = scan_secrets(files)
    residue_hits = scan_residue(files)
    large_hits = scan_large_files(files)
    data_hits = scan_data_files(files)
    pii_hits = scan_pii_patterns(files)

    print("=== OSS Release Sanity Check ===")
    print(f"Tracked files: {len(files)}")

    if secret_hits:
        print("\n[FAIL] Potential hardcoded secrets:")
        for rel, lineno, label in secret_hits:
            print(f"  - {rel}:{lineno} ({label})")
    else:
        print("\n[OK] No obvious hardcoded secret patterns found.")

    if residue_hits:
        print("\n[WARN] Possibly local/editor residue tracked:")
        for rel in residue_hits:
            print(f"  - {rel}")
    else:
        print("\n[OK] No common local residue files are tracked.")

    if large_hits:
        print("\n[WARN] Large tracked files (>= 5 MB):")
        for rel, size_mb in large_hits:
            print(f"  - {rel}: {size_mb:.2f} MB")
    else:
        print("\n[OK] No large tracked files (>= 5 MB).")

    if data_hits:
        print("\n[INFO] Tracked data-like files:")
        for rel, size_mb, summary in data_hits:
            suffix = f" | {summary}" if summary else ""
            print(f"  - {rel}: {size_mb:.2f} MB{suffix}")
    else:
        print("\n[OK] No tracked data-like files detected.")

    if pii_hits:
        print("\n[WARN] Possible PII-like patterns found (manual review recommended):")
        for rel, label in pii_hits:
            print(f"  - {rel} ({label})")
    else:
        print("\n[OK] No obvious PII-like patterns detected.")

    if secret_hits:
        print("\nResult: FAILED (review and clean flagged secret findings)")
        return 1

    print("\nResult: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
