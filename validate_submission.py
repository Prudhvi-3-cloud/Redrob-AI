#!/usr/bin/env python3
"""Validate Redrob submission format plus local content sanity checks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
from collections import Counter
from pathlib import Path


TODAY = dt.date(2026, 6, 21)
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")
CONSULTING_SUBSTRINGS = [
    "tata consultancy",
    "ltimindtree",
    "tech mahindra",
    "hcl technologies",
    "hcl tech",
    "cognizant technology",
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mphasis",
    "hexaware",
    "birlasoft",
    "mindtree",
    "hcl",
]


def is_consulting(company_name: str) -> bool:
    name = company_name.strip().lower()
    return any(substring in name for substring in CONSULTING_SUBSTRINGS)


def is_skill_stuffer(candidate: dict) -> bool:
    expert_zero = sum(
        1
        for skill in candidate.get("skills", [])
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0
    )
    return expert_zero >= 3


def is_tenure_inflator(candidate: dict) -> bool:
    for job in candidate.get("career_history", []):
        try:
            start_date = dt.date.fromisoformat(job["start_date"])
            end_text = job.get("end_date")
            end_date = dt.date.fromisoformat(end_text) if end_text else TODAY
        except (KeyError, TypeError, ValueError):
            continue

        actual = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        claimed = job.get("duration_months", 0)
        if actual > 0:
            ratio = claimed / actual
            if abs(claimed - actual) > 12 and (ratio > 1.8 or ratio < 0.5):
                return True
    return False


def load_rows(path: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != REQUIRED_HEADER:
                errors.append(f"Header must be exactly {REQUIRED_HEADER}; found {reader.fieldnames}")
                return [], errors
            rows = [row for row in reader if any((value or "").strip() for value in row.values())]
    except OSError as exc:
        return [], [f"Cannot read submission: {exc}"]
    except UnicodeDecodeError:
        return [], ["Submission must be UTF-8 encoded"]
    return rows, errors


def load_candidates(path: Path, candidate_ids: set[str]) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            candidate_id = candidate["candidate_id"]
            if candidate_id in candidate_ids:
                candidates[candidate_id] = candidate
    return candidates


def validate_submission(submission_path: Path, candidates_path: Path) -> list[str]:
    rows, errors = load_rows(submission_path)
    if errors:
        return errors

    if len(rows) != 100:
        errors.append(f"Expected 100 data rows, got {len(rows)}")

    ranks: list[int] = []
    scores: list[float] = []
    candidate_ids: list[str] = []

    for index, row in enumerate(rows, start=2):
        candidate_id = (row.get("candidate_id") or "").strip()
        rank_text = (row.get("rank") or "").strip()
        score_text = (row.get("score") or "").strip()
        reasoning = (row.get("reasoning") or "").strip()

        if not CANDIDATE_ID_RE.match(candidate_id):
            errors.append(f"Row {index}: invalid candidate_id {candidate_id!r}")
        candidate_ids.append(candidate_id)

        try:
            rank = int(rank_text)
            ranks.append(rank)
        except ValueError:
            errors.append(f"Row {index}: rank must be an integer")

        try:
            score = float(score_text)
            scores.append(score)
            if not 0.0 <= score <= 1.0:
                errors.append(f"Row {index}: score out of [0,1]: {score}")
        except ValueError:
            errors.append(f"Row {index}: score must be a float")

        if not reasoning:
            errors.append(f"Row {index}: reasoning is empty")

    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be 1-100 each exactly once")
    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("Duplicate candidate_ids found")
    if len(scores) == 100 and any(scores[index] < scores[index + 1] for index in range(99)):
        errors.append("Scores must be non-increasing by rank")

    candidates = load_candidates(candidates_path, set(candidate_ids))
    missing_candidates = set(candidate_ids) - set(candidates)
    if missing_candidates:
        errors.append(f"Candidate IDs missing from candidates.jsonl: {sorted(missing_candidates)[:5]}")
        return errors

    for row in rows:
        candidate_id = row["candidate_id"]
        rank = int(row["rank"])
        score = float(row["score"])
        candidate = candidates[candidate_id]
        profile = candidate["profile"]
        signals = candidate.get("redrob_signals", {})
        history = candidate.get("career_history", [])

        yoe = float(profile.get("years_of_experience", 0) or 0)
        country = profile.get("country", "").strip().lower()
        willing = signals.get("willing_to_relocate", False)
        reasoning = row.get("reasoning", "")

        if is_skill_stuffer(candidate):
            errors.append(f"Honeypot rank={rank} {candidate_id}: skill stuffing")
        if is_tenure_inflator(candidate):
            errors.append(f"Honeypot rank={rank} {candidate_id}: tenure inflation")
        if country != "india" and not willing:
            errors.append(f"Location wall rank={rank} {candidate_id}: country={country}, willing={willing}")
        if yoe < 3.0:
            errors.append(f"Under YOE floor rank={rank} {candidate_id}: yoe={yoe}")
        if score > 1.0:
            errors.append(f"Score > 1.0 rank={rank} {candidate_id}: score={score}")

        companies = [job.get("company", "") for job in history] + [profile.get("current_company", "")]
        has_consulting = any(is_consulting(company) for company in companies)
        if has_consulting and "avoids the consulting-only" in reasoning:
            errors.append(f"Reasoning lie rank={rank} {candidate_id}: consulting profile says avoids consulting")

    lead_counts = Counter(row["reasoning"][:60] for row in rows)
    repeated_leads = {lead: count for lead, count in lead_counts.items() if count > 5}
    if repeated_leads:
        errors.append(f"Templated reasoning: {len(repeated_leads)} lead phrases repeated more than 5 times")

    return errors


def print_summary(submission_path: Path, candidates_path: Path) -> None:
    rows = list(csv.DictReader(submission_path.open("r", encoding="utf-8")))
    candidate_ids = {row["candidate_id"] for row in rows}
    candidates = load_candidates(candidates_path, candidate_ids)
    scores = [float(row["score"]) for row in rows]
    notice_values = [
        int(candidates[row["candidate_id"]]["redrob_signals"]["notice_period_days"])
        for row in rows
    ]
    yoe_values = [
        float(candidates[row["candidate_id"]]["profile"]["years_of_experience"])
        for row in rows
    ]
    countries = Counter(candidates[row["candidate_id"]]["profile"]["country"] for row in rows)

    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print("Notice period distribution:")
    for bucket_start in range(0, 150, 30):
        bucket_end = bucket_start + 30
        count = sum(1 for value in notice_values if bucket_start <= value < bucket_end)
        print(f"  {bucket_start}-{bucket_end}d: {count}")
    print(f"YOE: min={min(yoe_values):.1f} mean={statistics.mean(yoe_values):.1f} max={max(yoe_values):.1f}")
    print(f"Countries: {countries.most_common(5)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", help="Submission CSV path")
    parser.add_argument("--submission", dest="submission", help="Submission CSV path")
    parser.add_argument("--candidates", default="candidates.jsonl", help="candidates.jsonl path")
    args = parser.parse_args()

    csv_path = Path(args.submission or args.csv_path or "submission.csv")
    candidates_path = Path(args.candidates)

    issues = validate_submission(csv_path, candidates_path)
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
