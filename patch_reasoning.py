#!/usr/bin/env python3
"""Patch two grounded reasoning strings in submission.csv without changing ranks or scores."""

import csv


SUBMISSION_PATH = "submission.csv"

PATCHES = {
    "CAND_0039754": (
        "Top-tier fit: 16yr over the 5-9yr band, but Apple role shipped hybrid "
        "BM25+dense semantic search (fine-tuned BGE-large, 35M items, NDCG@10 +18%, "
        "embedding drift monitoring); Meta role owns BGE-large → Pinecone → XGBoost "
        "LTR pipeline with A/B-calibrated offline eval; Observe.AI built "
        "recruiter-facing ranking serving 50M+ queries/mo. 30d notice, GitHub 77.5, "
        "RRR 0.81. Exception justified per JD's own outlier clause."
    ),
    "CAND_0043860": (
        "Moderate fit: 6.1yr at Aganitha (titled Junior ML by company convention); "
        "Aganitha work is collaborative filtering + gradient-boosted re-ranking over "
        "engagement signals — production but lighter than FAANG-scale retrieval. "
        "Concern: Nykaa role was primarily CV (ResNet image moderation); candidate "
        "self-describes NLP/LLM as a transition in progress. 30d notice, RRR 0.81. "
        "Ranked here on retrieval adjacency; NLP depth unconfirmed."
    ),
}


def patch_submission(path: str, patches: dict[str, str]) -> None:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames != ["candidate_id", "rank", "score", "reasoning"]:
        raise ValueError(f"Unexpected header: {fieldnames}")

    original_rows = [row.copy() for row in rows]
    patch_log: list[str] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in patches:
            row["reasoning"] = patches[candidate_id]
            patch_log.append(f"{candidate_id} (rank {row['rank']})")

    missing = set(patches) - {row["candidate_id"] for row in rows}
    if missing:
        raise ValueError(f"Patch candidate(s) not found: {sorted(missing)}")

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    changed_ids = set(patches)
    for before, after in zip(original_rows, rows):
        candidate_id = before["candidate_id"]
        if candidate_id in changed_ids:
            if (
                before["rank"] != after["rank"]
                or before["score"] != after["score"]
                or before["candidate_id"] != after["candidate_id"]
            ):
                raise RuntimeError(f"Unexpected rank/score/id change for {candidate_id}")
        elif before != after:
            raise RuntimeError(f"Unexpected change outside requested patches: {candidate_id}")

    if len(patch_log) != len(patches):
        raise RuntimeError(f"Expected {len(patches)} patches, applied {len(patch_log)}")


if __name__ == "__main__":
    patch_submission(SUBMISSION_PATH, PATCHES)
