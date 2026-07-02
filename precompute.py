#!/usr/bin/env python3
"""
Build offline artifacts for rank.py.

Usage:
    python precompute.py --candidates ./candidates.jsonl

This step is intentionally allowed to be slow. The ranking step reads these
artifacts and performs no network access.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import bm25s
import numpy as np
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
MODELS = BASE_DIR / "models"
EMBEDDINGS_PATH = ARTIFACTS / "candidate_embeddings.npy"
EMBEDDINGS_PROGRESS_PATH = ARTIFACTS / "candidate_embeddings_progress.json"
EXPECTED_EMBEDDING_DIM = 384


def build_candidate_text(candidate: dict) -> str:
    """Text used for sparse and dense retrieval."""
    profile = candidate["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
    ]

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("industry", ""))
        parts.append(job.get("description", ""))

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    return " ".join(part for part in parts if part).strip()


def resolve_embed_model_path() -> str:
    local = MODELS / "bge_small"
    if local.exists():
        return str(local)
    legacy = MODELS / "bge-small-en-v1.5"
    if legacy.exists():
        return str(legacy)
    return "BAAI/bge-small-en-v1.5"


def parse_candidates(path: Path) -> tuple[dict[str, dict], list[str], list[str]]:
    candidates: dict[str, dict] = {}
    ids_ordered: list[str] = []
    corpus_texts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            candidate_id = candidate["candidate_id"]
            candidates[candidate_id] = candidate
            ids_ordered.append(candidate_id)
            corpus_texts.append(build_candidate_text(candidate))

    return candidates, ids_ordered, corpus_texts


def load_embedding_progress(total: int) -> int:
    if not EMBEDDINGS_PROGRESS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return 0
    try:
        progress = json.loads(EMBEDDINGS_PROGRESS_PATH.read_text(encoding="utf-8"))
        rows_written = int(progress.get("rows_written", 0))
    except (ValueError, TypeError, OSError):
        return 0
    if rows_written < 0 or rows_written > total:
        return 0
    return rows_written


def write_embedding_progress(rows_written: int, total: int) -> None:
    EMBEDDINGS_PROGRESS_PATH.write_text(
        json.dumps(
            {
                "rows_written": rows_written,
                "total": total,
                "embedding_dim": EXPECTED_EMBEDDING_DIM,
                "complete": rows_written == total,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def existing_embeddings_complete(total: int) -> bool:
    if not EMBEDDINGS_PATH.exists():
        return False
    try:
        embeddings = np.load(str(EMBEDDINGS_PATH), mmap_mode="r")
        if embeddings.shape != (total, EXPECTED_EMBEDDING_DIM):
            return False
    except Exception:
        return False
    if not EMBEDDINGS_PROGRESS_PATH.exists():
        return True
    try:
        progress = json.loads(EMBEDDINGS_PROGRESS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return bool(progress.get("complete")) and int(progress.get("rows_written", -1)) == total


def encode_embeddings_resumable(
    corpus_texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
    encode_batch_size: int,
) -> None:
    total = len(corpus_texts)
    if existing_embeddings_complete(total):
        print(f"  Existing embeddings found: shape ({total}, {EXPECTED_EMBEDDING_DIM}); skipping")
        if EMBEDDINGS_PROGRESS_PATH.exists():
            EMBEDDINGS_PROGRESS_PATH.unlink()
        return

    rows_written = load_embedding_progress(total)
    if rows_written and rows_written % batch_size != 0:
        rows_written = (rows_written // batch_size) * batch_size

    mode = "r+" if EMBEDDINGS_PATH.exists() and rows_written > 0 else "w+"
    embeddings = open_memmap(
        str(EMBEDDINGS_PATH),
        mode=mode,
        dtype=np.float32,
        shape=(total, EXPECTED_EMBEDDING_DIM),
    )

    if rows_written:
        print(f"  Resuming embeddings at row {rows_written:,}/{total:,}")

    next_report = max(10000, ((rows_written // 10000) + 1) * 10000)
    for start in range(rows_written, total, batch_size):
        end = min(start + batch_size, total)
        batch = corpus_texts[start:end]
        batch_embeddings = model.encode(
            batch,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
        if batch_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_EMBEDDING_DIM}-dim embeddings, got {batch_embeddings.shape[1]}"
            )
        embeddings[start:end] = batch_embeddings
        embeddings.flush()
        write_embedding_progress(end, total)

        if end >= next_report or end == total:
            print(f"  {end:,}/{total:,} encoded...")
            while next_report <= end:
                next_report += 10000

    if EMBEDDINGS_PROGRESS_PATH.exists():
        EMBEDDINGS_PROGRESS_PATH.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(BASE_DIR / "candidates.jsonl"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    args = parser.parse_args()

    candidates_path = Path(args.candidates).resolve()
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate file not found: {candidates_path}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Step 1: Parsing candidates...")
    candidates, ids_ordered, corpus_texts = parse_candidates(candidates_path)
    print(f"  Loaded {len(candidates):,} candidates")

    with (ARTIFACTS / "candidates_parsed.pkl").open("wb") as handle:
        pickle.dump(candidates, handle, protocol=5)
    with (ARTIFACTS / "candidate_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(ids_ordered, handle)
    print("  Saved candidates_parsed.pkl and candidate_ids.json")

    print("\nStep 2: Building BM25S index...")
    if (ARTIFACTS / "bm25s_index").exists():
        print(f"  Existing BM25S index found at {ARTIFACTS / 'bm25s_index'}; skipping")
    else:
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(str(ARTIFACTS / "bm25s_index"))
        print(f"  BM25S index saved to {ARTIFACTS / 'bm25s_index'}")

    print("\nStep 3: Encoding candidates with bge-small-en-v1.5...")
    model = SentenceTransformer(
        resolve_embed_model_path(),
        cache_folder=str(MODELS / "hf_cache"),
        local_files_only=(MODELS / "bge_small").exists(),
    )

    encode_embeddings_resumable(corpus_texts, model, args.batch_size, args.encode_batch_size)
    candidate_embeddings = np.load(str(EMBEDDINGS_PATH), mmap_mode="r")
    print(f"  Embeddings saved: shape {candidate_embeddings.shape}")
    print("\nPre-computation complete.")


if __name__ == "__main__":
    main()
