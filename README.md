# Redrob Hack2Skill Candidate Ranker

This repository produces the top-100 candidate CSV for the Redrob AI Hack2Skill
candidate discovery challenge.

The ranking step is CPU-only and uses precomputed local artifacts:

- a BM25S index over candidate profile text
- normalized BGE-small candidate embeddings
- parsed candidate records
- local BGE-small and BGE reranker model snapshots

No hosted API is called by `rank.py`. The runtime path reads from `artifacts/`
and `models/`.

## Repository Layout

```text
.
|-- rank.py                    # Produces submission.csv from local artifacts
|-- precompute.py              # Builds artifacts from candidates.jsonl
|-- patch_reasoning.py          # Patches two audited reasoning strings
|-- validate_submission.py      # Local validator; silent on success
|-- candidates.jsonl            # Official 100,000-candidate pool
|-- job_description.docx        # Official job description
|-- submission.csv              # Current final CSV
|-- submission_metadata.yaml    # Portal/reproducibility metadata
|-- requirements.txt
|-- setup.sh                    # Optional model download helper for fresh setup
|-- artifacts/
|   |-- bm25s_index/
|   |-- candidate_embeddings.npy
|   |-- candidate_ids.json
|   `-- candidates_parsed.pkl
`-- models/
    |-- bge_small/
    `-- bge_reranker/
```

## Install

Use Python 3.10 or newer. On the tested Windows machine:

```powershell
cd "C:\Users\P.Prudhvi narayana\Music\Hack2Skill-Redrob AI\solution"
python -m pip install -r .\requirements.txt
```

The checked-in `models/` and `artifacts/` folders are enough for ranking. Run
`setup.sh` only when rebuilding a fresh environment that does not already have
the local model snapshots.

## Precomputation

Precomputation is already done for this release. To rebuild the artifacts from
the official candidate file:

```powershell
python .\precompute.py --candidates .\candidates.jsonl
```

This step may exceed the 5-minute ranking budget. The competition budget applies
to the ranking step that produces the CSV.

## Reproduce The Submission

Run these commands from the repository root:

```powershell
python .\rank.py --candidates .\candidates.jsonl --job-description .\job_description.docx --cross-encoder-limit 30 --out .\submission.csv
python .\patch_reasoning.py
python .\validate_submission.py --submission .\submission.csv
```

`patch_reasoning.py` and `validate_submission.py` are silent on success. If the
validator finds a problem, it prints the failure details and exits non-zero.

For a single shell command suitable for metadata:

```bash
python rank.py --candidates ./candidates.jsonl --job-description ./job_description.docx --cross-encoder-limit 30 --out ./submission.csv && python patch_reasoning.py
```

## What The Ranker Does

`rank.py` follows a bounded retrieval and scoring pipeline:

1. Load `job_description.docx` and build literal, behavioral, and rerank query text.
2. Load precomputed BM25S, dense embeddings, candidate ids, and parsed candidates.
3. Exclude detected honeypots before retrieval.
4. Retrieve a 600-candidate pool using BM25S and dense BGE-small search.
5. Fuse sparse and dense rankings with reciprocal-rank fusion.
6. Rescore the first 30 fused candidates with the local BGE reranker.
7. Apply JD evidence scoring: employer type, years of experience, production
   retrieval evidence, tenure, notice period, and location/relocation.
8. Apply a bounded behavioral-signal modifier from Redrob activity signals.
9. Normalize scores to the `[0.20, 0.99]` output range and write the top 100 rows.

The output CSV columns are exactly:

```text
candidate_id,rank,score,reasoning
```

## Offline Execution

The runtime model loads in `rank.py` are local:

- `models/bge_small`
- `models/bge_reranker`

Both are loaded with `local_files_only=True`. The ranking step does not require
network access after dependencies, artifacts, and local model snapshots are in
place.

## Expected Runtime

The release command uses `--cross-encoder-limit 30`. On the tested Windows
CPU-only machine, the full documented workflow completed in 128.3 seconds.
If a slower CPU is used for debugging, `--cross-encoder-limit 8` or `0` can be
used to reduce runtime, but the submitted reproduction command uses `30`.

## Validation

Run:

```powershell
python .\validate_submission.py --submission .\submission.csv
```

Success produces no output. The validator checks row count, header order,
rank uniqueness, score monotonicity, candidate IDs, honeypot exclusions,
location hard wall, YOE floor, and simple reasoning consistency checks.

## Sandbox

See `SANDBOX_INSTRUCTIONS.md` and `sandbox_demo.ipynb`. The notebook is prepared
for Google Colab or a similar hosted environment. It uses `TODO` placeholders
for the public repository URL because that URL must be created during final
submission.

## Limitations

- The cross-encoder is only applied to the first 30 fused candidates to stay
  within the CPU time budget.
- `precompute.py` is not part of the 5-minute ranking path.
- The reasoning patch is a small release-time correction for two manually
  audited candidates; it does not change ranks or scores.
- The repository still needs final portal-specific values in
  `submission_metadata.yaml`: team identity, GitHub URL, and sandbox URL.

## Troubleshooting

- If `rank.py` fails to load a model, check that `models/bge_small/` and
  `models/bge_reranker/` contain `config.json`, tokenizer files, and model
  weights.
- If validation prints nothing, the CSV passed local validation.
- If the official portal requires the CSV filename to be the registered team ID,
  rename `submission.csv` after validation; do not change the four CSV columns.
