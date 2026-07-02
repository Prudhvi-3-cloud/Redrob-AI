#!/usr/bin/env bash
set -euo pipefail

mkdir -p models/bge_small models/bge_reranker artifacts

echo "Downloading bge-small-en-v1.5..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "BAAI/bge-small-en-v1.5",
    local_dir="models/bge_small",
    ignore_patterns=[".git*", "onnx/*"],
)
print("bge-small downloaded")
PY

echo "Downloading bge-reranker-v2-m3..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    local_dir="models/bge_reranker",
    ignore_patterns=[".git*", "assets/*", "*.gguf", "onnx/*"],
)
print("bge-reranker-v2-m3 downloaded")
PY

echo "Setup complete. Now run:"
echo "  python precompute.py --candidates ./candidates.jsonl"
echo "  python rank.py --candidates ./candidates.jsonl --job-description ./job_description.docx --cross-encoder-limit 30 --out ./submission.csv"
echo "  python patch_reasoning.py"
