#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT_DIR/benchmarks/benchmark.py" \
  --root "$ROOT_DIR" \
  --n 20000 --dim 64 --nq 200 --k 10 \
  --runs 3 --threads 8 \
  --ivf-nlist 256 --ivf-nprobe 8 \
  --hnsw-m 16 --hnsw-ef-search 64 \
  --output "$ROOT_DIR/benchmarks/results.json"
