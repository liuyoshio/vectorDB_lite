#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Building..."
cmake -S . -B build
cmake --build build

echo "Generating demo data..."
python3 - <<'PY'
import struct, random

def write_bin(path, count, dim, seed):
    random.seed(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", count, dim))
        for _ in range(count * dim):
            f.write(struct.pack("<f", random.random()))

write_bin("data.bin", 2000, 32, 1)
write_bin("queries.bin", 5, 32, 2)
print("Wrote data.bin and queries.bin")
PY

echo "Running IVF search..."
./build/vdb_cli search --index ivf --data data.bin --queries queries.bin --k 5 --nprobe 8

echo "Running HNSW search..."
./build/vdb_cli search --index hnsw --data data.bin --queries queries.bin --k 5 --ef-search 64

echo "Done."
