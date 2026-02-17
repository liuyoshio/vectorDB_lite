# vector_db

Lightweight vector database for similarity search with:
- `IVF` index mode
- `HNSW` index mode
- Static index construction
- Batched query search

## Project layout

- `src/` core implementation + CLI
- `include/vdb/` public headers
- `benchmarks/` benchmark + plotting scripts
- `demo.sh` quick demo runner
- `PRESENTATION_CONTENT.md` presentation draft content

## Prerequisites

- CMake >= 3.18
- C++17 compiler
- Python 3 (for data generation and benchmarks)

Optional for benchmarking:
- `numpy` (required for benchmark script)
- `matplotlib` (for benchmark plots)
- `faiss-cpu` or `faiss-gpu` (for FAISS comparison)

## Quick start

### 1) Build (CPU)

```bash
cmake -S . -B build
cmake --build build
```

### 2) Generate sample data

```bash
python3 mock_data.py
```

This creates:
- `data.bin`
- `queries.bin`

Binary format:

```text
[uint32 count][uint32 dim][count*dim float32 values] (row-major)
```

### 3) Run search

```bash
./build/vdb_cli search --index ivf  --data data.bin --queries queries.bin --k 10 --nprobe 8
./build/vdb_cli search --index hnsw --data data.bin --queries queries.bin --k 10 --ef-search 64
```

Write output as CSV:

```bash
./build/vdb_cli search --index ivf --data data.bin --queries queries.bin --k 10 --out results.csv
```

### 4) Run full demo script

```bash
bash demo.sh
```

## Benchmarking

Run the benchmark pipeline (Debug + Release, VDB + optional FAISS):

```bash
bash benchmarks/run_bench.sh
```

Or run manually:

```bash
python3 benchmarks/benchmark.py \
  --root . \
  --n 20000 --dim 64 --nq 200 --k 10 \
  --runs 3 --threads 8 \
  --ivf-nlist 256 --ivf-nprobe 8 \
  --hnsw-m 16 --hnsw-ef-search 64 \
  --output benchmarks/results.json
```

Outputs:
- `benchmarks/results.json`
- `benchmarks/qps_search_comparison.png` (if matplotlib installed)
- `benchmarks/recall_comparison.png` (if matplotlib installed)
- `benchmarks/optimization_speedup.png` (if matplotlib installed)

## OpenMP notes (threaded query parallelism)

`--threads` is active only when OpenMP is found at configure time.

On macOS, if OpenMP is not detected by default, use Homebrew LLVM + libomp:

```bash
brew install llvm libomp

cmake -S . -B build_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
  -DOpenMP_ROOT="$(brew --prefix libomp)"

cmake --build build_release -j
```

## CLI summary

```bash
vdb_cli search --index ivf|hnsw --data <data.bin> --queries <queries.bin> --k <K> [options]
vdb_cli bench  --index ivf|hnsw --data <data.bin> --queries <queries.bin> --k <K> [options]
```

Common options:
- `--threads N`
- `--nlist`, `--nprobe`, `--kmeans-iters` (IVF)
- `--M`, `--ef-construction`, `--ef-search` (HNSW)
- `--out <path>`
