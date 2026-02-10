# vector_db

Lightweight vector database with IVF and HNSW indexing, static build, and batched query search.

## Build (CPU)

```bash
cmake -S . -B build
cmake --build build
```

## Build (CUDA)

```bash
cmake -S . -B build -DVDB_ENABLE_CUDA=ON
cmake --build build
```

## CLI usage

```bash
./build/vdb_cli search --index ivf --data data.bin --queries queries.bin --k 10 --nprobe 8
./build/vdb_cli search --index hnsw --data data.bin --queries queries.bin --k 10 --ef-search 64
```

Write results as CSV (qi,rank,id,dist):

```bash
./build/vdb_cli search --index ivf --data data.bin --queries queries.bin --k 10 --out results.csv
```

### Binary format

Both `--data` and `--queries` use the same simple binary format:

```
[uint32 count][uint32 dim][count*dim float32 values]  (row-major)
```

## API overview

- `vdb::IvfIndex` with `IvfConfig` for `nlist` and `kmeans_iters`.
- `vdb::HnswIndex` with `HnswConfig` for `M`, `ef_construction`, `ef_search`.
- `vdb::SearchParams` for `nprobe`, `ef_search`, `use_gpu`.

See `src/main.cpp` for a complete minimal example.
