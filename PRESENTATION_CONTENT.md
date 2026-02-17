# Lightweight Vector Database: Presentation Content

## Slide 1 - Title
**Lightweight Vector Database for Similarity Search**

- CPU-first ANN system with optional GPU direction
- Supports two core indexing strategies: IVF and HNSW
- Focus: static indexing + batched query throughput

---

## Slide 2 - Why This Project
**Problem Context**

- Embedding-based applications need fast nearest-neighbor search
- Teams often need a controllable, lightweight alternative for experimentation
- Our objective was to build and evaluate a practical ANN system end-to-end

---

## Slide 3 - Project Goals
**What we set out to deliver**

- Lightweight vector database implementation
- Two ANN modes: IVF and HNSW
- Static index build and batched query search
- Throughput-oriented behavior on CPU
- Benchmark comparison against FAISS

---

## Slide 4 - Solution Overview
**System at a high level**

- Input: vector dataset + query batch
- Index choice: IVF or HNSW
- Output: top-k nearest neighbors per query
- Interface supports batch queries to improve throughput
- CLI-based workflow for repeatable experiments

---

## Slide 5 - What We Evaluated
**Benchmark Setup**

- Dataset size: 20,000 vectors
- Dimension: 64
- Query batch: 200
- Top-k: 10
- Repeats: 3
- Thread setting: 8
- Compared:
- Our system in Debug and Release
- FAISS Flat, IVF, HNSW baselines

---

## Slide 6 - Main Performance Result
**Search Throughput (QPS, search-only)**

![QPS Comparison](/Users/yoshio/Desktop/vector_db/benchmarks/qps_search_comparison.png)

- Release build delivered large throughput gains vs Debug
- At the tested settings, our Release search throughput is competitive with FAISS IVF/HNSW
- Important context: throughput should be interpreted together with recall

---

## Slide 7 - Quality Result
**Recall@10 Comparison**

![Recall Comparison](/Users/yoshio/Desktop/vector_db/benchmarks/recall_comparison.png)

- IVF recall from our system is close to FAISS IVF in this configuration
- HNSW recall is lower than FAISS HNSW at the tested parameters
- Conclusion: speed is strong, but HNSW quality tuning remains a priority

---

## Slide 8 - Optimization Impact
**Debug vs Release**

![Optimization Speedup](/Users/yoshio/Desktop/vector_db/benchmarks/optimization_speedup.png)

- Compiler optimization had major impact on throughput
- This validates the optimization strategy as a key performance lever

---

## Slide 9 - Numeric Summary (Current Run)
**Key metrics from `results.json`**

| System | Mode | Search QPS | End-to-End QPS | Recall@10 |
|---|---:|---:|---:|---:|
| VDB Debug | IVF | 3,998.33 | 8.89 | 0.2705 |
| VDB Debug | HNSW | 940.88 | 6.12 | 0.5610 |
| VDB Release | IVF | 217,372.00 | 88.07 | 0.2705 |
| VDB Release | HNSW | 129,711.00 | 73.52 | 0.5610 |
| FAISS | Flat | 35,283.74 | 30,803.78 | 1.0000 |
| FAISS | IVF | 184,232.82 | 3,675.42 | 0.2795 |
| FAISS | HNSW | 120,536.40 | 1,459.18 | 0.7225 |

---

## Slide 10 - Interpretation
**What these numbers mean**

- Release optimization is non-negotiable for high performance
- Our IVF is close to FAISS IVF in recall at this operating point
- Our HNSW is faster at this setting but with lower recall than FAISS HNSW
- Therefore, this is not a "faster in all cases" claim
- Fair comparison should always be made at matched recall targets

---

## Slide 11 - Current Status
**Requirement coverage**

- CPU implementation: complete
- IVF + HNSW support: complete
- Static indexing: complete
- Batched query processing: complete
- Benchmarking and plots: complete
- GPU acceleration: planned/optional path, not the primary validated result in this run

---

## Slide 12 - Next Steps
**Roadmap to strengthen results**

- Tune for matched-recall comparisons (sweep `nprobe`, `ef_search`)
- Scale evaluation to larger datasets and multiple seeds
- Improve HNSW quality-performance balance
- Add persistence and production usability features
- Complete GPU benchmark track on CUDA-capable hardware

---

## Slide 13 - Closing
**Takeaway**

- We built a functional ANN vector DB with competitive search throughput
- Batched processing and optimization strategy are effective
- The strongest next value is quality tuning and matched-recall benchmarking
