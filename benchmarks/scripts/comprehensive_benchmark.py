#!/usr/bin/env python3
"""Comprehensive benchmark for vector_db – OpenMP-enabled Release build.

6 Scenarios
-----------
1  Thread Scalability   OpenMP speedup curves (1 → 8 threads)
2  Recall-QPS Tradeoff  Pareto frontier  (IVF nprobe / HNSW ef_search sweep)
3  Dataset-Size Scaling QPS & build time vs N
4  Dimensionality       QPS & recall vs embedding dimension D
5  IVF Parameter Grid   nlist × nprobe heat-map (QPS + Recall)
6  HNSW Parameter Grid  M × ef_search heat-map  (QPS + Recall)

Results → benchmarks/results/comprehensive_results.json
"""

import csv, json, os, struct, subprocess, sys
from pathlib import Path
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
# __file__ lives in  benchmarks/scripts/  →  .parent.parent = benchmarks/
BENCH   = Path(__file__).parent.parent.resolve()
ROOT    = BENCH.parent                               # project root
BUILD   = ROOT / "build_release"
CLI     = BUILD / "vdb_cli"
TMP     = BENCH / "data" / "comprehensive"
RESULTS = BENCH / "results" / "comprehensive_results.json"
TMP.mkdir(parents=True, exist_ok=True)
(BENCH / "results").mkdir(parents=True, exist_ok=True)

# ── binary I/O ────────────────────────────────────────────────────────────────
def write_bin(path, arr):
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(np.ascontiguousarray(arr.astype(np.float32)).tobytes())

# ── ground truth ──────────────────────────────────────────────────────────────
def exact_topk(data, queries, k):
    nq  = queries.shape[0]
    out = np.empty((nq, k), dtype=np.int64)
    for i in range(nq):
        diff  = data - queries[i]
        dists = np.einsum("nd,nd->n", diff, diff)
        kk = min(k, dists.shape[0])
        if kk <= 0:
            out[i] = np.array([], dtype=np.int64)
            continue
        nn    = np.argpartition(dists, kk - 1)[:kk]
        nn    = nn[np.argsort(dists[nn])]
        out[i] = nn
    return out.tolist()

def recall_at_k(pred, gt, k):
    if not pred or not gt:
        return 0.0
    scores = [len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)]
    return float(np.mean(scores))

# ── CLI wrapper ───────────────────────────────────────────────────────────────
_TMP_CSV = TMP / "_result.csv"

def _index_flags(index, **kw):
    f = []
    if index == "ivf":
        if "nlist"  in kw: f += ["--nlist",  str(kw["nlist"])]
        if "nprobe" in kw: f += ["--nprobe", str(kw["nprobe"])]
    else:
        if "M"               in kw: f += ["--M",               str(kw["M"])]
        if "ef_construction" in kw: f += ["--ef-construction", str(kw["ef_construction"])]
        if "ef_search"       in kw: f += ["--ef-search",       str(kw["ef_search"])]
    return f

def run_search(index, data_p, queries_p, k, threads=8, **kw):
    """Run vdb_cli search --out csv; return pred_ids list[list[int]]."""
    _TMP_CSV.unlink(missing_ok=True)
    cmd = ([str(CLI), "search",
            "--index",   index,
            "--data",    str(data_p),
            "--queries", str(queries_p),
            "--k",       str(k),
            "--threads", str(threads),
            "--out",     str(_TMP_CSV)]
           + _index_flags(index, **kw))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] search {index} {kw}"); return []
    except Exception as e:
        print(f"    [ERROR]   search {e}");          return []
    if r.returncode != 0:
        print(f"    [WARN] search failed: {r.stderr[:200]}")
        return []
    if not _TMP_CSV.exists() or _TMP_CSV.stat().st_size == 0:
        print("    [WARN] search CSV missing/empty")
        return []

    pred = {}
    with open(_TMP_CSV) as f:
        for row in csv.reader(f):
            if len(row) < 3: continue
            qi, rank, idx = int(row[0]), int(row[1]), int(row[2])
            pred.setdefault(qi, []).append((rank, idx))
    nq = max(pred) + 1 if pred else 0
    ordered = []
    for qi in range(nq):
        items = pred.get(qi, [])
        items.sort(key=lambda x: x[0])
        ordered.append([idx for _, idx in items])
    return ordered

def run_bench(index, data_p, queries_p, k, threads=8, repeat=3, warmup=1,
              get_recall=False, **kw):
    """Run vdb_cli bench --out csv; return dict with METRIC values.
    If get_recall=True, also includes 'pred_ids': list[list[int]]."""
    _TMP_CSV.unlink(missing_ok=True)
    cmd = ([str(CLI), "bench",
            "--index",   index,
            "--data",    str(data_p),
            "--queries", str(queries_p),
            "--k",       str(k),
            "--threads", str(threads),
            "--repeat",  str(repeat),
            "--warmup",  str(warmup),
            "--out",     str(_TMP_CSV)]
           + _index_flags(index, **kw))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {index} {kw}"); return {}
    except Exception as e:
        print(f"    [ERROR]   {e}");          return {}

    metrics = {}
    for line in r.stdout.splitlines():
        if line.startswith("METRIC"):
            parts = line.split()
            if len(parts) == 3:
                metrics[parts[1]] = float(parts[2])
    if not metrics:
        print(f"    [WARN] no METRIC lines.  stderr: {r.stderr[:200]}")
        return {}

    if get_recall:
        metrics["pred_ids"] = run_search(index, data_p, queries_p, k, threads=threads, **kw)
    return metrics

# ── data cache ────────────────────────────────────────────────────────────────
_cache = {}

def gen_data(n, d, nq=200, seed=42):
    key = (n, d, nq, seed)
    if key not in _cache:
        rng  = np.random.default_rng(seed)
        data = rng.random((n,  d), dtype=np.float32)
        qs   = rng.random((nq, d), dtype=np.float32)
        dp   = TMP / f"data_{n}_{d}.bin"
        # include N in query filename to avoid cross-scenario overwrite
        qp   = TMP / f"queries_{nq}_{d}_n{n}.bin"
        write_bin(dp, data); write_bin(qp, qs)
        _cache[key] = (dp, qp, data, qs)
    return _cache[key]

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1 – Thread Scalability
# ══════════════════════════════════════════════════════════════════════════════
def s1_thread_scalability(results):
    print("\n[1/6] Thread Scalability  (N=20k, D=64, k=10)")
    N, D, K = 20_000, 64, 10
    dp, qp, _, _ = gen_data(N, D)
    thread_counts = [1, 2, 4, 8]
    ivf_qps, hnsw_qps = [], []
    for t in thread_counts:
        print(f"  threads={t}")
        mi = run_bench("ivf",  dp, qp, K, threads=t, nlist=256, nprobe=16)
        mh = run_bench("hnsw", dp, qp, K, threads=t, M=16, ef_search=64)
        ivf_qps.append( mi.get("qps", float("nan")))
        hnsw_qps.append(mh.get("qps", float("nan")))
        print(f"    IVF={ivf_qps[-1]:>10,.0f} QPS   HNSW={hnsw_qps[-1]:>10,.0f} QPS")
    results["thread_scalability"] = {
        "threads": thread_counts, "ivf_qps": ivf_qps, "hnsw_qps": hnsw_qps}

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2 – Recall-QPS Pareto Frontier
# ══════════════════════════════════════════════════════════════════════════════
def s2_recall_qps(results):
    print("\n[2/6] Recall-QPS Tradeoff  (N=20k, D=64, k=10)")
    N, D, K = 20_000, 64, 10
    dp, qp, data, queries = gen_data(N, D)
    print("  Computing ground truth …")
    gt = exact_topk(data, queries, K)

    ivf_pts = []
    for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        print(f"  IVF nlist=256 nprobe={nprobe}")
        m = run_bench("ivf", dp, qp, K, nlist=256, nprobe=nprobe, get_recall=True)
        if m and "pred_ids" in m:
            r = recall_at_k(m["pred_ids"], gt, K)
            ivf_pts.append({"nprobe": nprobe, "recall": r, "qps": m.get("qps", 0)})
            print(f"    recall@{K}={r:.3f}  qps={m.get('qps',0):,.0f}")

    hnsw_pts = []
    for ef in [8, 16, 32, 64, 128, 256, 512]:
        print(f"  HNSW M=16 ef_search={ef}")
        m = run_bench("hnsw", dp, qp, K, M=16, ef_search=ef, get_recall=True)
        if m and "pred_ids" in m:
            r = recall_at_k(m["pred_ids"], gt, K)
            hnsw_pts.append({"ef_search": ef, "recall": r, "qps": m.get("qps", 0)})
            print(f"    recall@{K}={r:.3f}  qps={m.get('qps',0):,.0f}")

    results["recall_qps"] = {"ivf": ivf_pts, "hnsw": hnsw_pts}

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3 – Dataset-Size Scaling
# ══════════════════════════════════════════════════════════════════════════════
def s3_size_scaling(results):
    print("\n[3/6] Dataset-Size Scaling  (D=64, k=10, 8 threads)")
    D, K = 64, 10
    sizes = [1_000, 5_000, 10_000, 50_000]
    ivf_qps, hnsw_qps, ivf_build, hnsw_build = [], [], [], []
    for N in sizes:
        nlist = max(16, int(N ** 0.5))
        print(f"  N={N:,}  nlist={nlist}")
        dp, qp, _, _ = gen_data(N, D)
        mi = run_bench("ivf",  dp, qp, K, nlist=nlist, nprobe=8)
        mh = run_bench("hnsw", dp, qp, K, M=16, ef_search=64)
        ivf_qps.append(   mi.get("qps",      float("nan")))
        hnsw_qps.append(  mh.get("qps",      float("nan")))
        ivf_build.append( mi.get("build_sec", float("nan")))
        hnsw_build.append(mh.get("build_sec", float("nan")))
        print(f"    IVF  qps={ivf_qps[-1]:>10,.0f}  build={ivf_build[-1]:.3f}s")
        print(f"    HNSW qps={hnsw_qps[-1]:>10,.0f}  build={hnsw_build[-1]:.3f}s")
    results["size_scaling"] = {
        "sizes": sizes,
        "ivf_qps": ivf_qps,   "hnsw_qps":   hnsw_qps,
        "ivf_build": ivf_build, "hnsw_build": hnsw_build}

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 4 – Dimensionality Impact
# ══════════════════════════════════════════════════════════════════════════════
def s4_dim_impact(results):
    print("\n[4/6] Dimensionality Impact  (N=10k, k=10, 8 threads)")
    N, K = 10_000, 10
    dims = [16, 32, 64, 128, 256]
    ivf_qps, hnsw_qps, ivf_recall, hnsw_recall = [], [], [], []
    for D in dims:
        print(f"  D={D}")
        dp, qp, data, queries = gen_data(N, D)
        print("    Computing ground truth …")
        gt = exact_topk(data, queries, K)
        mi = run_bench("ivf",  dp, qp, K, nlist=128, nprobe=8, get_recall=True)
        mh = run_bench("hnsw", dp, qp, K, M=16, ef_search=64,  get_recall=True)
        ivf_qps.append(    mi.get("qps", float("nan")))
        hnsw_qps.append(   mh.get("qps", float("nan")))
        ivf_recall.append( recall_at_k(mi.get("pred_ids", []), gt, K)
                           if "pred_ids" in mi else float("nan"))
        hnsw_recall.append(recall_at_k(mh.get("pred_ids", []), gt, K)
                           if "pred_ids" in mh else float("nan"))
        print(f"    IVF  qps={ivf_qps[-1]:>10,.0f}  recall={ivf_recall[-1]:.3f}")
        print(f"    HNSW qps={hnsw_qps[-1]:>10,.0f}  recall={hnsw_recall[-1]:.3f}")
    results["dim_impact"] = {
        "dims": dims,
        "ivf_qps": ivf_qps,    "hnsw_qps":    hnsw_qps,
        "ivf_recall": ivf_recall, "hnsw_recall": hnsw_recall}

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 5 – IVF Parameter Heat-map
# ══════════════════════════════════════════════════════════════════════════════
def s5_ivf_heatmap(results):
    print("\n[5/6] IVF Parameter Heat-map  (N=20k, D=64, k=10)")
    N, D, K = 20_000, 64, 10
    dp, qp, data, queries = gen_data(N, D)
    print("  Computing ground truth …")
    gt = exact_topk(data, queries, K)
    nlists  = [64, 128, 256, 512]
    nprobes = [1, 2, 4, 8, 16, 32]
    qps_grid, recall_grid = [], []
    for nlist in nlists:
        qrow, rrow = [], []
        for nprobe in nprobes:
            np_ = min(nprobe, nlist)
            print(f"  nlist={nlist:4d}  nprobe={np_:2d}")
            m = run_bench("ivf", dp, qp, K, nlist=nlist, nprobe=np_, get_recall=True)
            qrow.append( m.get("qps", float("nan")))
            rrow.append( recall_at_k(m.get("pred_ids", []), gt, K)
                         if "pred_ids" in m else float("nan"))
        qps_grid.append(qrow); recall_grid.append(rrow)
    results["ivf_heatmap"] = {
        "nlists": nlists, "nprobes": nprobes,
        "qps": qps_grid,  "recall":  recall_grid}

# ══════════════════════════════════════════════════════════════════════════════
# Scenario 6 – HNSW Parameter Heat-map
# ══════════════════════════════════════════════════════════════════════════════
def s6_hnsw_heatmap(results):
    print("\n[6/6] HNSW Parameter Heat-map  (N=20k, D=64, k=10)")
    N, D, K = 20_000, 64, 10
    dp, qp, data, queries = gen_data(N, D)
    print("  Computing ground truth …")
    gt = exact_topk(data, queries, K)
    Ms  = [4, 8, 16, 32]
    efs = [16, 32, 64, 128, 256]
    qps_grid, recall_grid = [], []
    for M in Ms:
        qrow, rrow = [], []
        for ef in efs:
            print(f"  M={M:2d}  ef_search={ef:3d}")
            m = run_bench("hnsw", dp, qp, K, M=M, ef_search=ef, get_recall=True)
            qrow.append( m.get("qps", float("nan")))
            rrow.append( recall_at_k(m.get("pred_ids", []), gt, K)
                         if "pred_ids" in m else float("nan"))
        qps_grid.append(qrow); recall_grid.append(rrow)
    results["hnsw_heatmap"] = {
        "Ms": Ms, "efs": efs,
        "qps": qps_grid, "recall": recall_grid}

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not CLI.exists():
        sys.exit(f"Error: {CLI} not found. Please build the Release version of vector_db first.")
    print(f"Binary : {CLI}")
    print(f"Results: {RESULTS}")

    results = {}
    s1_thread_scalability(results)
    s2_recall_qps(results)
    s3_size_scaling(results)
    s4_dim_impact(results)
    s5_ivf_heatmap(results)
    s6_hnsw_heatmap(results)

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone – results saved to {RESULTS.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
