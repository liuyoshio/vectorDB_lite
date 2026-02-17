#!/usr/bin/env python3
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
except Exception as e:
    print(
        "numpy is required for benchmarks. Install it first, e.g. `pip install numpy`.",
        file=sys.stderr,
    )
    raise


def write_bin_matrix(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    n, d = arr.shape
    with path.open("wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(arr.tobytes(order="C"))


def run_cmd(cmd, cwd: Path, check=True):
    return subprocess.run(cmd, cwd=str(cwd), check=check, capture_output=True, text=True)


def build_vdb(root: Path, build_dir: Path, build_type: str):
    run_cmd(["cmake", "-S", str(root), "-B", str(build_dir), f"-DCMAKE_BUILD_TYPE={build_type}"], root)
    run_cmd(["cmake", "--build", str(build_dir), "-j"], root)


def parse_vdb_stdout(stdout: str, nq: int, k: int) -> np.ndarray:
    ids = np.full((nq, k), -1, dtype=np.int64)
    line_re = re.compile(r"^query\s+(\d+):\s*(.*)$")
    item_re = re.compile(r"(\d+)\(([^\)]+)\)")

    for line in stdout.splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        qi = int(m.group(1))
        tail = m.group(2)
        cols = item_re.findall(tail)
        for j, (idx_s, _dist_s) in enumerate(cols[:k]):
            ids[qi, j] = int(idx_s)

    return ids


def parse_vdb_metrics(stdout: str):
    metrics = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("METRIC "):
            continue
        parts = stripped.split()
        if len(parts) != 3:
            continue
        _, key, val_s = parts
        try:
            metrics[key] = float(val_s)
        except ValueError:
            continue
    return metrics


def exact_topk(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    nq = xq.shape[0]
    out = np.empty((nq, k), dtype=np.int64)
    for i in range(nq):
        diff = xb - xq[i]
        dists = np.einsum("ij,ij->i", diff, diff)
        nn = np.argpartition(dists, k)[:k]
        nn = nn[np.argsort(dists[nn])]
        out[i] = nn
    return out


def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    total = pred.shape[0] * k
    hit = 0
    for i in range(pred.shape[0]):
        s = set(int(x) for x in truth[i, :k])
        for j in range(k):
            if int(pred[i, j]) in s:
                hit += 1
    return float(hit) / float(total) if total else 0.0


def bench_vdb(root: Path, build_dir: Path, index: str, data_bin: Path, query_bin: Path,
              nq: int, k: int, runs: int, threads: int, ivf_nlist: int,
              ivf_nprobe: int, hnsw_m: int, hnsw_ef_search: int):
    exe = build_dir / "vdb_cli"
    if not exe.exists():
        raise RuntimeError(f"vdb_cli not found: {exe}")

    cmd = [
        str(exe), "bench",
        "--index", index,
        "--data", str(data_bin),
        "--queries", str(query_bin),
        "--k", str(k),
        "--threads", str(threads),
        "--repeat", str(runs),
        "--warmup", "1",
    ]
    if index == "ivf":
        cmd += ["--nlist", str(ivf_nlist), "--nprobe", str(ivf_nprobe)]
    elif index == "hnsw":
        cmd += ["--M", str(hnsw_m), "--ef-search", str(hnsw_ef_search)]

    print(
        f"  [VDB] build={build_dir.name} index={index} threads={threads} runs={runs}",
        flush=True,
    )
    print(f"  [VDB] command: {' '.join(cmd)}", flush=True)

    t0 = time.perf_counter()
    proc = run_cmd(cmd, root, check=True)
    t1 = time.perf_counter()
    wall = t1 - t0

    metrics = parse_vdb_metrics(proc.stdout)
    if not metrics:
        raise RuntimeError(
            f"No VDB metrics parsed from output for {build_dir.name}/{index}. "
            "Expected lines like `METRIC build_sec ...`."
        )
    last_ids = parse_vdb_stdout(proc.stdout, nq=nq, k=k)
    stderr_lines = [proc.stderr.strip()] if proc.stderr.strip() else []
    build_sec = float(metrics.get("build_sec", 0.0))
    search_sec_mean = float(metrics.get("search_sec_mean", 0.0))
    qps_search = float(metrics.get("qps", 0.0))
    e2e = build_sec + search_sec_mean
    e2e_qps = float(nq / e2e) if e2e > 0 else 0.0

    print(
        f"  [VDB] complete build={build_dir.name} index={index} "
        f"build={build_sec:.4f}s search_mean={search_sec_mean:.4f}s "
        f"qps_search={qps_search:.2f} qps_e2e={e2e_qps:.2f} wall={wall:.4f}s",
        flush=True,
    )
    return {
        "engine": "vdb",
        "index": index,
        "build_dir": str(build_dir.name),
        "threads": threads,
        "build_sec": build_sec,
        "search_sec_mean": search_sec_mean,
        "qps_search": qps_search,
        "qps_end_to_end": e2e_qps,
        "wall_sec": wall,
        "ids": last_ids,
        "stderr": stderr_lines,
    }


def bench_faiss(xb: np.ndarray, xq: np.ndarray, index: str, k: int,
                ivf_nlist: int, ivf_nprobe: int, hnsw_m: int, hnsw_ef_search: int):
    try:
        import faiss  # type: ignore
    except Exception as e:
        return {"engine": "faiss", "index": index, "available": False, "error": str(e)}

    xb = np.ascontiguousarray(xb.astype(np.float32))
    xq = np.ascontiguousarray(xq.astype(np.float32))
    d = xb.shape[1]

    t0 = time.perf_counter()
    if index == "flat":
        idx = faiss.IndexFlatL2(d)
        idx.add(xb)
    elif index == "ivf":
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(quantizer, d, ivf_nlist, faiss.METRIC_L2)
        idx.train(xb)
        idx.add(xb)
        idx.nprobe = ivf_nprobe
    elif index == "hnsw":
        idx = faiss.IndexHNSWFlat(d, hnsw_m)
        idx.hnsw.efSearch = hnsw_ef_search
        idx.add(xb)
    else:
        raise ValueError(index)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _dist, ids = idx.search(xq, k)
    t3 = time.perf_counter()

    return {
        "engine": "faiss",
        "index": index,
        "available": True,
        "build_sec": float(t1 - t0),
        "search_sec": float(t3 - t2),
        "qps": float(xq.shape[0] / (t3 - t2)),
        "qps_end_to_end": float(xq.shape[0] / ((t1 - t0) + (t3 - t2))),
        "ids": ids.astype(np.int64),
    }


def ids_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    ap = argparse.ArgumentParser(description="Benchmark VDB vs FAISS")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--ivf-nlist", type=int, default=256)
    ap.add_argument("--ivf-nprobe", type=int, default=8)
    ap.add_argument("--hnsw-m", type=int, default=16)
    ap.add_argument("--hnsw-ef-search", type=int, default=64)
    ap.add_argument("--output", default="benchmarks/results.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    bench_dir = root / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    xb = rng.random((args.n, args.dim), dtype=np.float32)
    xq = rng.random((args.nq, args.dim), dtype=np.float32)

    data_bin = bench_dir / "data.bin"
    query_bin = bench_dir / "queries.bin"
    write_bin_matrix(data_bin, xb)
    write_bin_matrix(query_bin, xq)

    truth = exact_topk(xb, xq, args.k)

    build_debug = root / "build_debug"
    build_release = root / "build_release"

    print("[1/4] Building Debug (no optimization)...", flush=True)
    build_vdb(root, build_debug, "Debug")
    print("[2/4] Building Release (optimized)...", flush=True)
    build_vdb(root, build_release, "Release")

    print("[3/4] Running VDB benchmarks...", flush=True)
    vdb_results = []
    for build_dir in [build_debug, build_release]:
        for index in ["ivf", "hnsw"]:
            r = bench_vdb(
                root, build_dir, index,
                data_bin, query_bin,
                args.nq, args.k, args.runs, args.threads,
                args.ivf_nlist, args.ivf_nprobe,
                args.hnsw_m, args.hnsw_ef_search,
            )
            r["recall_at_k"] = recall_at_k(r["ids"], truth, args.k)
            r["ids"] = ids_to_list(r["ids"])
            vdb_results.append(r)

    print("[4/4] Running FAISS benchmarks (if installed)...", flush=True)
    faiss_results = []
    for index in ["flat", "ivf", "hnsw"]:
        r = bench_faiss(
            xb, xq, index, args.k,
            args.ivf_nlist, args.ivf_nprobe,
            args.hnsw_m, args.hnsw_ef_search,
        )
        if r.get("available"):
            r["recall_at_k"] = recall_at_k(r["ids"], truth, args.k)
            r["ids"] = ids_to_list(r["ids"])
        faiss_results.append(r)

    out = {
        "config": {
            "n": args.n,
            "dim": args.dim,
            "nq": args.nq,
            "k": args.k,
            "runs": args.runs,
            "threads": args.threads,
            "ivf_nlist": args.ivf_nlist,
            "ivf_nprobe": args.ivf_nprobe,
            "hnsw_m": args.hnsw_m,
            "hnsw_ef_search": args.hnsw_ef_search,
        },
        "vdb": vdb_results,
        "faiss": faiss_results,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
      out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    try:
        from plot_results import plot_results  # type: ignore

        plot_results(out_path, out_path.parent)
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    print(f"Wrote benchmark report: {out_path}")
    print("Summary:")
    for row in vdb_results:
        print(
            f"  VDB {row['build_dir']} {row['index']}: "
            f"qps_search={row['qps_search']:.2f} qps_e2e={row['qps_end_to_end']:.2f} "
            f"build={row['build_sec']:.4f}s search={row['search_sec_mean']:.4f}s "
            f"recall@{args.k}={row['recall_at_k']:.4f}"
        )
    for row in faiss_results:
        if row.get("available"):
            print(
                f"  FAISS {row['index']}: qps_search={row['qps']:.2f} "
                f"qps_e2e={row['qps_end_to_end']:.2f} "
                f"build={row['build_sec']:.4f}s search={row['search_sec']:.4f}s "
                f"recall@{args.k}={row['recall_at_k']:.4f}"
            )
        else:
            print(f"  FAISS {row['index']}: unavailable ({row.get('error', 'unknown')})")


if __name__ == "__main__":
    sys.exit(main())
