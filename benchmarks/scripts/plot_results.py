#!/usr/bin/env python3
import json
from pathlib import Path


def _pick_vdb_release(rows):
    out = {}
    for r in rows:
        if r.get("build_dir") == "build_release":
            out[r["index"]] = r
    return out


def _pick_vdb_debug(rows):
    out = {}
    for r in rows:
        if r.get("build_dir") == "build_debug":
            out[r["index"]] = r
    return out


def plot_results(results_path: Path, out_dir: Path):
    import matplotlib.pyplot as plt

    payload = json.loads(results_path.read_text())
    vdb_rows = payload.get("vdb", [])
    faiss_rows = [r for r in payload.get("faiss", []) if r.get("available")]

    vdb_rel = _pick_vdb_release(vdb_rows)
    vdb_dbg = _pick_vdb_debug(vdb_rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Search QPS comparison
    labels = []
    values = []

    for idx in ["ivf", "hnsw"]:
        if idx in vdb_rel:
            labels.append(f"vdb-{idx}")
            values.append(vdb_rel[idx].get("qps_search", 0.0))

    for idx in ["flat", "ivf", "hnsw"]:
        row = next((r for r in faiss_rows if r.get("index") == idx), None)
        if row:
            labels.append(f"faiss-{idx}")
            values.append(row.get("qps", 0.0))

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylabel("QPS (search-only)")
    plt.title("Search Throughput: VDB (Release) vs FAISS")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "qps_search_comparison.png", dpi=150)
    plt.close()

    # 2) Recall comparison
    labels = []
    values = []
    for idx in ["ivf", "hnsw"]:
        if idx in vdb_rel:
            labels.append(f"vdb-{idx}")
            values.append(vdb_rel[idx].get("recall_at_k", 0.0))

    for idx in ["flat", "ivf", "hnsw"]:
        row = next((r for r in faiss_rows if r.get("index") == idx), None)
        if row:
            labels.append(f"faiss-{idx}")
            values.append(row.get("recall_at_k", 0.0))

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Recall@k")
    plt.title("Recall Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "recall_comparison.png", dpi=150)
    plt.close()

    # 3) Debug -> Release speedup (VDB)
    labels = []
    speedups = []
    for idx in ["ivf", "hnsw"]:
        if idx in vdb_dbg and idx in vdb_rel:
            q_dbg = vdb_dbg[idx].get("qps_search", 0.0)
            q_rel = vdb_rel[idx].get("qps_search", 0.0)
            if q_dbg > 0:
                labels.append(idx)
                speedups.append(q_rel / q_dbg)

    if labels:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, speedups)
        plt.ylabel("Speedup (Release / Debug)")
        plt.title("Optimization Impact")
        plt.tight_layout()
        plt.savefig(out_dir / "optimization_speedup.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--out-dir", default=None)
    ns = ap.parse_args()

    rp = Path(ns.results_json).resolve()
    od = Path(ns.out_dir).resolve() if ns.out_dir else rp.parent
    plot_results(rp, od)
