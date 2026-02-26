#!/usr/bin/env python3
"""Generate presentation-quality charts from comprehensive_results.json.

Figures produced in benchmarks/figures/comprehensive/:
  fig1_thread_scalability.png   – OpenMP speedup curves
  fig2_recall_qps_tradeoff.png  – Pareto frontier
  fig3_size_scaling.png         – QPS & build time vs N
  fig4_dim_impact.png           – QPS & recall vs D
  fig5_ivf_heatmap.png          – nlist × nprobe grids
  fig6_hnsw_heatmap.png         – M × ef_search grids
  fig7_summary_dashboard.png    – all-in-one overview
"""

import json, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ── paths ─────────────────────────────────────────────────────────────────────
# __file__ lives in  benchmarks/scripts/  →  .parent.parent = benchmarks/
BENCH   = Path(__file__).parent.parent.resolve()
ROOT    = BENCH.parent                            # project root
RESULTS = BENCH / "results" / "comprehensive_results.json"
OUT     = BENCH / "figures" / "comprehensive"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette & rcParams ────────────────────────────────────────────────────────
IVF_C    = "#1E88E5"   # blue
HNSW_C   = "#F4511E"   # deep-orange
IDEAL_C  = "#43A047"   # green

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          13,
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    2.4,
    "lines.markersize":   8,
})

def _save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved → {p.relative_to(ROOT)}")

def _fmt_qps(x, _=None):
    if x >= 1e6: return f"{x/1e6:.1f}M"
    if x >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"

def _heatmap_annotate(ax, grid, fmt_fn):
    """Write cell values on a heatmap."""
    vmax = np.nanmax(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v): continue
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, fmt_fn(v), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Thread Scalability
# ══════════════════════════════════════════════════════════════════════════════
def fig1_thread_scalability(data):
    d        = data["thread_scalability"]
    threads  = d["threads"]
    ivf_q    = np.array(d["ivf_qps"],  float)
    hnsw_q   = np.array(d["hnsw_qps"], float)
    ivf_sp   = ivf_q  / ivf_q[0]
    hnsw_sp  = hnsw_q / hnsw_q[0]
    ideal    = np.array(threads, float) / threads[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("OpenMP Thread Scalability  (N=20k, D=64, k=10)",
                 fontsize=17, fontweight="bold", y=1.02)

    # Left: absolute QPS
    ax1.plot(threads, ivf_q,  "o-", color=IVF_C,  label="IVF")
    ax1.plot(threads, hnsw_q, "s-", color=HNSW_C, label="HNSW")
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Search Throughput (QPS)")
    ax1.set_title("Absolute Throughput")
    ax1.set_xticks(threads)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax1.legend()

    # Right: speedup
    ax2.plot(threads, ivf_sp,  "o-", color=IVF_C,   label="IVF")
    ax2.plot(threads, hnsw_sp, "s-", color=HNSW_C,  label="HNSW")
    ax2.plot(threads, ideal,   "--", color=IDEAL_C,  alpha=0.75, label="Ideal Linear", linewidth=1.8)
    ax2.set_xlabel("Number of Threads")
    ax2.set_ylabel("Speedup  (× relative to 1 thread)")
    ax2.set_title("Parallel Speedup")
    ax2.set_xticks(threads)
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fig1_thread_scalability.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Recall-QPS Pareto Frontier
# ══════════════════════════════════════════════════════════════════════════════
def fig2_recall_qps(data):
    d    = data["recall_qps"]
    ivf  = d["ivf"]
    hnsw = d["hnsw"]

    ivf_r  = [p["recall"] for p in ivf]
    ivf_q  = [p["qps"]    for p in ivf]
    ivf_np = [p["nprobe"] for p in ivf]

    hnsw_r  = [p["recall"]    for p in hnsw]
    hnsw_q  = [p["qps"]       for p in hnsw]
    hnsw_ef = [p["ef_search"] for p in hnsw]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Recall-QPS Pareto Frontier  (N=20k, D=64, k=10)",
                 fontsize=16, fontweight="bold")

    ax.plot(ivf_r,  ivf_q,  "o-", color=IVF_C,  label="IVF  (nlist=256, vary nprobe)", zorder=3)
    ax.plot(hnsw_r, hnsw_q, "s-", color=HNSW_C, label="HNSW (M=16, vary ef_search)",  zorder=3)

    # Annotate parameter values
    for r, q, np_ in zip(ivf_r, ivf_q, ivf_np):
        ax.annotate(f"np={np_}", (r, q),
                    textcoords="offset points", xytext=(6, 5),
                    fontsize=9, color=IVF_C)
    for r, q, ef in zip(hnsw_r, hnsw_q, hnsw_ef):
        ax.annotate(f"ef={ef}", (r, q),
                    textcoords="offset points", xytext=(6, -14),
                    fontsize=9, color=HNSW_C)

    ax.set_xlabel("Recall@10  (higher → better)")
    ax.set_ylabel("Search Throughput (QPS)  (higher ↑ better)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig2_recall_qps_tradeoff.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Dataset-Size Scaling
# ══════════════════════════════════════════════════════════════════════════════
def fig3_size_scaling(data):
    d      = data["size_scaling"]
    sizes  = np.array(d["sizes"])
    ivf_q  = np.array(d["ivf_qps"],   float)
    hnsw_q = np.array(d["hnsw_qps"],  float)
    ivf_b  = np.array(d["ivf_build"],  float)
    hnsw_b = np.array(d["hnsw_build"], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scalability with Dataset Size  (D=64, k=10, 8 threads)",
                 fontsize=17, fontweight="bold", y=1.02)

    ax1.plot(sizes, ivf_q,  "o-", color=IVF_C,  label="IVF")
    ax1.plot(sizes, hnsw_q, "s-", color=HNSW_C, label="HNSW")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Dataset Size N")
    ax1.set_ylabel("Search Throughput (QPS)")
    ax1.set_title("Throughput vs Dataset Size")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax1.legend()

    ax2.plot(sizes, ivf_b,  "o-", color=IVF_C,  label="IVF")
    ax2.plot(sizes, hnsw_b, "s-", color=HNSW_C, label="HNSW")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Dataset Size N")
    ax2.set_ylabel("Index Build Time (s)")
    ax2.set_title("Build Time vs Dataset Size")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fig3_size_scaling.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Dimensionality Impact
# ══════════════════════════════════════════════════════════════════════════════
def fig4_dim_impact(data):
    d       = data["dim_impact"]
    dims    = d["dims"]
    ivf_q   = np.array(d["ivf_qps"],    float)
    hnsw_q  = np.array(d["hnsw_qps"],   float)
    ivf_r   = np.array(d["ivf_recall"],  float)
    hnsw_r  = np.array(d["hnsw_recall"], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Impact of Embedding Dimension  (N=10k, k=10, 8 threads)",
                 fontsize=17, fontweight="bold", y=1.02)

    ax1.plot(dims, ivf_q,  "o-", color=IVF_C,  label="IVF")
    ax1.plot(dims, hnsw_q, "s-", color=HNSW_C, label="HNSW")
    ax1.set_xlabel("Embedding Dimension D")
    ax1.set_ylabel("Search Throughput (QPS)")
    ax1.set_title("Throughput vs Dimension")
    ax1.set_xticks(dims)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax1.legend()

    ax2.plot(dims, ivf_r,  "o-", color=IVF_C,  label="IVF")
    ax2.plot(dims, hnsw_r, "s-", color=HNSW_C, label="HNSW")
    ax2.set_xlabel("Embedding Dimension D")
    ax2.set_ylabel("Recall@10")
    ax2.set_title("Recall vs Dimension")
    ax2.set_xticks(dims)
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fig4_dim_impact.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – IVF Heat-map
# ══════════════════════════════════════════════════════════════════════════════
def fig5_ivf_heatmap(data):
    d       = data["ivf_heatmap"]
    nlists  = d["nlists"]
    nprobes = d["nprobes"]
    qgrid   = np.array(d["qps"],    float)
    rgrid   = np.array(d["recall"], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IVF Parameter Sensitivity  (N=20k, D=64, k=10)",
                 fontsize=17, fontweight="bold", y=1.02)

    im1 = ax1.imshow(qgrid, aspect="auto", cmap="Blues",
                     vmin=np.nanmin(qgrid), vmax=np.nanmax(qgrid))
    ax1.set_xticks(range(len(nprobes))); ax1.set_xticklabels(nprobes)
    ax1.set_yticks(range(len(nlists)));  ax1.set_yticklabels(nlists)
    ax1.set_xlabel("nprobe"); ax1.set_ylabel("nlist")
    ax1.set_title("Search Throughput (QPS)")
    plt.colorbar(im1, ax=ax1, format=ticker.FuncFormatter(_fmt_qps))
    _heatmap_annotate(ax1, qgrid, lambda v: f"{v/1e3:.0f}K")

    im2 = ax2.imshow(rgrid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(len(nprobes))); ax2.set_xticklabels(nprobes)
    ax2.set_yticks(range(len(nlists)));  ax2.set_yticklabels(nlists)
    ax2.set_xlabel("nprobe"); ax2.set_ylabel("nlist")
    ax2.set_title("Recall@10")
    plt.colorbar(im2, ax=ax2, format="%.2f")
    _heatmap_annotate(ax2, rgrid, lambda v: f"{v:.2f}")

    fig.tight_layout()
    _save(fig, "fig5_ivf_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – HNSW Heat-map
# ══════════════════════════════════════════════════════════════════════════════
def fig6_hnsw_heatmap(data):
    d     = data["hnsw_heatmap"]
    Ms    = d["Ms"]
    efs   = d["efs"]
    qgrid = np.array(d["qps"],    float)
    rgrid = np.array(d["recall"], float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HNSW Parameter Sensitivity  (N=20k, D=64, k=10)",
                 fontsize=17, fontweight="bold", y=1.02)

    im1 = ax1.imshow(qgrid, aspect="auto", cmap="Oranges",
                     vmin=np.nanmin(qgrid), vmax=np.nanmax(qgrid))
    ax1.set_xticks(range(len(efs))); ax1.set_xticklabels(efs)
    ax1.set_yticks(range(len(Ms))); ax1.set_yticklabels(Ms)
    ax1.set_xlabel("ef_search"); ax1.set_ylabel("M")
    ax1.set_title("Search Throughput (QPS)")
    plt.colorbar(im1, ax=ax1, format=ticker.FuncFormatter(_fmt_qps))
    _heatmap_annotate(ax1, qgrid, lambda v: f"{v/1e3:.0f}K")

    im2 = ax2.imshow(rgrid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(len(efs))); ax2.set_xticklabels(efs)
    ax2.set_yticks(range(len(Ms))); ax2.set_yticklabels(Ms)
    ax2.set_xlabel("ef_search"); ax2.set_ylabel("M")
    ax2.set_title("Recall@10")
    plt.colorbar(im2, ax=ax2, format="%.2f")
    _heatmap_annotate(ax2, rgrid, lambda v: f"{v:.2f}")

    fig.tight_layout()
    _save(fig, "fig6_hnsw_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 – Summary Dashboard  (all 6 scenarios, one slide)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_summary_dashboard(data):
    fig = plt.figure(figsize=(21, 12))
    fig.suptitle(
        "Vector-Database Benchmark  ·  IVF vs HNSW  ·  OpenMP Acceleration",
        fontsize=20, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # ① Thread speedup
    ax = fig.add_subplot(gs[0, 0])
    d  = data["thread_scalability"]
    t  = d["threads"]
    sp_ivf  = np.array(d["ivf_qps"],  float) / d["ivf_qps"][0]
    sp_hnsw = np.array(d["hnsw_qps"], float) / d["hnsw_qps"][0]
    ideal   = np.array(t, float) / t[0]
    ax.plot(t, sp_ivf,  "o-", color=IVF_C,   label="IVF")
    ax.plot(t, sp_hnsw, "s-", color=HNSW_C,  label="HNSW")
    ax.plot(t, ideal,   "--", color=IDEAL_C,  alpha=0.65, label="Ideal", lw=1.8)
    ax.set_title("① Thread Scalability")
    ax.set_xlabel("Threads"); ax.set_ylabel("Speedup ×")
    ax.set_xticks(t); ax.legend(fontsize=9)

    # ② Recall-QPS
    ax  = fig.add_subplot(gs[0, 1])
    d   = data["recall_qps"]
    ax.plot([p["recall"] for p in d["ivf"]],
            [p["qps"]    for p in d["ivf"]],  "o-", color=IVF_C,  label="IVF")
    ax.plot([p["recall"] for p in d["hnsw"]],
            [p["qps"]    for p in d["hnsw"]], "s-", color=HNSW_C, label="HNSW")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax.set_title("② Recall-QPS Tradeoff")
    ax.set_xlabel("Recall@10"); ax.set_ylabel("QPS")
    ax.legend(fontsize=9)

    # ③ Size scaling – QPS
    ax = fig.add_subplot(gs[0, 2])
    d  = data["size_scaling"]
    sz = np.array(d["sizes"])
    ax.plot(sz, np.array(d["ivf_qps"],  float), "o-", color=IVF_C,  label="IVF")
    ax.plot(sz, np.array(d["hnsw_qps"], float), "s-", color=HNSW_C, label="HNSW")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax.set_title("③ Dataset Size Scaling")
    ax.set_xlabel("N (vectors)"); ax.set_ylabel("QPS")
    ax.legend(fontsize=9)

    # ④ Dimension – QPS
    ax = fig.add_subplot(gs[1, 0])
    d  = data["dim_impact"]
    ax.plot(d["dims"], np.array(d["ivf_qps"],  float), "o-", color=IVF_C,  label="IVF")
    ax.plot(d["dims"], np.array(d["hnsw_qps"], float), "s-", color=HNSW_C, label="HNSW")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_qps))
    ax.set_title("④ Dimension Impact")
    ax.set_xlabel("Dimension D"); ax.set_ylabel("QPS")
    ax.set_xticks(d["dims"]); ax.legend(fontsize=9)

    # ⑤ IVF Recall heatmap
    ax = fig.add_subplot(gs[1, 1])
    d  = data["ivf_heatmap"]
    rg = np.array(d["recall"], float)
    im = ax.imshow(rg, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(d["nprobes"]))); ax.set_xticklabels(d["nprobes"])
    ax.set_yticks(range(len(d["nlists"]))); ax.set_yticklabels(d["nlists"])
    ax.set_xlabel("nprobe"); ax.set_ylabel("nlist")
    ax.set_title("⑤ IVF Recall@10 Grid")
    plt.colorbar(im, ax=ax, format="%.2f")

    # ⑥ HNSW Recall heatmap
    ax = fig.add_subplot(gs[1, 2])
    d  = data["hnsw_heatmap"]
    rg = np.array(d["recall"], float)
    im = ax.imshow(rg, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(d["efs"]))); ax.set_xticklabels(d["efs"])
    ax.set_yticks(range(len(d["Ms"]))); ax.set_yticklabels(d["Ms"])
    ax.set_xlabel("ef_search"); ax.set_ylabel("M")
    ax.set_title("⑥ HNSW Recall@10 Grid")
    plt.colorbar(im, ax=ax, format="%.2f")

    _save(fig, "fig7_summary_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
PLOTTERS = {
    "thread_scalability": fig1_thread_scalability,
    "recall_qps":         fig2_recall_qps,
    "size_scaling":       fig3_size_scaling,
    "dim_impact":         fig4_dim_impact,
    "ivf_heatmap":        fig5_ivf_heatmap,
    "hnsw_heatmap":       fig6_hnsw_heatmap,
}

def main():
    if not RESULTS.exists():
        sys.exit(f"[ERROR] {RESULTS} not found.\nRun comprehensive_benchmark.py first.")

    with open(RESULTS) as f:
        data = json.load(f)

    print(f"Generating figures in {OUT.relative_to(ROOT)}/")
    for key, fn in PLOTTERS.items():
        if key in data:
            fn(data)
        else:
            print(f"  [SKIP] {key} not in results")

    # Summary only when all 6 scenarios present
    if all(k in data for k in PLOTTERS):
        fig7_summary_dashboard(data)
    else:
        print("  [SKIP] fig7 (not all scenarios present)")

    print(f"\nAll done → {OUT.relative_to(ROOT)}/")

if __name__ == "__main__":
    main()
