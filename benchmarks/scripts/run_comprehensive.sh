#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_comprehensive.sh  –  Build, benchmark, and plot everything
#
# Usage:
#   cd /path/to/vector_db
#   bash benchmarks/scripts/run_comprehensive.sh
#
# Outputs:
#   benchmarks/results/comprehensive_results.json
#   benchmarks/figures/comprehensive/fig1_*.png … fig7_*.png
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build_release"
CLI="$BUILD/vdb_cli"

echo "════════════════════════════════════════════════════════════"
echo "  Vector-DB Comprehensive Benchmark"
echo "  Root  : $ROOT"
echo "════════════════════════════════════════════════════════════"

# ── 1. Build release binary ──────────────────────────────────────────────────
if [[ ! -f "$CLI" ]]; then
    echo ""
    echo "▶ Building release binary …"
    cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD" -j
    echo "  Build complete → $CLI"
else
    echo ""
    echo "▶ Release binary found: $CLI"
fi

# ── 2. Run benchmark ─────────────────────────────────────────────────────────
echo ""
echo "▶ Running comprehensive benchmark (this may take 5-15 minutes) …"
cd "$ROOT"
python3 benchmarks/scripts/comprehensive_benchmark.py

# ── 3. Generate plots ────────────────────────────────────────────────────────
echo ""
echo "▶ Generating presentation figures …"
python3 benchmarks/scripts/comprehensive_plots.py

# ── 4. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done!"
echo "  Results → benchmarks/results/comprehensive_results.json"
echo "  Figures → benchmarks/figures/comprehensive/"
ls -1 "$ROOT/benchmarks/figures/comprehensive/"*.png 2>/dev/null | while read f; do
    echo "            $(basename "$f")"
done
echo "════════════════════════════════════════════════════════════"
