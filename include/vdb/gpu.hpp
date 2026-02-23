#pragma once

#include <cstddef>

#include "vdb/types.hpp"

namespace vdb {

bool gpu_available();

void gpu_knn_l2(const float* data, std::size_t n, std::size_t dim,
                const float* queries, std::size_t nq, std::size_t k,
                IndexId* out_indices, float* out_distances);

// --- GPU IVF index with persistent device memory ---

struct GpuIvfContext;

GpuIvfContext* gpu_ivf_build(const float* data, std::size_t n, std::size_t dim,
                             std::size_t nlist, std::size_t kmeans_iters,
                             float* out_centroids,
                             IndexId* out_assignments);

void gpu_ivf_search(const GpuIvfContext* ctx,
                    const float* queries, std::size_t nq,
                    std::size_t k, std::size_t nprobe,
                    IndexId* out_indices, float* out_distances);

void gpu_ivf_free(GpuIvfContext* ctx);

} // namespace vdb
