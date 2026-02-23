#include "vdb/gpu.hpp"

#include "vdb/distance.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace vdb {

bool gpu_available() {
  return false;
}

void gpu_knn_l2(const float* data, std::size_t n, std::size_t dim,
                const float* queries, std::size_t nq, std::size_t k,
                IndexId* out_indices, float* out_distances) {
  // CPU brute-force fallback when CUDA is not available.
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const float* q = queries + qi * dim;

    struct Pair {
      float dist;
      IndexId id;
      bool operator<(const Pair& o) const { return dist < o.dist; }
    };
    std::vector<Pair> all(n);
    for (std::size_t i = 0; i < n; ++i) {
      all[i] = {l2_distance(q, data + i * dim, dim), static_cast<IndexId>(i)};
    }

    std::size_t actual_k = k < n ? k : n;
    std::partial_sort(all.begin(), all.begin() + actual_k, all.end());

    for (std::size_t j = 0; j < k; ++j) {
      std::size_t idx = qi * k + j;
      if (j < actual_k) {
        out_indices[idx] = all[j].id;
        out_distances[idx] = all[j].dist;
      } else {
        out_indices[idx] = 0;
        out_distances[idx] = std::numeric_limits<float>::infinity();
      }
    }
  }
}

GpuIvfContext* gpu_ivf_build(const float*, std::size_t, std::size_t,
                             std::size_t, std::size_t,
                             float*, IndexId*) {
  return nullptr;
}

void gpu_ivf_search(const GpuIvfContext*, const float*, std::size_t,
                    std::size_t, std::size_t,
                    IndexId*, float*) {
}

void gpu_ivf_free(GpuIvfContext*) {
}

} // namespace vdb
