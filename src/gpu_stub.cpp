#include "vdb/gpu.hpp"

#include "vdb/distance.hpp"

namespace vdb {

bool gpu_available() {
  return false;
}

void gpu_batched_l2(const float* queries, const float* vectors, std::size_t nq,
                    std::size_t nv, std::size_t dim, float* out) {
  for (std::size_t i = 0; i < nq; ++i) {
    for (std::size_t j = 0; j < nv; ++j) {
      const float* q = queries + i * dim;
      const float* v = vectors + j * dim;
      out[i * nv + j] = l2_distance(q, v, dim);
    }
  }
}

} // namespace vdb
