#pragma once

#include <cstddef>

#include "vdb/types.hpp"

namespace vdb {

void gpu_knn_l2(const float* data, std::size_t n, std::size_t dim,
                const float* queries, std::size_t nq, std::size_t k,
                IndexId* out_indices, float* out_distances);

} // namespace vdb
