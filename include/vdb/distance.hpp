#pragma once

#include <cstddef>

namespace vdb {

float l2_distance(const float* a, const float* b, std::size_t dim);
float cosine_distance(const float* a, const float* b, std::size_t dim);
float dot_product(const float* a, const float* b, std::size_t dim);
float l2_norm(const float* a, std::size_t dim);

} // namespace vdb
