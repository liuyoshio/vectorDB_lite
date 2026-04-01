#pragma once

#include <cstddef>

namespace vdb {

float l2_distance(const float* a, const float* b, std::size_t dim);
float cosine_distance(const float* a, const float* b, std::size_t dim);
float dot_product(const float* a, const float* b, std::size_t dim);
float l2_norm(const float* a, std::size_t dim);

/** Squared L2 norm: sum of a[i]^2 (no sqrt). */
float l2_norm_sq(const float* a, std::size_t dim);

/** L2 squared distance via dot-product form: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*<a,b>. */
float l2_sq_from_norms(float norm_a_sq, float norm_b_sq, float dot_ab);

} // namespace vdb
