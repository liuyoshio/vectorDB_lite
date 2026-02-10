#include "vdb/distance.hpp"

#include <cmath>

namespace vdb {

float dot_product(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2_norm(const float* a, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return std::sqrt(sum);
}

float l2_distance(const float* a, const float* b, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float cosine_distance(const float* a, const float* b, std::size_t dim) {
  float denom = l2_norm(a, dim) * l2_norm(b, dim);
  if (denom == 0.0f) {
    return 1.0f;
  }
  float sim = dot_product(a, b, dim) / denom;
  return 1.0f - sim;
}

} // namespace vdb
