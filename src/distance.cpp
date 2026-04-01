#include "vdb/distance.hpp"

#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__SSE4_1__) && !defined(__AVX2__)
#include <smmintrin.h>
#endif

namespace vdb {

#if defined(__AVX2__)
namespace {

float dot_product_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 sum8 = _mm256_setzero_ps();
  std::size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum8 = _mm256_fmadd_ps(va, vb, sum8);
  }

  float sum = 0.0f;
  __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8),
                           _mm256_extractf128_ps(sum8, 1));
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2_distance_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 sum8 = _mm256_setzero_ps();
  std::size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 d = _mm256_sub_ps(va, vb);
    sum8 = _mm256_fmadd_ps(d, d, sum8);
  }

  float sum = 0.0f;
  __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8),
                           _mm256_extractf128_ps(sum8, 1));
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float l2_norm_sq_avx2(const float* a, std::size_t dim) {
  __m256 sum8 = _mm256_setzero_ps();
  std::size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    sum8 = _mm256_fmadd_ps(va, va, sum8);
  }

  float sum = 0.0f;
  __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8),
                           _mm256_extractf128_ps(sum8, 1));
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return sum;
}

} // namespace
#elif defined(__SSE4_1__)
namespace {

float dot_product_sse(const float* a, const float* b, std::size_t dim) {
  __m128 sum4 = _mm_setzero_ps();
  std::size_t i = 0;

  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    sum4 = _mm_add_ps(sum4, _mm_mul_ps(va, vb));
  }

  float sum = 0.0f;
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float l2_distance_sse(const float* a, const float* b, std::size_t dim) {
  __m128 sum4 = _mm_setzero_ps();
  std::size_t i = 0;

  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    __m128 vb = _mm_loadu_ps(b + i);
    __m128 d = _mm_sub_ps(va, vb);
    sum4 = _mm_add_ps(sum4, _mm_mul_ps(d, d));
  }

  float sum = 0.0f;
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float l2_norm_sq_sse(const float* a, std::size_t dim) {
  __m128 sum4 = _mm_setzero_ps();
  std::size_t i = 0;

  for (; i + 4 <= dim; i += 4) {
    __m128 va = _mm_loadu_ps(a + i);
    sum4 = _mm_add_ps(sum4, _mm_mul_ps(va, va));
  }

  float sum = 0.0f;
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum4 = _mm_hadd_ps(sum4, sum4);
  sum = _mm_cvtss_f32(sum4);

  for (; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return sum;
}

} // namespace
#endif

float dot_product(const float* a, const float* b, std::size_t dim) {
#if defined(__AVX2__)
  return dot_product_avx2(a, b, dim);
#elif defined(__SSE4_1__)
  return dot_product_sse(a, b, dim);
#else
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

float l2_norm(const float* a, std::size_t dim) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return std::sqrt(sum);
}

float l2_norm_sq(const float* a, std::size_t dim) {
#if defined(__AVX2__)
  return l2_norm_sq_avx2(a, dim);
#elif defined(__SSE4_1__)
  return l2_norm_sq_sse(a, dim);
#else
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    sum += a[i] * a[i];
  }
  return sum;
#endif
}

float l2_sq_from_norms(float norm_a_sq, float norm_b_sq, float dot_ab) {
  return norm_a_sq + norm_b_sq - 2.0f * dot_ab;
}

float l2_distance(const float* a, const float* b, std::size_t dim) {
#if defined(__AVX2__)
  return l2_distance_avx2(a, b, dim);
#elif defined(__SSE4_1__)
  return l2_distance_sse(a, b, dim);
#else
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
#endif
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
