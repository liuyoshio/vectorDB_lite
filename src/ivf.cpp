#include "vdb/ivf.hpp"

#include "vdb/distance.hpp"
#include "vdb/gpu.hpp"
#include "vdb/utils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>

#ifdef VDB_USE_OPENMP
#include <omp.h>
#endif

namespace vdb {

IvfIndex::IvfIndex(IvfConfig cfg) : cfg_(cfg) {}

void IvfIndex::build(const Dataset& data) {
  data_ = data;
  centroid_norms_sq_.clear();
  clustered_data_ = Dataset();
  data_norms_sq_.clear();
  list_offsets_.clear();
  list_ids_.clear();

  if (data_.count() == 0) {
    centroids_ = Dataset();
    return;
  }
  if (cfg_.nlist == 0) {
    cfg_.nlist = 1;
  }
  if (cfg_.nlist > data_.count()) {
    cfg_.nlist = data_.count();
  }

  train_kmeans(data_);

  // Precompute centroid norms (needed for nearest_centroid in assignment)
  centroid_norms_sq_.resize(cfg_.nlist);
  for (std::size_t c = 0; c < cfg_.nlist; ++c) {
    centroid_norms_sq_[c] = l2_norm_sq(centroids_.at(c), data_.dim());
  }

  // Assign vectors to clusters
  list_ids_.assign(cfg_.nlist, {});
  for (std::size_t i = 0; i < data_.count(); ++i) {
    std::size_t c = nearest_centroid(data_.at(i));
    list_ids_[c].push_back(static_cast<IndexId>(i));
  }

  // Build clustered storage: vectors contiguous per cluster
  list_offsets_.resize(cfg_.nlist + 1);
  list_offsets_[0] = 0;
  for (std::size_t c = 0; c < cfg_.nlist; ++c) {
    list_offsets_[c + 1] = list_offsets_[c] + list_ids_[c].size();
  }
  std::size_t total = list_offsets_[cfg_.nlist];
  clustered_data_ = Dataset(total, data_.dim());
  data_norms_sq_.resize(total);

  std::size_t dim = data_.dim();
  for (std::size_t c = 0; c < cfg_.nlist; ++c) {
    std::size_t base = list_offsets_[c];
    const auto& ids = list_ids_[c];
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const float* src = data_.at(ids[i]);
      float* dst = clustered_data_.at(base + i);
      std::copy(src, src + dim, dst);
      data_norms_sq_[base + i] = l2_norm_sq(dst, dim);
    }
  }
}

void IvfIndex::search(const Dataset& queries, std::size_t k, SearchResult& out,
                      const SearchParams& params) const {
  out.resize(queries.count(), k);
  if (queries.count() == 0 || data_.count() == 0 || k == 0) {
    return;
  }
#ifdef VDB_USE_CUDA
  if (params.use_gpu) {
    gpu_knn_l2(data_.data(), data_.count(), data_.dim(),
               queries.data(), queries.count(), k,
               out.indices.data(), out.distances.data());
    return;
  }
#endif

  std::size_t nprobe = params.nprobe > 0 ? params.nprobe : 1;
  if (nprobe > cfg_.nlist) {
    nprobe = cfg_.nlist;
  }

  std::size_t dim = clustered_data_.dim();

#ifdef VDB_USE_OPENMP
#pragma omp parallel
#endif
  {
    std::vector<Neighbor> centroid_dists;
    centroid_dists.reserve(cfg_.nlist);
    std::vector<Neighbor> candidates;

#ifdef VDB_USE_OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (std::int64_t qi = 0; qi < static_cast<std::int64_t>(queries.count()); ++qi) {
      const std::size_t qidx = static_cast<std::size_t>(qi);
      const float* q = queries.at(qidx);
      float q_norm_sq = l2_norm_sq(q, dim);

      // Centroid distances via dot-product form: ||c-q||^2 = ||c||^2 + ||q||^2 - 2*<c,q>
      centroid_dists.clear();
      for (std::size_t c = 0; c < cfg_.nlist; ++c) {
        float dot_cq = dot_product(centroids_.at(c), q, dim);
        float d = l2_sq_from_norms(centroid_norms_sq_[c], q_norm_sq, dot_cq);
        centroid_dists.push_back({static_cast<IndexId>(c), d});
      }
      std::vector<Neighbor> probes = select_topk(centroid_dists, nprobe);

      // Candidate distances from clustered storage (contiguous per list)
      candidates.clear();
      for (const auto& p : probes) {
        std::size_t start = list_offsets_[p.id];
        std::size_t end = list_offsets_[p.id + 1];
        for (std::size_t j = start; j < end; ++j) {
          float dot_vq = dot_product(clustered_data_.at(j), q, dim);
          float d = l2_sq_from_norms(data_norms_sq_[j], q_norm_sq, dot_vq);
          candidates.push_back({list_ids_[p.id][j - start], d});
        }
      }

      std::vector<Neighbor> topk = select_topk(candidates, k);
      for (std::size_t i = 0; i < k; ++i) {
        std::size_t out_idx = qidx * k + i;
        if (i < topk.size()) {
          out.indices[out_idx] = topk[i].id;
          out.distances[out_idx] = topk[i].dist;
        } else {
          out.indices[out_idx] = 0;
          out.distances[out_idx] = std::numeric_limits<float>::infinity();
        }
      }
    }
  }
}

void IvfIndex::train_kmeans(const Dataset& data) {
  centroids_ = Dataset(cfg_.nlist, data.dim());
  std::vector<std::size_t> indices(data.count());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  std::mt19937 rng(42);
  shuffle_indices(indices, rng);
  for (std::size_t c = 0; c < cfg_.nlist; ++c) {
    const float* src = data.at(indices[c]);
    std::copy(src, src + data.dim(), centroids_.at(c));
  }

  std::vector<std::size_t> counts(cfg_.nlist);
  std::vector<float> sums(cfg_.nlist * data.dim());

  for (std::size_t iter = 0; iter < cfg_.kmeans_iters; ++iter) {
    std::fill(counts.begin(), counts.end(), 0);
    std::fill(sums.begin(), sums.end(), 0.0f);

    for (std::size_t i = 0; i < data.count(); ++i) {
      std::size_t c = nearest_centroid(data.at(i));
      counts[c] += 1;
      float* sum = sums.data() + c * data.dim();
      const float* vec = data.at(i);
      for (std::size_t d = 0; d < data.dim(); ++d) {
        sum[d] += vec[d];
      }
    }

    for (std::size_t c = 0; c < cfg_.nlist; ++c) {
      float* dst = centroids_.at(c);
      if (counts[c] == 0) {
        const float* src = data.at(indices[c % indices.size()]);
        std::copy(src, src + data.dim(), dst);
        continue;
      }
      float inv = 1.0f / static_cast<float>(counts[c]);
      float* sum = sums.data() + c * data.dim();
      for (std::size_t d = 0; d < data.dim(); ++d) {
        dst[d] = sum[d] * inv;
      }
    }
  }
}

std::size_t IvfIndex::nearest_centroid(const float* vec) const {
  std::size_t best = 0;
  float best_dist;

  if (centroid_norms_sq_.size() == cfg_.nlist) {
    // Use dot-product form when norms are precomputed (after build, during assignment)
    float vec_norm_sq = l2_norm_sq(vec, data_.dim());
    float dot0 = dot_product(centroids_.at(0), vec, data_.dim());
    best_dist = l2_sq_from_norms(centroid_norms_sq_[0], vec_norm_sq, dot0);
    for (std::size_t c = 1; c < cfg_.nlist; ++c) {
      float dot_c = dot_product(centroids_.at(c), vec, data_.dim());
      float d = l2_sq_from_norms(centroid_norms_sq_[c], vec_norm_sq, dot_c);
      if (d < best_dist) {
        best_dist = d;
        best = c;
      }
    }
  } else {
    // Fallback during train_kmeans (centroid_norms_sq_ not yet computed)
    best_dist = l2_distance(centroids_.at(0), vec, data_.dim());
    for (std::size_t c = 1; c < cfg_.nlist; ++c) {
      float d = l2_distance(centroids_.at(c), vec, data_.dim());
      if (d < best_dist) {
        best_dist = d;
        best = c;
      }
    }
  }
  return best;
}

} // namespace vdb
