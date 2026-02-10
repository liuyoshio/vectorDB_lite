#include "vdb/ivf.hpp"

#include "vdb/distance.hpp"
#include "vdb/gpu.hpp"
#include "vdb/utils.hpp"

#include <algorithm>
#include <limits>
#include <random>

namespace vdb {

IvfIndex::IvfIndex(IvfConfig cfg) : cfg_(cfg) {}

void IvfIndex::build(const Dataset& data) {
  data_ = data;
  if (data_.count() == 0) {
    centroids_ = Dataset();
    lists_.clear();
    return;
  }
  if (cfg_.nlist == 0) {
    cfg_.nlist = 1;
  }
  if (cfg_.nlist > data_.count()) {
    cfg_.nlist = data_.count();
  }
  train_kmeans(data_);
  lists_.assign(cfg_.nlist, {});
  for (std::size_t i = 0; i < data_.count(); ++i) {
    std::size_t c = nearest_centroid(data_.at(i));
    lists_[c].push_back(static_cast<IndexId>(i));
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

  for (std::size_t qi = 0; qi < queries.count(); ++qi) {
    const float* q = queries.at(qi);
    std::vector<Neighbor> centroid_dists;
    centroid_dists.reserve(cfg_.nlist);
    for (std::size_t c = 0; c < cfg_.nlist; ++c) {
      float d = l2_distance(centroids_.at(c), q, data_.dim());
      centroid_dists.push_back({static_cast<IndexId>(c), d});
    }
    std::vector<Neighbor> probes = select_topk(centroid_dists, nprobe);

    std::vector<Neighbor> candidates;
    for (const auto& p : probes) {
      const auto& list = lists_[p.id];
      for (IndexId id : list) {
        float d = l2_distance(data_.at(id), q, data_.dim());
        candidates.push_back({id, d});
      }
    }

    std::vector<Neighbor> topk = select_topk(candidates, k);
    for (std::size_t i = 0; i < k; ++i) {
      std::size_t out_idx = qi * k + i;
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
  float best_dist = l2_distance(centroids_.at(0), vec, data_.dim());
  for (std::size_t c = 1; c < cfg_.nlist; ++c) {
    float d = l2_distance(centroids_.at(c), vec, data_.dim());
    if (d < best_dist) {
      best_dist = d;
      best = c;
    }
  }
  return best;
}

} // namespace vdb
