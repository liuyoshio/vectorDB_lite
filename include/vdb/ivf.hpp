#pragma once

#include <cstddef>
#include <vector>

#include "vdb/dataset.hpp"
#include "vdb/index.hpp"
#include "vdb/types.hpp"

namespace vdb {

struct IvfConfig {
  std::size_t nlist = 1024;
  std::size_t kmeans_iters = 20;
};

class IvfIndex : public Index {
public:
  explicit IvfIndex(IvfConfig cfg = {});

  void build(const Dataset& data) override;
  void search(const Dataset& queries, std::size_t k, SearchResult& out,
              const SearchParams& params) const override;

  const Dataset& centroids() const { return centroids_; }

private:
  IvfConfig cfg_;
  Dataset data_;           // Original data (for GPU path)
  Dataset centroids_;
  std::vector<float> centroid_norms_sq_;  // Precomputed ||c||^2 per centroid

  // Clustered storage: vectors reordered by cluster for cache-friendly access
  Dataset clustered_data_;  // Vectors contiguous per cluster
  std::vector<float> data_norms_sq_;      // ||v||^2 for each vector in clustered_data_
  std::vector<std::size_t> list_offsets_; // list_offsets_[c] = start index for cluster c
  std::vector<std::vector<IndexId>> list_ids_;  // Original indices per cluster

  void train_kmeans(const Dataset& data);
  std::size_t nearest_centroid(const float* vec) const;
};

} // namespace vdb
