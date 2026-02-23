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
  ~IvfIndex();

  IvfIndex(const IvfIndex&) = delete;
  IvfIndex& operator=(const IvfIndex&) = delete;
  IvfIndex(IvfIndex&& other) noexcept;
  IvfIndex& operator=(IvfIndex&& other) noexcept;

  void build(const Dataset& data) override;
  void search(const Dataset& queries, std::size_t k, SearchResult& out,
              const SearchParams& params) const override;

  const Dataset& centroids() const { return centroids_; }

private:
  IvfConfig cfg_;
  Dataset data_;
  Dataset centroids_;
  std::vector<std::vector<IndexId>> lists_;
  void* gpu_ctx_ = nullptr;

  void train_kmeans(const Dataset& data);
  std::size_t nearest_centroid(const float* vec) const;
};

} // namespace vdb
