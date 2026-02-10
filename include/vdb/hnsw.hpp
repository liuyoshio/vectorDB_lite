#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "vdb/dataset.hpp"
#include "vdb/index.hpp"
#include "vdb/types.hpp"

namespace vdb {

struct HnswConfig {
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t ef_search = 64;
  unsigned int seed = 42;
};

class HnswIndex : public Index {
public:
  explicit HnswIndex(HnswConfig cfg = {});

  void build(const Dataset& data) override;
  void search(const Dataset& queries, std::size_t k, SearchResult& out,
              const SearchParams& params) const override;

private:
  struct Node {
    std::vector<std::vector<IndexId>> neighbors; // per level
  };

  HnswConfig cfg_;
  Dataset data_;
  std::vector<Node> nodes_;
  int enterpoint_ = -1;
  int max_level_ = -1;
  std::mt19937 rng_;

  int random_level();
  float distance_to(IndexId id, const float* vec) const;

  void connect_new_node(IndexId id, int level, const std::vector<IndexId>& neighbors);
  std::vector<IndexId> search_layer(const float* query, IndexId entry, int level,
                                    std::size_t ef) const;
  std::vector<IndexId> select_neighbors(const std::vector<IndexId>& candidates,
                                        const float* query, std::size_t M) const;
};

} // namespace vdb
