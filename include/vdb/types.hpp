#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vdb {

using IndexId = std::uint32_t;

struct SearchResult {
  std::vector<IndexId> indices; // size = nq * k
  std::vector<float> distances; // size = nq * k
  std::size_t nq = 0;
  std::size_t k = 0;

  void resize(std::size_t nq_, std::size_t k_) {
    nq = nq_;
    k = k_;
    indices.assign(nq * k, 0);
    distances.assign(nq * k, 0.0f);
  }
};

struct SearchParams {
  std::size_t ef_search = 64; // HNSW
  std::size_t nprobe = 8;     // IVF
  bool use_gpu = false;
};

} // namespace vdb
