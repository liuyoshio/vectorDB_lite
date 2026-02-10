#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "vdb/dataset.hpp"
#include "vdb/types.hpp"

namespace vdb {

struct Neighbor {
  IndexId id = 0;
  float dist = 0.0f;
};

std::size_t argmin_distance(const Dataset& data, const float* vec);
void shuffle_indices(std::vector<std::size_t>& indices, std::mt19937& rng);
std::vector<Neighbor> select_topk(const std::vector<Neighbor>& input, std::size_t k);

} // namespace vdb
