#include "vdb/utils.hpp"

#include "vdb/distance.hpp"

#include <algorithm>
#include <random>

namespace vdb {

std::size_t argmin_distance(const Dataset& data, const float* vec) {
  std::size_t best = 0;
  float best_dist = l2_distance(data.at(0), vec, data.dim());
  for (std::size_t i = 1; i < data.count(); ++i) {
    float d = l2_distance(data.at(i), vec, data.dim());
    if (d < best_dist) {
      best_dist = d;
      best = i;
    }
  }
  return best;
}

void shuffle_indices(std::vector<std::size_t>& indices, std::mt19937& rng) {
  std::shuffle(indices.begin(), indices.end(), rng);
}

std::vector<Neighbor> select_topk(const std::vector<Neighbor>& input, std::size_t k) {
  if (k == 0 || input.empty()) {
    return {};
  }
  std::vector<Neighbor> tmp = input;
  if (k >= tmp.size()) {
    std::sort(tmp.begin(), tmp.end(), [](const Neighbor& a, const Neighbor& b) {
      return a.dist < b.dist;
    });
    return tmp;
  }
  std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(),
                   [](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; });
  tmp.resize(k);
  std::sort(tmp.begin(), tmp.end(), [](const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist;
  });
  return tmp;
}

} // namespace vdb
