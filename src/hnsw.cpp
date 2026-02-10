#include "vdb/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_set>

#include "vdb/distance.hpp"
#include "vdb/gpu.hpp"
#include "vdb/utils.hpp"

namespace vdb {

HnswIndex::HnswIndex(HnswConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
  if (cfg_.M < 2) {
    cfg_.M = 2;
  }
  if (cfg_.ef_construction == 0) {
    cfg_.ef_construction = 1;
  }
  if (cfg_.ef_search == 0) {
    cfg_.ef_search = 1;
  }
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float p = 1.0f / static_cast<float>(cfg_.M);
  int level = 0;
  while (dist(rng_) < p && level < 32) {
    ++level;
  }
  return level;
}

float HnswIndex::distance_to(IndexId id, const float* vec) const {
  return l2_distance(vec, data_.at(id), data_.dim());
}

std::vector<IndexId> HnswIndex::search_layer(const float* query, IndexId entry, int level,
                                             std::size_t ef) const {
  struct NodeDist {
    IndexId id;
    float dist;
  };
  struct MinCmp {
    bool operator()(const NodeDist& a, const NodeDist& b) const { return a.dist > b.dist; }
  };
  struct MaxCmp {
    bool operator()(const NodeDist& a, const NodeDist& b) const { return a.dist < b.dist; }
  };

  std::priority_queue<NodeDist, std::vector<NodeDist>, MinCmp> candidates;
  std::priority_queue<NodeDist, std::vector<NodeDist>, MaxCmp> results;

  std::vector<char> visited(data_.count(), 0);

  float dist = distance_to(entry, query);
  candidates.push({entry, dist});
  results.push({entry, dist});
  visited[entry] = 1;

  while (!candidates.empty()) {
    NodeDist curr = candidates.top();
    candidates.pop();

    NodeDist worst = results.top();
    if (curr.dist > worst.dist && results.size() >= ef) {
      break;
    }

    if (level >= static_cast<int>(nodes_[curr.id].neighbors.size())) {
      continue;
    }
    const auto& neighbors = nodes_[curr.id].neighbors[level];
    for (IndexId nb : neighbors) {
      if (visited[nb]) {
        continue;
      }
      visited[nb] = 1;
      float d = distance_to(nb, query);
      if (results.size() < ef || d < results.top().dist) {
        candidates.push({nb, d});
        results.push({nb, d});
        if (results.size() > ef) {
          results.pop();
        }
      }
    }
  }

  std::vector<IndexId> out;
  out.reserve(results.size());
  std::vector<NodeDist> tmp;
  tmp.reserve(results.size());
  while (!results.empty()) {
    tmp.push_back(results.top());
    results.pop();
  }
  std::sort(tmp.begin(), tmp.end(),
            [](const NodeDist& a, const NodeDist& b) { return a.dist < b.dist; });
  for (const auto& nd : tmp) {
    out.push_back(nd.id);
  }
  return out;
}

std::vector<IndexId> HnswIndex::select_neighbors(const std::vector<IndexId>& candidates,
                                                 const float* query, std::size_t M) const {
  std::vector<Neighbor> tmp;
  tmp.reserve(candidates.size());
  for (IndexId id : candidates) {
    tmp.push_back({id, distance_to(id, query)});
  }
  auto top = select_topk(tmp, M);
  std::vector<IndexId> out;
  out.reserve(top.size());
  for (const auto& n : top) {
    out.push_back(n.id);
  }
  return out;
}

void HnswIndex::connect_new_node(IndexId id, int level, const std::vector<IndexId>& neighbors) {
  auto& nlist = nodes_[id].neighbors[level];
  for (IndexId nb : neighbors) {
    if (nb == id) {
      continue;
    }
    if (std::find(nlist.begin(), nlist.end(), nb) == nlist.end()) {
      nlist.push_back(nb);
    }
  }

  auto prune = [&](IndexId node_id) {
    auto& list = nodes_[node_id].neighbors[level];
    if (list.size() <= cfg_.M) {
      return;
    }
    std::vector<Neighbor> tmp;
    tmp.reserve(list.size());
    for (IndexId nb : list) {
      tmp.push_back({nb, distance_to(nb, data_.at(node_id))});
    }
    auto top = select_topk(tmp, cfg_.M);
    list.clear();
    for (const auto& t : top) {
      list.push_back(t.id);
    }
  };

  prune(id);
  for (IndexId nb : neighbors) {
    auto& list = nodes_[nb].neighbors[level];
    if (nb == id) {
      continue;
    }
    if (std::find(list.begin(), list.end(), id) == list.end()) {
      list.push_back(id);
    }
    prune(nb);
  }
}

void HnswIndex::build(const Dataset& data) {
  data_ = data;
  nodes_.clear();
  enterpoint_ = -1;
  max_level_ = -1;

  if (data_.count() == 0 || data_.dim() == 0) {
    return;
  }

  nodes_.reserve(data_.count());

  for (std::size_t i = 0; i < data_.count(); ++i) {
    int level = random_level();
    Node node;
    node.neighbors.resize(level + 1);
    nodes_.push_back(std::move(node));

    if (enterpoint_ == -1) {
      enterpoint_ = static_cast<int>(i);
      max_level_ = level;
      continue;
    }

    IndexId ep = static_cast<IndexId>(enterpoint_);
    for (int l = max_level_; l > level; --l) {
      auto candidates = search_layer(data_.at(i), ep, l, 1);
      if (!candidates.empty()) {
        ep = candidates.front();
      }
    }

    for (int l = std::min(level, max_level_); l >= 0; --l) {
      auto candidates = search_layer(data_.at(i), ep, l, cfg_.ef_construction);
      auto selected = select_neighbors(candidates, data_.at(i), cfg_.M);
      connect_new_node(static_cast<IndexId>(i), l, selected);
    }

    if (level > max_level_) {
      enterpoint_ = static_cast<int>(i);
      max_level_ = level;
    }
  }
}

void HnswIndex::search(const Dataset& queries, std::size_t k, SearchResult& out,
                       const SearchParams& params) const {
  out.resize(queries.count(), k);
  if (queries.count() == 0 || k == 0 || enterpoint_ == -1) {
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

  std::size_t ef = params.ef_search ? params.ef_search : cfg_.ef_search;

  for (std::size_t qi = 0; qi < queries.count(); ++qi) {
    IndexId ep = static_cast<IndexId>(enterpoint_);
    for (int l = max_level_; l > 0; --l) {
      auto candidates = search_layer(queries.at(qi), ep, l, 1);
      if (!candidates.empty()) {
        ep = candidates.front();
      }
    }

    auto candidates = search_layer(queries.at(qi), ep, 0, ef);
    std::vector<Neighbor> tmp;
    tmp.reserve(candidates.size());
    for (IndexId id : candidates) {
      tmp.push_back({id, distance_to(id, queries.at(qi))});
    }
    auto top = select_topk(tmp, k);

    for (std::size_t j = 0; j < k; ++j) {
      std::size_t out_idx = qi * k + j;
      if (j < top.size()) {
        out.indices[out_idx] = top[j].id;
        out.distances[out_idx] = top[j].dist;
      } else {
        out.indices[out_idx] = 0;
        out.distances[out_idx] = std::numeric_limits<float>::max();
      }
    }
  }
}

} // namespace vdb
