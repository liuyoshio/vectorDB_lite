#include "vdb/dataset.hpp"
#include "vdb/hnsw.hpp"
#include "vdb/ivf.hpp"
#include "vdb/types.hpp"

#include <iostream>
#include <random>

int main() {
  const std::size_t n = 1000;
  const std::size_t dim = 32;
  const std::size_t nq = 5;
  const std::size_t k = 5;

  vdb::Dataset data(n, dim);
  vdb::Dataset queries(nq, dim);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (std::size_t i = 0; i < n * dim; ++i) {
    data.data()[i] = dist(rng);
  }
  for (std::size_t i = 0; i < nq * dim; ++i) {
    queries.data()[i] = dist(rng);
  }

  vdb::IvfIndex ivf({64, 15});
  ivf.build(data);

  vdb::HnswIndex hnsw({16, 200, 64, 42});
  hnsw.build(data);

  vdb::SearchParams params;
  params.nprobe = 8;
  params.ef_search = 64;
  params.use_gpu = false;

  vdb::SearchResult ivf_out;
  ivf.search(queries, k, ivf_out, params);

  vdb::SearchResult hnsw_out;
  hnsw.search(queries, k, hnsw_out, params);

  std::cout << "IVF results (first query):\n";
  for (std::size_t i = 0; i < k; ++i) {
    std::cout << "  id=" << ivf_out.indices[i]
              << " dist=" << ivf_out.distances[i] << "\n";
  }

  std::cout << "HNSW results (first query):\n";
  for (std::size_t i = 0; i < k; ++i) {
    std::cout << "  id=" << hnsw_out.indices[i]
              << " dist=" << hnsw_out.distances[i] << "\n";
  }

  return 0;
}
