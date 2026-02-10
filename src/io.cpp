#include "vdb/io.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vdb {

Dataset load_f32_matrix(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open file: " + path);
  }

  std::uint32_t count = 0;
  std::uint32_t dim = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!in) {
    throw std::runtime_error("failed to read header: " + path);
  }

  Dataset data(count, dim);
  std::size_t bytes = static_cast<std::size_t>(count) * dim * sizeof(float);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  if (!in) {
    throw std::runtime_error("failed to read payload: " + path);
  }
  return data;
}

void save_search_results_text(const std::string& path, const std::vector<IndexId>& indices,
                              const std::vector<float>& distances, std::size_t nq,
                              std::size_t k) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      std::size_t idx = qi * k + j;
      out << qi << "," << j << "," << indices[idx] << "," << distances[idx] << "\n";
    }
  }
}

} // namespace vdb
