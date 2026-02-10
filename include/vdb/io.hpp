#pragma once

#include <string>

#include "vdb/dataset.hpp"
#include "vdb/types.hpp"

namespace vdb {

// Simple binary format:
// [uint32 count][uint32 dim][count*dim float32 values] row-major.
Dataset load_f32_matrix(const std::string& path);
void save_search_results_text(const std::string& path, const std::vector<IndexId>& indices,
                              const std::vector<float>& distances, std::size_t nq, std::size_t k);

} // namespace vdb
