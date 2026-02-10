#pragma once

#include <cstddef>

#include "vdb/dataset.hpp"
#include "vdb/types.hpp"

namespace vdb {

class Index {
public:
  virtual ~Index() = default;
  virtual void build(const Dataset& data) = 0;
  virtual void search(const Dataset& queries, std::size_t k, SearchResult& out,
                      const SearchParams& params) const = 0;
};

} // namespace vdb
