#pragma once

#include <cstddef>
#include <vector>

namespace vdb {

class Dataset {
public:
  Dataset() = default;
  Dataset(std::size_t count, std::size_t dim);

  float* data();
  const float* data() const;

  std::size_t count() const { return count_; }
  std::size_t dim() const { return dim_; }

  float* at(std::size_t idx);
  const float* at(std::size_t idx) const;

private:
  std::size_t count_ = 0;
  std::size_t dim_ = 0;
  std::vector<float> data_;
};

} // namespace vdb
