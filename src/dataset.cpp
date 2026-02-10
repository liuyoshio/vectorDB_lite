#include "vdb/dataset.hpp"

namespace vdb {

Dataset::Dataset(std::size_t count, std::size_t dim)
    : count_(count), dim_(dim), data_(count * dim, 0.0f) {}

float* Dataset::data() { return data_.data(); }
const float* Dataset::data() const { return data_.data(); }

float* Dataset::at(std::size_t idx) { return data_.data() + idx * dim_; }
const float* Dataset::at(std::size_t idx) const { return data_.data() + idx * dim_; }

} // namespace vdb
