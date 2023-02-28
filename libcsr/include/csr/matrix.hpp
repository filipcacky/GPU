#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <cmath>

namespace csr {

template <typename T, template <typename...> typename Storage> class vector {
  static_assert(std::is_floating_point_v<T>,
                "CSR vector is only available for floating point types");

public:
  using value_t = T;
  using value_storage_t = Storage<T>;
  using index_storage_t = Storage<size_t>;

  explicit vector(size_t width) : width_(width) {}

  vector(value_storage_t &&data, index_storage_t &&col_idx, size_t width_)
      : data_(data), col_idx_(col_idx), width_(width_) {}

  const value_storage_t &data() const { return data_; }
  const index_storage_t &col_idx() const { return col_idx_; }

  size_t height() const { return 1; }
  size_t width() const { return width_; }
  value_t norm(size_t l) const {
    return std::accumulate(
        data_.begin(), data_.end(), 0.,
        [&](auto result, auto value) { return result + std::pow(value, l); });
  };
  std::pair<size_t, size_t> shape() const {
    return std::make_pair(height(), width());
  }

  value_storage_t to_dense() const {
    value_storage_t dense(height() * width());

    for (size_t data_idx = 0; data_idx < data_.size(); ++data_idx) {
      size_t col_idx = col_idx_[data_idx];
      dense[col_idx] = data_[data_idx];
    }

    return dense;
  }

  value_t at(size_t y, size_t x) const {
    assert(y == 0);
    return at(x);
  }

  value_t at(size_t idx) const {
    assert(idx < width());

    for (size_t data_idx = 0; data_idx < data_.size(); ++data_idx) {

      if (col_idx_[data_idx] > idx)
        return {};

      if (col_idx_[data_idx] == idx)
        return data_[data_idx];
    }

    return {};
  }

  value_t dot(const vector &other) {
    value_t result = 0;

    for (size_t this_idx = 0, other_idx = 0;
         this_idx < data_.size() && other_idx < other.data_.size();) {
      if (col_idx_[this_idx] == other.col_idx_[other_idx]) {
        result += data_[this_idx] * other.data_[other_idx];
        this_idx += 1;
        other_idx += 1;
        continue;
      }

      if (col_idx_[this_idx] < other.col_idx_[other_idx]) {
        this_idx += 1;
      } else {
        other_idx += 1;
      }
    }

    return result;
  }

  template <typename U> vector<U, Storage> as() {
    Storage<U> data(data_.begin(), data_.end());
    Storage<size_t> col_idx = col_idx_;
    return vector(std::move(data), std::move(col_idx), width());
  }

  vector &operator*=(value_t s) {
    std::for_each(data_.begin(), data_.end(), [&](auto &v) { v *= s; });
    return *this;
  }

  vector operator*(value_t s) {
    vector res = *this;
    res *= s;
    return res;
  }

  friend vector operator*(value_t s, const vector &mx) {
    vector res = mx;
    res *= s;
    return res;
  }

  vector operator+(const vector &other) const {
    value_storage_t new_storage;
    index_storage_t new_index;

    for (size_t this_idx = 0, other_idx = 0;
         this_idx < data_.size() || other_idx < other.data_.size();) {
      if (this_idx >= data_.size()) {
        new_storage.push_back(other.data_[other_idx]);
        new_index.push_back(other.col_idx_[other_idx]);
        other_idx += 1;
        continue;
      }

      if (other_idx >= other.data_.size()) {
        new_storage.push_back(data_[this_idx]);
        new_index.push_back(col_idx_[this_idx]);
        this_idx += 1;
        continue;
      }

      if (col_idx_[this_idx] == other.col_idx_[other_idx]) {
        new_storage.push_back(data_[this_idx] + other.data_[other_idx]);
        new_index.push_back(col_idx_[this_idx]);
        this_idx += 1;
        other_idx += 1;
        continue;
      }

      if (col_idx_[this_idx] < other.col_idx_[other_idx]) {
        new_storage.push_back(data_[this_idx]);
        new_index.push_back(col_idx_[this_idx]);
        this_idx += 1;
      } else {
        new_storage.push_back(other.data_[other_idx]);
        new_index.push_back(other.col_idx_[other_idx]);
        other_idx += 1;
      }
    }

    return vector(std::move(new_storage), std::move(new_index), width());
  }

  vector operator-(const vector &other) const { return operator+(-other); }

  vector operator-() const {
    value_storage_t new_storage;
    new_storage.reserve(data_.size());
    std::transform(data_.begin(), data_.end(), std::back_inserter(new_storage),
                   [](auto v) { return -v; });
    index_storage_t new_index = col_idx_;
    return vector(std::move(new_storage), std::move(new_index), width());
  }

private:
  value_storage_t data_;
  index_storage_t col_idx_;
  size_t width_;
};

template <typename T, template <typename...> typename Storage> class matrix {
  static_assert(std::is_floating_point_v<T>,
                "CSR matrix is only available for floating point types");

public:
  using value_t = T;
  using value_storage_t = Storage<T>;
  using index_storage_t = Storage<size_t>;

  using vector_t = vector<value_t, Storage>;

  matrix(value_storage_t &&data, index_storage_t &&col_idx,
         index_storage_t &&row_ptr, size_t width_)
      : data_(data), col_idx_(col_idx), row_ptr_(row_ptr), width_(width_) {}

  const value_storage_t &data() const { return data_; }
  const index_storage_t &col_idx() const { return col_idx_; }
  const index_storage_t &row_ptr() const { return row_ptr_; }

  size_t height() const { return row_ptr_.size() - 1; }
  size_t width() const { return width_; }

  std::pair<size_t, size_t> shape() const {
    return std::make_pair(height(), width());
  }

  value_storage_t to_dense() const {
    value_storage_t dense(height() * width());

    for (size_t row = 0; row < height(); ++row) {
      const size_t segment_size = row_ptr_[row + 1] - row_ptr_[row];
      for (size_t index_within_segment = 0; index_within_segment < segment_size;
           ++index_within_segment) {
        const size_t data_idx = row_ptr_[row] + index_within_segment;
        const value_t &value = data_[data_idx];
        const size_t col = col_idx_[data_idx];
        dense[row * width() + col] = value;
      }
    }

    return dense;
  }

  value_t at(size_t row, size_t col) const {
    assert(row < height());
    assert(col < width());

    size_t segment_size = row_ptr_[row + 1] - row_ptr_[row];

    for (size_t index_within_segment = 0; index_within_segment < segment_size;
         ++index_within_segment) {
      const size_t data_idx = row_ptr_[row] + index_within_segment;

      if (col_idx_[data_idx] > col)
        return {};

      if (col_idx_[data_idx] == col)
        return data_[data_idx];
    }

    return {};
  }

  template <typename U> matrix<U, Storage> as() {
    Storage<U> data(data_.begin(), data_.end());
    Storage<size_t> col_idx = col_idx_;
    Storage<size_t> row_ptr = row_ptr_;
    return matrix(std::move(data), std::move(col_idx), std::move(row_ptr),
                  width());
  }

  matrix &append(const vector_t &other) {
    assert(width() == other.width());

    size_t data_size_before = data_.size();
    data_.insert(data_.end(), other.data().begin(), other.data().end());
    col_idx_.insert(col_idx_.end(), other.col_idx().begin(),
                    other.col_idx().end());

    row_ptr_.push_back(data_size_before + other.width());

    return *this;
  }

  matrix &append(const matrix &other) {
    assert(width() == other.width());

    size_t data_size_before = data_.size();
    data_.insert(data_.end(), other.data_.begin(), other.data_.end());
    col_idx_.insert(col_idx_.end(), other.col_idx_.begin(),
                    other.col_idx_.end());

    row_ptr_.pop_back();
    for (size_t other_ptr : other.row_ptr_) {
      row_ptr_.push_back(other_ptr + data_size_before);
    }

    return *this;
  }

  vector_t dot(const vector_t &vector) const {
    typename vector_t::value_storage_t new_values;
    new_values.reserve(height());
    typename vector_t::index_storage_t new_indices;
    new_indices.reserve(new_values.size());

    for (size_t row = 0; row < height(); ++row) {
      const size_t segment_size = row_ptr_[row + 1] - row_ptr_[row];

      value_t value = 0;

      for (size_t index_within_segment = 0, vector_idx = 0;
           index_within_segment < segment_size &&
           vector_idx < vector.data().size();) {
        size_t data_idx = row_ptr_[row] + index_within_segment;

        if (col_idx_[data_idx] == vector.col_idx()[vector_idx]) {
          value += data_[data_idx] * vector.data()[vector_idx];
          index_within_segment += 1;
          vector_idx += 1;
          continue;
        }

        if (col_idx_[data_idx] < vector.col_idx()[vector_idx]) {
          index_within_segment += 1;
        } else {
          vector_idx += 1;
        }
      }

      if (value != 0) {
        new_values.push_back(value);
        new_indices.push_back(row);
      }
    }

    return vector_t(std::move(new_values), std::move(new_indices), height());
  }

private:
  value_storage_t data_;
  index_storage_t col_idx_;
  index_storage_t row_ptr_;
  size_t width_;
};

template <template <typename...> typename Storage, typename... Ts>
static auto from_dense(const Storage<Ts...> &dense,
                       std::pair<size_t, size_t> shape) {
  const auto [height, width] = shape;

  Storage<Ts...> data;
  Storage<size_t> col_idx;
  Storage<size_t> row_ptr(height + 1, 0);

  size_t idx = 0;
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x, ++idx) {
      if (dense[idx] != 0) {
        data.push_back(dense[idx]);
        col_idx.push_back(x);
      }
      row_ptr[y + 1] = data.size();
    }
  }

  return matrix(std::move(data), std::move(col_idx), std::move(row_ptr), width);
}

template <template <typename...> typename Storage, typename... Ts>
static auto from_dense(const Storage<Ts...> &dense) {
  const auto width = dense.size();

  Storage<Ts...> data;
  Storage<size_t> col_idx;

  size_t idx = 0;
  for (size_t x = 0; x < width; ++x, ++idx) {
    if (dense[idx] != 0) {
      data.push_back(dense[idx]);
      col_idx.push_back(x);
    }
  }

  return vector(std::move(data), std::move(col_idx), width);
}

} // namespace csr
