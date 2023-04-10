#pragma once

#include <csr/forward.hpp>
#include <csr/traits.hpp>

#include <cassert>
#include <utility>

namespace csr {

template <typename T, template <typename...> typename Storage> class matrix {
  static_assert(std::is_arithmetic<T>(),
                "CSR matrix is only available for arithmetic types");

public:
  using value_t = T;
  using value_storage_t = Storage<value_t>;
  using index_storage_t = Storage<size_t>;

  using vector_t = Storage<value_t>;

  using algorithms = matrix_algorithms<value_t, Storage>;

  matrix(value_storage_t &&data, index_storage_t &&col_idx,
         index_storage_t &&row_ptr, size_t width_, size_t mean_nonzero)
      : data_(std::move(data)), col_idx_(std::move(col_idx)),
        row_ptr_(std::move(row_ptr)), width_(width_),
        mean_nonzero_(mean_nonzero) {}

  const value_storage_t &data() const { return data_; }
  const index_storage_t &col_idx() const { return col_idx_; }
  const index_storage_t &row_ptr() const { return row_ptr_; }

  size_t height() const { return row_ptr_.size() - 1; }
  size_t width() const { return width_; }
  size_t mean_nonzero() const { return mean_nonzero_; }

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

  template <typename U> matrix<U, Storage> as() const {
    Storage<U> data(data_.begin(), data_.end());
    Storage<size_t> col_idx = col_idx_;
    Storage<size_t> row_ptr = row_ptr_;
    return matrix<value_t, Storage>(std::move(data), std::move(col_idx),
                                    std::move(row_ptr), width(),
                                    mean_nonzero());
  }

  template <template <typename...> typename S>
  matrix<value_t, S> transfer() const {
    S<value_t> data = data_;
    S<size_t> col_idx = col_idx_;
    S<size_t> row_ptr = row_ptr_;
    return matrix<value_t, S>(std::move(data), std::move(col_idx),
                              std::move(row_ptr), width(), mean_nonzero());
  }

  value_storage_t dot(const vector_t &vector) const {
    value_storage_t result(vector.size());
    algorithms::dot(*this, vector, result);
    return result;
  }

private:
  value_storage_t data_;
  index_storage_t col_idx_;
  index_storage_t row_ptr_;
  size_t width_;
  size_t mean_nonzero_;
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

  size_t mean_nonzero = data.size() / height;

  return matrix(std::move(data), std::move(col_idx), std::move(row_ptr), width,
                mean_nonzero);
}

} // namespace csr
