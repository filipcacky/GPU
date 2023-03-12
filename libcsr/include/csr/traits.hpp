#pragma once

#include <csr/forward.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>

namespace csr {

template <typename T, template <typename...> typename Storage>
struct matrix_algorithms {
  using value_t = T;
  using matrix_t = matrix<value_t, Storage>;
  using vector_t = Storage<value_t>;

  static void dot(const matrix_t &mx, const vector_t &v, vector_t &result) {
    assert(v.size() == mx.width());

    result.resize(v.size());

#pragma omp parallel for
    for (size_t row = 0; row < mx.height(); ++row) {
      const auto segment_begin = mx.row_ptr()[row];
      const auto segment_end = mx.row_ptr()[row + 1];

      result[row] = 0;
      for (size_t idx = segment_begin; idx < segment_end; ++idx)
        result[row] += v[mx.col_idx()[idx]] * mx.data()[idx];
    }
  }
};

template <typename T, template <typename...> typename Storage>
struct vector_algorithms {
  using value_t = T;
  using vector_t = Storage<value_t>;
  using device_ptr_t = std::unique_ptr<value_t>;
  using host_ptr_t = std::unique_ptr<value_t>;

  static device_ptr_t make_device_ptr() {
    return std::make_unique<value_t>();
  }

  static host_ptr_t make_host_ptr() {
    return std::make_unique<value_t>();
  }

  static void delete_device_ptr(device_ptr_t&&) { }

  static void delete_host_ptr(host_ptr_t&&) {}

  static value_t norm(const vector_t &vector, size_t l, device_ptr_t &,
                      host_ptr_t &) {
    value_t result = 0;

#pragma omp parallel for reduction(+ : result)
    for (size_t idx = 0; idx < vector.size(); ++idx) {
      result += std::pow(vector[idx], l);
    }

    return result;
  }

  static value_t dot(const vector_t &lhs, const vector_t &rhs, device_ptr_t &,
                     host_ptr_t &) {
    value_t result = 0;

#pragma omp parallel for reduction(+ : result)
    for (size_t idx = 0; idx < lhs.size(); ++idx) {
      result += lhs[idx] * rhs[idx];
    }

    return result;
  }

  static void scale(vector_t &vector, value_t scalar) {
#pragma omp parallel for
    for (size_t idx = 0; idx < vector.size(); ++idx) {
      vector[idx] *= scalar;
    }
  }

  static void scale_and_add(vector_t &first, value_t scalar,
                            const vector_t &second) {
    assert(first.size() == second.size());
#pragma omp parallel for
    for (size_t idx = 0; idx < first.size(); ++idx) {
      first[idx] *= scalar;
      first[idx] += second[idx];
    }
  }

  static void add(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());
#pragma omp parallel for
    for (size_t idx = 0; idx < first.size(); ++idx) {
      first[idx] += second[idx];
    }
  }

  static void add_scaled(vector_t &first, value_t s, const vector_t &second) {
    assert(first.size() == second.size());
#pragma omp parallel for
    for (size_t idx = 0; idx < first.size(); ++idx) {
      first[idx] += s * second[idx];
    }
  }

  static void sub(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());
#pragma omp parallel for
    for (size_t idx = 0; idx < first.size(); ++idx) {
      first[idx] -= second[idx];
    }
  }

  static void sub_scaled(vector_t &first, value_t s, const vector_t &second) {
    assert(first.size() == second.size());
#pragma omp parallel for
    for (size_t idx = 0; idx < first.size(); ++idx) {
      first[idx] -= s * second[idx];
    }
  }
};

} // namespace csr
