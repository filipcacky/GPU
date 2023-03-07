#pragma once

#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include <csr/matrix.hpp>
#include <npy/npy.hpp>
#include <thrust/host_vector.h>

namespace fs = std::filesystem;

namespace cli {

template <typename T>
csr::matrix<T, thrust::host_vector> load_npy_matrix(const fs::path &path) {
  std::vector<size_t> shape{};
  std::vector<T> npy_data;
  npy::LoadArrayFromNumpy(path.string(), shape, npy_data);

  if (shape.size() != 2 || shape[0] != shape[1]) {
    throw std::invalid_argument("Invalid lhs shape.");
  }

  thrust::host_vector<T> data = std::move(npy_data);
  return csr::from_dense(data, std::make_pair(shape[0], shape[1]));
}

template <typename T> std::vector<T> load_npy_vector(const fs::path &path) {
  std::vector<size_t> shape{};
  std::vector<T> data;
  npy::LoadArrayFromNumpy(path.string(), shape, data);

  if (shape.size() != 1) {
    throw std::invalid_argument("Invalid rhs shape.");
  }

  return data;
}

template <typename T>
csr::matrix<T, thrust::host_vector> load_txt_matrix(const fs::path &path) {
  std::ifstream file(path);

  size_t width, height, non_zero;

  if (!(file >> width >> height >> non_zero)) {
    throw std::runtime_error("Invalid file.");
  }

  thrust::host_vector<size_t> row_ptr_data;
  row_ptr_data.reserve(height);
  thrust::host_vector<size_t> col_idx_data;
  col_idx_data.reserve(non_zero);
  thrust::host_vector<T> value_data;
  value_data.reserve(non_zero);

  size_t last_row = 0;
  for (size_t non_zero_idx = 0; non_zero_idx < non_zero; ++non_zero_idx) {
    size_t row, column;
    T value;
    file >> row >> column >> value;

    if (last_row != row) {
      last_row = row;
      row_ptr_data.push_back(non_zero_idx);
    }
    col_idx_data.push_back(column - 1);
    value_data.push_back(value);
  }

  row_ptr_data.push_back(value_data.size());

  return csr::matrix(std::move(value_data), std::move(col_idx_data),
                     std::move(row_ptr_data), width);
}

template <typename T> std::vector<T> load_txt_vector(const fs::path &path) {
  std::vector<T> result;

  std::ifstream file(path);

  result.insert(result.end(), std::istream_iterator<T>(file),
                std::istream_iterator<T>());

  return result;
}

template <typename T>
csr::matrix<T, thrust::host_vector> load_matrix(const fs::path &path) {
  if (path.extension() == ".npy") {
    return load_npy_matrix<T>(path);
  } else if (path.extension() == ".txt") {
    return load_txt_matrix<T>(path);
  }
  throw std::runtime_error("Unknown data format.");
}

template <typename T> std::vector<T> load_vector(const fs::path &path) {
  if (path.extension() == ".npy") {
    return load_npy_vector<T>(path);
  } else if (path.extension() == ".txt") {
    return load_txt_vector<T>(path);
  }
  throw std::runtime_error("Unknown data format.");
}

} // namespace cli
