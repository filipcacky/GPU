#pragma once

#include <exception>

#include <cgm/cu_math.cuh>
#include <cgm/sanity.cuh>
#include <csr/traits.hpp>

#include <thrust/device_free.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace csr {

using namespace cgm;

template <typename T> struct matrix_algorithms<T, thrust::device_vector> {
  using value_t = T;
  using matrix_t = matrix<value_t, thrust::device_vector>;
  using vector_t = thrust::device_vector<value_t>;
  using index_t = thrust::device_vector<size_t>;

  __host__ static void dot(const matrix_t &mx, const vector_t &v,
                           vector_t &result) {
    assert(mx.width() == v.size());

    const vector_t &mx_data = mx.data();
    const index_t &mx_col_idx = mx.col_idx();
    const index_t &mx_row_ptr = mx.row_ptr();

    result.resize(v.size());

    if (mx.mean_nonzero() <= cu::cuWarpSize / 4) {
      dim3 block_size = std::min(mx.height(), cu::cuMaxThreads / 2);
      dim3 grid_size =
          std::ceil(static_cast<double>(mx.height()) / block_size.x);
      cu::csr_dot_vec_1<value_t><<<grid_size, block_size>>>(
          thrust::raw_pointer_cast(mx_data.data()),
          thrust::raw_pointer_cast(mx_col_idx.data()),
          thrust::raw_pointer_cast(mx_row_ptr.data()), mx.height(),
          thrust::raw_pointer_cast(v.data()),
          thrust::raw_pointer_cast(result.data()));
    } else {
      dim3 block_size =
          std::min(cu::cuWarpSize * mx.height(), cu::cuMaxThreads / 2);
      dim3 grid_size = std::ceil(
          cu::cuWarpSize * static_cast<double>(mx.height()) / block_size.x);

      cu::csr_dot_vec_32<value_t><<<grid_size, block_size>>>(
          thrust::raw_pointer_cast(mx_data.data()),
          thrust::raw_pointer_cast(mx_col_idx.data()),
          thrust::raw_pointer_cast(mx_row_ptr.data()), mx.height(),
          thrust::raw_pointer_cast(v.data()),
          thrust::raw_pointer_cast(result.data()));
    }
  }
};

template <typename T> struct vector_algorithms<T, thrust::device_vector> {
  using value_t = T;
  using vector_t = thrust::device_vector<value_t>;
  using device_ptr_t = thrust::device_ptr<value_t>;
  using host_ptr_t = value_t *;

  static device_ptr_t make_device_ptr() {
    return thrust::device_new<value_t>(1);
  }

  static host_ptr_t make_host_ptr() {
    value_t *res;
    cuTry(cudaHostAlloc(&res, sizeof(value_t), cudaHostAllocDefault));
    return res;
  }

  static void delete_device_ptr(device_ptr_t &&ptr) {
    thrust::device_free(ptr);
  }

  static void delete_host_ptr(host_ptr_t &&ptr) { cudaFreeHost(ptr); }

  __host__ static value_t norm(const vector_t &vector, int l,
                               device_ptr_t &device_ptr, host_ptr_t &host_ptr) {
    cuTry(cudaMemsetAsync(thrust::raw_pointer_cast(device_ptr), 0,
                          sizeof(value_t)));

    dim3 block_size = std::min(cu::cuMaxThreads, vector.size());
    dim3 grid_size =
        std::ceil(static_cast<double>(vector.size()) / block_size.x);

    cu::vec_norm<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(vector.data()), vector.size(), l,
        thrust::raw_pointer_cast(device_ptr));

    cu::cuCopy(thrust::raw_pointer_cast(host_ptr),
               thrust::raw_pointer_cast(device_ptr), 1, cudaMemcpyDeviceToHost);

    return *host_ptr;
  }

  __host__ static value_t dot(const vector_t &first, const vector_t &second,
                              device_ptr_t &device_ptr, host_ptr_t &host_ptr) {
    assert(first.size() == second.size());

    cuTry(cudaMemsetAsync(thrust::raw_pointer_cast(device_ptr), 0,
                          sizeof(value_t)));

    dim3 block_size = std::min(cu::cuMaxThreads, first.size());
    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / block_size.x);

    cu::vec_dot_vec<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()),
        thrust::raw_pointer_cast(second.data()), first.size(),
        thrust::raw_pointer_cast(device_ptr));

    cu::cuCopy(thrust::raw_pointer_cast(host_ptr),
               thrust::raw_pointer_cast(device_ptr), 1, cudaMemcpyDeviceToHost);

    return *host_ptr;
  }

  __host__ static void scale(vector_t &vector, value_t scalar) {

    dim3 block_size = std::min(cu::cuMaxThreads / 2, vector.size());
    dim3 grid_size = std::ceil(static_cast<double>(vector.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::scale<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(vector.data()), vector.size(), scalar);
  }

  __host__ static void scale_and_add(vector_t &first, value_t scalar,
                                     const vector_t &second) {
    assert(first.size() == second.size());

    dim3 block_size = std::min(cu::cuMaxThreads / 2, first.size());
    dim3 grid_size = std::ceil(static_cast<double>(first.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::scale_and_add<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());
  }

  __host__ static void add(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());

    dim3 block_size = std::min(cu::cuMaxThreads / 2, first.size());
    dim3 grid_size = std::ceil(static_cast<double>(first.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::add_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), value_t(1),
        thrust::raw_pointer_cast(second.data()), first.size());
  }

  __host__ static void add_scaled(vector_t &first, value_t scalar,
                                  const vector_t &second) {
    assert(first.size() == second.size());

    dim3 block_size = std::min(cu::cuMaxThreads / 2, first.size());
    dim3 grid_size = std::ceil(static_cast<double>(first.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::add_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());
  }

  __host__ static void sub(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());

    dim3 block_size = std::min(cu::cuMaxThreads / 2, first.size());
    dim3 grid_size = std::ceil(static_cast<double>(first.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::sub_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), value_t(1),
        thrust::raw_pointer_cast(second.data()), first.size());
  }

  __host__ static void sub_scaled(vector_t &first, value_t scalar,
                                  const vector_t &second) {
    assert(first.size() == second.size());

    dim3 block_size = std::min(cu::cuMaxThreads / 2, first.size());
    dim3 grid_size = std::ceil(static_cast<double>(first.size()) /
                               (block_size.x * cu::cuWarpSize));

    cu::sub_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());
  }
};

} // namespace csr
