#pragma once

#include <exception>

#include <cgm/cu_math.cuh>
#include <cgm/sanity.cuh>
#include <csr/traits.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/memory.h>

/* #include <thrust/for_each.h> */
/* #include <thrust/reduce.h> */
/* #include <thrust/tabulate.h> */
/* #include <thrust/transform_reduce.h> */

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

    /* dim3 grid_size = */
    /*     std::ceil(static_cast<double>(mx.height()) / cu::cuMaxThreads); */
    /* dim3 block_size = std::min(cu::cuMaxThreads, mx.height()); */

    dim3 block_size = std::ceil(static_cast<double>(mx.height()) / cu::cuSM);
    dim3 grid_size = std::ceil(static_cast<double>(mx.height()) / block_size.x);

    cu::csr_dot_vec<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(mx_data.data()), mx_data.size(),
        thrust::raw_pointer_cast(mx_col_idx.data()), mx_col_idx.size(),
        thrust::raw_pointer_cast(mx_row_ptr.data()), mx_row_ptr.size(),
        thrust::raw_pointer_cast(v.data()),
        thrust::raw_pointer_cast(result.data()), v.size());

    cuTry(cudaDeviceSynchronize());
  }
};

template <typename T> struct vector_algorithms<T, thrust::device_vector> {
  using value_t = T;
  using vector_t = thrust::device_vector<value_t>;

  __host__ static value_t norm(const vector_t &vector, int l) {
    /* return thrust::transform_reduce(thrust::device, vector.begin(), */
    /*     vector.end(), cu::power_op<value_t>(l), */
    /*     0, thrust::plus<value_t>()); */

    // TODO should check sizes here for OOM
    auto [host_ptr, host_size] =
        thrust::get_temporary_buffer<value_t>(thrust::device, 1);
    *host_ptr = 0;
    auto [device_ptr, device_size] =
        thrust::get_temporary_buffer<value_t>(thrust::device, 1);

    cu::cuCopy(thrust::raw_pointer_cast(device_ptr),
               thrust::raw_pointer_cast(host_ptr), 1, cudaMemcpyHostToDevice);

    dim3 grid_size =
        std::ceil(static_cast<double>(vector.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, vector.size());

    cu::vec_norm<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(vector.data()), vector.size(), l,
        thrust::raw_pointer_cast(device_ptr));

    cuTry(cudaDeviceSynchronize());

    cu::cuCopy(thrust::raw_pointer_cast(host_ptr),
               thrust::raw_pointer_cast(device_ptr), 1, cudaMemcpyDeviceToHost);

    T result = *host_ptr;
    thrust::return_temporary_buffer(thrust::host, host_ptr, host_size);
    thrust::return_temporary_buffer(thrust::device, device_ptr, device_size);
    return result;
  }

  __host__ static value_t dot(const vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());
    // TODO: fix
    /* vector_t temp(first.size()); */
    /* thrust::tabulate( */
    /*     thrust::device, temp.begin(), temp.end(), */
    /*     cu::mul_vector_op(thrust::raw_pointer_cast(first.data()), */
    /*       thrust::raw_pointer_cast(second.data()))); */
    /* return thrust::reduce(thrust::device, temp.begin(), temp.end(), 0); */

    // TODO should check sizes here for OOM
    auto [host_ptr, host_size] =
        thrust::get_temporary_buffer<value_t>(thrust::host, 1);
    *host_ptr = 0;
    auto [device_ptr, device_size] =
        thrust::get_temporary_buffer<value_t>(thrust::device, 1);

    cu::cuCopy(thrust::raw_pointer_cast(device_ptr),
               thrust::raw_pointer_cast(host_ptr), 1, cudaMemcpyHostToDevice);

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::vec_dot_vec<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()),
        thrust::raw_pointer_cast(second.data()), first.size(),
        thrust::raw_pointer_cast(device_ptr));
    cuTry(cudaDeviceSynchronize());

    cu::cuCopy(thrust::raw_pointer_cast(host_ptr),
               thrust::raw_pointer_cast(device_ptr), 1, cudaMemcpyDeviceToHost);

    T result = *host_ptr;
    thrust::return_temporary_buffer(thrust::host, host_ptr, host_size);
    thrust::return_temporary_buffer(thrust::device, device_ptr, device_size);
    return result;
  }

  __host__ static void scale(vector_t &vector, value_t scalar) {
    /* thrust::for_each(thrust::device, vector.begin(), vector.end(), */
    /*     cu::scale_op(scalar)); */

    dim3 grid_size =
        std::ceil(static_cast<double>(vector.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, vector.size());

    cu::scale<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(vector.data()), vector.size(), scalar);
    cuTry(cudaDeviceSynchronize());
  }

  __host__ static void scale_and_add(vector_t &first, value_t scalar,
                                     const vector_t &second) {
    assert(first.size() == second.size());

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::scale_and_add<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());
    cuTry(cudaDeviceSynchronize());
  }

  __host__ static void add(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());
    /* thrust::tabulate(first.begin(), first.end(), */
    /*     cu::add_vector_scaled_op( */
    /*       thrust::raw_pointer_cast(first.data()), value_t(1), */
    /*       thrust::raw_pointer_cast(second.data()))); */

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::add_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), value_t(1),
        thrust::raw_pointer_cast(second.data()), first.size());
    cuTry(cudaDeviceSynchronize());
  }

  __host__ static void add_scaled(vector_t &first, value_t scalar,
                                  const vector_t &second) {
    assert(first.size() == second.size());
    /* auto op = */
    /*   cu::add_vector_scaled_op(thrust::raw_pointer_cast(first.data()),
     * scalar, */
    /*       thrust::raw_pointer_cast(second.data())); */
    /* thrust::tabulate(first.begin(), first.end(), op); */

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::add_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());

    cuTry(cudaDeviceSynchronize());
  }

  __host__ static void sub(vector_t &first, const vector_t &second) {
    assert(first.size() == second.size());
    /* thrust::tabulate(first.begin(), first.end(), */
    /*     cu::sub_vector_scaled_op( */
    /*       thrust::raw_pointer_cast(first.data()), value_t(1), */
    /*       thrust::raw_pointer_cast(second.data()))); */

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::sub_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), value_t(1),
        thrust::raw_pointer_cast(second.data()), first.size());

    cuTry(cudaDeviceSynchronize());
  }

  __host__ static void sub_scaled(vector_t &first, value_t scalar,
                                  const vector_t &second) {
    assert(first.size() == second.size());
    /* auto op = */
    /*   cu::sub_vector_scaled_op(thrust::raw_pointer_cast(first.data()),
     * scalar, */
    /*       thrust::raw_pointer_cast(second.data())); */
    /* thrust::tabulate(first.begin(), first.end(), op); */

    dim3 grid_size =
        std::ceil(static_cast<double>(first.size()) / cu::cuMaxThreads);
    dim3 block_size = std::min(cu::cuMaxThreads, first.size());

    cu::sub_vector_scaled<value_t><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(first.data()), scalar,
        thrust::raw_pointer_cast(second.data()), first.size());

    cuTry(cudaDeviceSynchronize());
  }
};

} // namespace csr
