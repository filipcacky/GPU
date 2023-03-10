#pragma once

#include <cgm/sanity.cuh>
#include <thrust/functional.h>

namespace cgm::cu {

__device__ __inline__ unsigned cuWarpActive(bool active) {
  return __ballot_sync(cuFullMask, active);
}

template <typename T>
__device__ __inline__ T cuWarpReduce(T value, unsigned mask) {
#pragma unroll
  for (size_t offset = cuWarpSize / 2; offset != 0; offset /= 2)
    value += __shfl_down_sync(mask, value, offset);

  return value;
}

template <typename T>
__global__ void
csr_dot_vec(const T *mx_data, size_t mx_data_size, const size_t *col_idx,
            size_t col_idx_size, const size_t *row_ptr, size_t row_ptr_size,
            const T *vector_data, T *output, size_t vector_size) {
  auto row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row > row_ptr_size - 1)
    return;

  const auto segment_begin = row_ptr[row];
  const auto segment_end = row_ptr[row + 1];

  output[row] = 0;
  for (size_t idx = segment_begin; idx < segment_end; ++idx)
    output[row] += vector_data[col_idx[idx]] * mx_data[idx];
}

template <typename T> struct power_op : public thrust::unary_function<T, T> {
  power_op(int l) : l_(l) {}

  __host__ __device__ T operator()(T x) const { return pow(x, l_); }

  const int l_;
};

template <typename T>
__global__ void vec_norm(const T *vector, size_t size, int l, T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned mask = cuWarpActive(threadId < size);
  if (threadId > size)
    return;

  T local_res = std::pow(vector[threadId], l);
  local_res = cuWarpReduce(local_res, mask);

  if (threadId % cuWarpSize == 0)
    atomicAdd(result, local_res);
}

template <typename T> struct scale_op : public thrust::unary_function<T, void> {
  scale_op(T s) : s_(s) {}

  __host__ __device__ void operator()(T &x) { x *= s_; }

  const T s_;
};

template <typename T> __global__ void scale(T *first, size_t size, T scale) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] *= scale;
}

template <typename T>
struct add_vector_scaled_op : public thrust::unary_function<size_t, T> {
  add_vector_scaled_op(const T *first, T scale, const T *second)
      : first_(first), scale_(scale), second_(second) {}

  __host__ __device__ T operator()(size_t idx) {
    return first_[idx] + scale_ * second_[idx];
  }

  const T *first_;
  T scale_;
  const T *second_;
};

template <typename T>
__global__ void scale_and_add(T *first, T scale, const T *second, size_t size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] *= scale;
  first[threadId] += second[threadId];
}

template <typename T>
__global__ void add_vector_scaled(T *first, T scale, const T *second,
                                  size_t size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] += scale * second[threadId];
}

template <typename T>
struct sub_vector_scaled_op : public thrust::unary_function<size_t, T> {
  sub_vector_scaled_op(const T *first, T scale, const T *second)
      : first_(first), scale_(scale), second_(second) {}

  __host__ __device__ T operator()(size_t idx) {
    return first_[idx] - scale_ * second_[idx];
  }

  const T *first_;
  T scale_;
  const T *second_;
};

template <typename T>
__global__ void sub_vector_scaled(T *first, T scale, const T *second,
                                  size_t size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] -= scale * second[threadId];
}

template <typename T>
struct mul_vector_op : public thrust::unary_function<size_t, T> {
  mul_vector_op(const T *first, const T *second)
      : first_(first), second_(second) {}

  __host__ __device__ T operator()(size_t idx) {
    return first_[idx] * second_[idx];
  }

  const T *first_;
  const T *second_;
};

template <typename T>
__global__ void multiply(const T *first, const T *second, T *out, size_t size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  out[threadId] = first[threadId] * second[threadId];
}

template <typename T>
__global__ void vec_dot_vec(const T *first, const T *second, size_t size,
                            T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned mask = cuWarpActive(threadId < size);
  if (threadId > size)
    return;

  T local_res = first[threadId] * second[threadId];
  local_res = cuWarpReduce(local_res, mask);

  if (threadId % cuWarpSize == 0)
    atomicAdd(result, local_res);
}

} // namespace cgm::cu
