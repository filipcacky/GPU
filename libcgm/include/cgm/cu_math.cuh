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
csr_dot_vec(const T *__restrict__ mx_data, size_t mx_data_size,
            const size_t *__restrict__ col_idx, size_t col_idx_size,
            const size_t *__restrict__ row_ptr, size_t row_ptr_size,
            const T *__restrict__ vector_data, T *output, size_t vector_size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto warpId = threadId / cuWarpSize;
  const auto lane = threadId % cuWarpSize;
  const auto rowCnt = row_ptr_size - 1;
  unsigned mask = cuWarpActive(warpId < rowCnt);
  if (warpId >= rowCnt)
    return;

  T result = 0;

  const auto segment_begin = row_ptr[warpId];
  const auto segment_end = row_ptr[warpId + 1];

#pragma unroll
  for (size_t idx = segment_begin + lane; idx < segment_end; idx += cuWarpSize)
    result += vector_data[col_idx[idx]] * mx_data[idx];

  result = cuWarpReduce(result, mask);

  if (lane == 0)
    output[warpId] = result;
}

template <typename T>
__global__ void
csr_dot_vec_old(const T *__restrict mx_data, size_t mx_data_size,
                const size_t *__restrict col_idx, size_t col_idx_size,
                const size_t *__restrict row_ptr, size_t row_ptr_size,
                const T *__restrict vector_data, T *output,
                size_t vector_size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto rowCnt = row_ptr_size - 1;
  if (threadId > rowCnt)
    return;

  const auto segment_begin = row_ptr[threadId];
  const auto segment_end = row_ptr[threadId + 1];

  T result = 0;
#pragma unroll
  for (size_t idx = segment_begin; idx < segment_end; ++idx)
    result = __fma_rn(vector_data[col_idx[idx]], mx_data[idx], result);

  output[threadId] = result;
}

template <typename T> struct power_op : public thrust::unary_function<T, T> {
  power_op(int l) : l_(l) {}

  __host__ __device__ T operator()(T x) const { return pow(x, l_); }

  const int l_;
};

template <typename T>
__global__ void vec_norm(const T *__restrict__ vector, size_t size, int l,
                         T *__restrict__ result) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned mask = cuWarpActive(threadId < size);
  const auto lane = threadId % cuWarpSize;
  if (threadId >= size)
    return;

  T local_res = pow(vector[threadId], l);
  local_res = cuWarpReduce(local_res, mask);

  if (lane == 0)
    atomicAdd(result, local_res);
}

template <typename T> struct scale_op : public thrust::unary_function<T, void> {
  scale_op(T s) : s_(s) {}

  __host__ __device__ void operator()(T &x) { x *= s_; }

  const T s_;
};

template <typename T>
__global__ void scale(T *__restrict__ first, size_t size, T scale) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] *= scale;
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
__global__ void scale_and_add(T *__restrict__ first, T scale,
                              const T *__restrict__ second, size_t size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] = scale * first[idx] + second[idx];
}

template <typename T>
__global__ void add_vector_scaled(T *__restrict__ first, T scale,
                                  const T *__restrict__ second, size_t size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] = scale * second[idx] + first[idx];
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
__global__ void sub_vector_scaled(T *__restrict__ first, T scale,
                                  const T *__restrict__ second, size_t size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] = -scale * second[idx] + first[idx];
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
__global__ void multiply(const T *__restrict__ first,
                         const T *__restrict__ second, T *out, size_t size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    out[idx] = first[idx] * second[idx];
}

template <typename T>
__global__ void vec_dot_vec(const T *__restrict__ first,
                            const T *__restrict__ second, size_t size,
                            T *result) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned mask = cuWarpActive(threadId < size);
  const auto lane = threadId % cuWarpSize;
  if (threadId >= size)
    return;

  T local_res = first[threadId] * second[threadId];
  local_res = cuWarpReduce(local_res, mask);

  if (lane == 0)
    atomicAdd(result, local_res);
}

} // namespace cgm::cu
