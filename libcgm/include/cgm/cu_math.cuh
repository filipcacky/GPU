#pragma once

#include <cgm/sanity.cuh>

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
__global__ void csr_dot_vec_32(const T *__restrict__ mx_data,
                               const size_t *__restrict__ col_idx,
                               const size_t *__restrict__ row_ptr,
                               size_t rowCnt, const T *__restrict__ vector_data,
                               T *output) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto warpId = threadId / cuWarpSize;
  const auto warpCount = (blockDim.x * gridDim.x) / cuWarpSize;
  const auto lane = threadId % cuWarpSize;

  for (size_t row = warpId; row < rowCnt; row += warpCount) {
    T result = 0;

    const auto segment_begin = row_ptr[row];
    const auto segment_end = row_ptr[row + 1];

    for (size_t idx = segment_begin + lane; idx < segment_end;
         idx += cuWarpSize)
      result += vector_data[col_idx[idx]] * mx_data[idx];

    result = cuWarpReduce(result, cuFullMask);

    if (lane == 0)
      output[row] = result;
  }
}

template <typename T>
__global__ void csr_dot_vec_1(const T *__restrict mx_data,
                              const size_t *__restrict col_idx,
                              const size_t *__restrict row_ptr, size_t rowCnt,
                              const T *__restrict vector_data, T *output) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t row = threadId; row < rowCnt; row += threadCnt) {
    const auto segment_begin = row_ptr[row];
    const auto segment_end = row_ptr[row + 1];

    T result = 0;

    for (size_t idx = segment_begin; idx < segment_end; ++idx)
      result += vector_data[col_idx[idx]] * mx_data[idx];

    output[row] = result;
  }
}

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

template <typename T>
__global__ void scale(T *__restrict__ first, size_t size, T scale) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] *= scale;
}

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
__global__ void sub_vector_scaled(T *__restrict__ first, T scale,
                                  const T *__restrict__ second, size_t size) {
  const auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const auto threadCnt = blockDim.x * gridDim.x;

  for (size_t idx = threadId; idx < size; idx += threadCnt)
    first[idx] = -scale * second[idx] + first[idx];
}

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
  const auto threadCnt = blockDim.x * gridDim.x;
  const auto lane = threadId % cuWarpSize;

  size_t idx = threadId;
  unsigned mask = cuWarpActive(idx < size);
  while (idx < size) {
    T local_res = first[idx] * second[idx];
    local_res = cuWarpReduce(local_res, mask);
    if (lane == 0)
      atomicAdd(result, local_res);

    idx += threadCnt;
    mask = cuWarpActive(idx < size);
  }
}

} // namespace cgm::cu
