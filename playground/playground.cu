#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "utils.cuh"

static constexpr unsigned long dataSize = 2001;

template <typename T> void generate_data(T *data, unsigned long size) {
  for (int i = 0; i < size; ++i)
    data[i] = 1;
}

template <typename T>
__global__ void vec_dot_vec(const T *first, const T *second, unsigned long size,
                            T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  T res = first[threadId] * second[threadId];

  unsigned mask = __ballot_sync(cuFullMask, threadIdx.x < size);
  res = cuWarpReduce(res, mask);

  if (threadId % warpSize == 0)
    atomicAdd(result, res);
}

template <typename T>
__global__ void vec_norm(const T *vector, unsigned long size, int l,
                         T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  T res = std::pow(vector[threadId], l);

  res = cuWarpReduce(res);

  if (threadId % warpSize == 0) {
    atomicAdd(result, res);
  }
}

template <typename T>
__global__ void add_scaled(T *first, T scale, const T *second,
                           unsigned long size, T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] += scale * second[threadId];
}

template <typename T>
__global__ void sub_scaled(T *first, T scale, const T *second,
                           unsigned long size, T *result) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] -= scale * second[threadId];
}

template <typename T>
__global__ void scale(T *first, T scale, unsigned long size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  first[threadId] *= scale;
}

template <typename T>
__global__ void multiply(const T *first, const T *second, T *out,
                         unsigned long size) {
  auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
  if (threadId > size)
    return;

  out[threadId] = first[threadId] * second[threadId];
}

int main() {
  auto data_host = cuHostAlloc<double>(dataSize);
  auto data_device = cuDeviceAlloc<double>(dataSize);
  generate_data(data_host, dataSize);
  cuCopy(data_device, data_host, dataSize, cudaMemcpyHostToDevice);

  auto result_host = cuHostAlloc<double>(dataSize);
  auto result_device = cuDeviceAlloc<double>(dataSize);
  cuCopy(result_device, result_host, 1, cudaMemcpyHostToDevice);

  dim3 grid_size = std::ceil(static_cast<double>(dataSize) / cuMaxThreads);
  dim3 block_size = std::min(cuMaxThreads, dataSize);

  vec_dot_vec<<<grid_size, block_size>>>(data_device, data_device, dataSize,
                                         result_device);
  cuTry(cudaDeviceSynchronize());
  cuCopy(result_host, result_device, 1, cudaMemcpyDeviceToHost);
  printf("Result: %f\n", *result_host);

  cuHostFree(data_host);
  cuHostFree(result_host);
  cuDeviceFree(data_device);
  cuDeviceFree(result_device);

  return 0;
}
