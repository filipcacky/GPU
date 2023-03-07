#pragma once

#include <stdio.h>

static constexpr unsigned long cuWarpSize = 32;
static constexpr unsigned long cuMaxThreads = 1024;
static constexpr unsigned cuFullMask = 0xffffffff;

template <typename T>
__device__ __host__ constexpr void cuTry(unsigned long line, const char *fn, T err) {
  if (err != cudaSuccess) {
    printf("[%ld %s] %s %s\n", line, fn, cudaGetErrorName(err),
           cudaGetErrorString(err));
  }
}

#define cuTry(x) cuTry(__LINE__, __FUNCTION__, (x))

template <typename T> T *cuHostAlloc(size_t n) {
  T *res;
  cuTry(cudaHostAlloc(&res, n * sizeof(*res), cudaHostAllocDefault));
  return res;
}

template <typename T> T *cuDeviceAlloc(size_t n) {
  T *res;
  cuTry(cudaMalloc(&res, n * (sizeof(*res))));
  return res;
}

template <typename T, typename U>
void cuCopy(T *host, T *device, size_t count, U direction) {
  cuTry(cudaMemcpy(host, device, count * sizeof(T), direction));
}

#define cuHostFree(ptr) cuTry(cudaFreeHost(ptr))

#define cuDeviceFree(ptr) cuTry(cudaFree(ptr))

template <typename T>
__device__ __inline__ T cuWarpReduce(T value, unsigned mask) {
#pragma unroll
  for (size_t offset = cuWarpSize / 2; offset != 0; offset /= 2)
    value += __shfl_down_sync(mask, value, offset);

  return value;
}
