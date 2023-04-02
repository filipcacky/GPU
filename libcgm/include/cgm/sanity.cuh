#pragma once

#include <assert.h>
#include <stdio.h>

namespace cgm::cu {

static constexpr size_t cuWarpSize = 32;
static constexpr size_t cuSM = 48; // read from cuda info
static constexpr size_t cuMaxThreads = 512;
static constexpr unsigned cuFullMask = 0xffffffff;

__device__ __host__ void cuTry(const char *file, size_t line, const char *fn,
                               cudaError_t err) {
  if (err != cudaSuccess) {
    printf("[%s %ld %s] %s %s\n", file, line, fn, cudaGetErrorName(err),
           cudaGetErrorString(err));
  }
}

#define cuTry(x) cgm::cu::cuTry(__FILE__, __LINE__, __FUNCTION__, (x))

template <typename T> T *cuHostAlloc(size_t n, int type = cudaHostAllocDefault) {
  T* res;
  cuTry(cudaHostAlloc(&res, n * sizeof(*res), type));
  return res;
}

template <typename T> T *cuDeviceAlloc(size_t n) {
  T *res;
  cuTry(cudaMalloc(&res, n * (sizeof(*res))));
  return res;
}

template <typename T, typename U>
void cuCopy(T *to, const T *from, size_t count, U direction) {
  cuTry(cudaMemcpy(to, from, count * sizeof(T), direction));
}

#define cuHostFree(ptr) cuTry(cudaFreeHost(ptr))

#define cuDeviceFree(ptr) cuTry(cudaFree(ptr))

template <typename T> class cu_ptr {
public:
  virtual ~cu_ptr() {
    if (dealloc_)
      cuTry(dealloc_(ptr_));
  };

  const T *ptr() const { return ptr_; }
  T *ptr() { return ptr_; }

  size_t size() const { return size_; }

protected:
  cu_ptr(T *ptr, size_t size, cudaError_t (*dealloc)(void *))
      : size_(size), ptr_(ptr), dealloc_(dealloc) {}

  size_t size_;
  T *ptr_;
  cudaError_t (*dealloc_)(void *);
};

template <typename T> class cu_host_ptr;
template <typename T> class cu_device_ptr;

template <typename T> class cu_host_ptr : public cu_ptr<T> {
public:
  explicit cu_host_ptr(size_t size)
      : cu_ptr<T>(cuHostAlloc<T>(size), size, cudaFreeHost) {}

  explicit cu_host_ptr(const cu_device_ptr<T> &other)
      : cu_ptr<T>(cuHostAlloc<T>(other.size()), other.size(), cudaFreeHost) {
    cuCopy(cu_ptr<T>::ptr_, other.ptr(), other.size(), cudaMemcpyDeviceToHost);
  }

  cu_host_ptr &operator=(const cu_device_ptr<T> &other) {
    if (cu_ptr<T>::ptr_ && cu_ptr<T>::dealloc_)
      cuTry(cu_ptr<T>::dealloc_(cu_ptr<T>::ptr_));

    cu_ptr<T>::ptr_ = cuHostAlloc<T>(other.size());
    cuCopy(cu_ptr<T>::ptr_, other.ptr(), other.size(), cudaMemcpyDeviceToHost);
    cu_ptr<T>::size_ = other.size();

    return *this;
  }

  cu_host_ptr(const cu_host_ptr &) = delete;
  cu_host_ptr &operator=(const cu_host_ptr &) = delete;

  void copy_over(cu_device_ptr<T> &other) const {
    assert(cu_ptr<T>::size_ == other.size());
    cuCopy(other.ptr(), cu_ptr<T>::ptr_, other.size(), cudaMemcpyHostToDevice);
  }
};

template <typename T> class cu_device_ptr : public cu_ptr<T> {
public:
  explicit cu_device_ptr(size_t size)
      : cu_ptr<T>(cuDeviceAlloc<T>(size), size, cudaFree) {}

  explicit cu_device_ptr(const cu_host_ptr<T> &other)
      : cu_ptr<T>(cuDeviceAlloc<T>(other.size()), other.size(), cudaFree) {
    cuCopy(cu_ptr<T>::ptr_, other.ptr(), other.size(), cudaMemcpyHostToDevice);
  }

  cu_device_ptr &operator=(const cu_host_ptr<T> &other) {
    if (cu_ptr<T>::ptr_ && cu_ptr<T>::dealloc_)
      cuTry(cu_ptr<T>::dealloc_(cu_ptr<T>::ptr_));

    cu_ptr<T>::ptr_ = cuDeviceAlloc<T>(other.size());
    cuCopy(cu_ptr<T>::ptr_, other.ptr(), other.size(), cudaMemcpyHostToDevice);
    cu_ptr<T>::size_ = other.size();

    return *this;
  }

  cu_device_ptr(const cu_device_ptr &) = delete;
  cu_device_ptr &operator=(const cu_device_ptr &) = delete;

  void copy_over(cu_host_ptr<T> &other) const {
    assert(cu_ptr<T>::size_ == other.size());
    cuCopy(other.ptr(), cu_ptr<T>::ptr_, other.size(), cudaMemcpyDeviceToHost);
  }
};

} // namespace cgm::cu
