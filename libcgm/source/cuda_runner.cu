#include <cgm/cuda_runner.cuh>

#include <cgm/cgm.h>
#include <cgm/traits.cuh>
#include <thrust/device_vector.h>

namespace cgm {

using device_matrix = csr::matrix<double, thrust::device_vector>;
using device_vector = thrust::device_vector<double>;

std::pair<thrust::host_vector<double>, size_t>
cuda_runner::solve(const host_matrix &A, const host_vector &b,
                   double permissible_error, size_t max_it) {
  device_matrix A_device = A.transfer<thrust::device_vector>();
  device_vector b_device = b;

  auto [device_result, it] =
      cgm::solve(A_device, b_device, permissible_error, max_it);

  host_vector host_result = device_result;

  return std::make_pair(std::move(host_result), it);
}

} // namespace cgm
