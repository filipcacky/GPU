#include <cgm/cpu_runner.hpp>

#include <cgm/cgm.h>
#include <thrust/device_vector.h>

namespace cgm {

using device_matrix = csr::matrix<double, thrust::device_vector>;
using device_vector = thrust::device_vector<double>;

std::pair<thrust::host_vector<double>, size_t>
cpu_runner::solve(const host_matrix &A, const host_vector &b,
                  double permissible_error, size_t max_it) {
  return cgm::solve(A, b, permissible_error, max_it);
}

} // namespace cgm
