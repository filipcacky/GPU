#include <cgm/cpu_runner.hpp>

#include <cgm/cgm.h>
#include <thrust/device_vector.h>

namespace cgm {

std::pair<host_vector, size_t>
cpu_runner::solve(const host_matrix &A, const host_vector &b,
                  float permissible_error, size_t max_it) {
  return cgm::solve(A, b, permissible_error, max_it);
}

} // namespace cgm
