#pragma once

#include <csr/matrix.hpp>
#include <memory>
#include <thrust/host_vector.h>

namespace cgm {

using host_matrix = csr::matrix<float, thrust::host_vector>;
using host_vector = thrust::host_vector<float>;

class runner {
public:
  virtual ~runner() = default;

  enum class runtime { CPU, GPU };
  static std::shared_ptr<runner> make(runtime);

  virtual std::pair<host_vector, size_t>
  solve(const host_matrix &A, const host_vector &b, float permissible_error,
        size_t max_it) = 0;
};

} // namespace cgm
