#pragma once

#include <cgm/runner.hpp>

namespace cgm {

class cpu_runner : public runner {
public:
  std::pair<host_vector, size_t>
  solve(const host_matrix &A, const host_vector &b,
        float permissible_error, size_t max_it) override;
};

} // namespace cgm
