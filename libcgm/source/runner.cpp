#include <cgm/runner.hpp>

#include <cgm/cpu_runner.hpp>
#include <cgm/cuda_runner.cuh>

#include <exception>

namespace cgm {

// static
std::shared_ptr<runner> runner::make(runtime rt) {
  switch (rt) {
  case runtime::CPU:
    return std::make_shared<cpu_runner>();
  case runtime::GPU:
    return std::make_shared<cuda_runner>();
  default:
    throw std::runtime_error("Not implemented.");
  }
}

} // namespace cgm
