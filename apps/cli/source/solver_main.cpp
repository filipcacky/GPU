#include <cli/main.hpp>

#include <chrono>
#include <fmt/core.h>
#include <iostream>

#include "loader.hpp"

#include <cgm/runner.hpp>
#include <thrust/host_vector.h>

void check_path(const fs::path &path) {
  if (not fs::exists(path)) {
    LOG(error) << fmt::format("Invalid input path {}", path.string());
    exit(1);
  }
}

void cuda_info() {
  int count;
  cudaGetDeviceCount(&count);

  if (count == 0) {
    LOG(error) << "No cuda enabled devices found. Exiting.\n";
    exit(1);
  }

  LOG(info) << fmt::format("Found {} CUDA enabled device(s).\n", count);

  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    LOG(info) << fmt::format("Device {}: {}\n", i, prop.name);
  }
}

int cli::main(const cli::program_args::arguments &args) {
  check_path(args.lhs_path);
  check_path(args.rhs_path);

  cuda_info();

  auto lhs = load_matrix<double>(args.lhs_path);
  auto rhs = load_vector<double>(args.rhs_path);

  if (lhs.width() != lhs.height() || rhs.size() != lhs.width()) {
    LOG(error) << "Invalid input size.";
    exit(1);
  }

  cgm::runner::runtime runtime;

  std::shared_ptr<cgm::runner> runner = cgm::runner::runner::make(
      args.cuda ? cgm::runner::runtime::GPU : cgm::runner::runtime::CPU);

  auto start = std::chrono::high_resolution_clock::now();

  auto [scr_result, iterations] = runner->solve(lhs, rhs, 1e-12, 10'000);

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::vector<double> result(scr_result.begin(), scr_result.end());

  std::vector<size_t> result_shape{scr_result.size()};

  npy::SaveArrayAsNumpy(args.output_path.string(), false, result_shape.size(),
                        result_shape.data(), result);

  std::cout << fmt::format("[{}]\n",
                           fmt::join(result.begin(), result.end(), ","));
  std::cout << fmt::format("{}it {}ns\n", iterations, duration);

  return iterations > 10'000;
}
