#include <cli/main.hpp>

#include <chrono>
#include <fmt/core.h>
#include <iostream>

#include "loader.hpp"

#include <cgm/runner.hpp>
#include <thrust/host_vector.h>

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
  if (!fs::exists(args.lhs_path)) {
    LOG(error) << fmt::format("Invalid input path {}", args.lhs_path.string());
    exit(1);
  }

  auto lhs = load_matrix<float>(args.lhs_path);

  std::vector<float> rhs;
  if (!fs::exists(args.rhs_path)) {
    rhs = std::vector<float>(lhs.width(), 1);
  } else {
    auto data = load_vector<double>(args.rhs_path);
    rhs = std::vector<float>{data.begin(), data.end()};
  }

  if (lhs.width() != lhs.height() || rhs.size() != lhs.width()) {
    LOG(error) << fmt::format("Invalid input size. lhs: {}x{} rhs: {}x{}",
                              lhs.height(), lhs.width(), rhs.size(), 1);
    exit(1);
  }

  cgm::runner::runtime runtime;

  std::shared_ptr<cgm::runner> runner = cgm::runner::runner::make(
      args.cuda ? cgm::runner::runtime::GPU : cgm::runner::runtime::CPU);

  auto start = std::chrono::high_resolution_clock::now();

  auto [scr_result, iterations] = runner->solve(lhs, rhs, 1e-6, 10'000);

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::vector<float> result(scr_result.begin(), scr_result.end());

  std::vector<size_t> result_shape{scr_result.size()};

  if (!args.output_path.string().empty()) {
    npy::SaveArrayAsNumpy(args.output_path.string(), false, result_shape.size(),
                          result_shape.data(), result);
  }

  if (args.stdout) {
    std::cout << fmt::format("[{}]\n",
                             fmt::join(result.begin(), result.end(), ","));
  }

  std::cout << fmt::format("{} {} {}\n", iterations, duration, lhs.mean_nonzero());

  return 0;
}
