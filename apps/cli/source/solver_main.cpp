#include <cli/main.hpp>

#include <csr/matrix.hpp>
#include <filesystem>
#include <fstream>
#include <gcm/gcm.hpp>
#include <iostream>
#include <npy/npy.hpp>

void check_path(const fs::path &path) {
  if (not fs::exists(path) or path.extension() != ".npy") {
    LOG(error) << fmt::format("Invalid input path {}", path.string());
    exit(1);
  }
}

std::pair<std::vector<double>, std::pair<size_t, size_t>>
load_matrix(const fs::path &path) {
  std::vector<size_t> shape{};
  bool fortran_order;
  std::vector<double> data;
  npy::LoadArrayFromNumpy(path.string(), shape, fortran_order, data);

  if (shape.size() != 2) {
    throw std::invalid_argument("Invalid lhs shape.");
  }

  return std::make_pair(std::move(data),
                        std::make_pair(shape.at(0), shape.at(1)));
}

std::vector<double> load_vector(const fs::path &path) {
  std::vector<size_t> shape{};
  bool fortran_order;
  std::vector<double> data;
  npy::LoadArrayFromNumpy(path.string(), shape, fortran_order, data);

  if (shape.size() != 1) {
    throw std::invalid_argument("Invalid rhs shape.");
  }

  return data;
}

int cli::main(const cli::program_args::arguments &args) {
  check_path(args.lhs_path);
  check_path(args.rhs_path);

  auto [lhs_data, lhs_shape] = load_matrix(args.lhs_path);
  auto rhs_data = load_vector(args.rhs_path);

  auto lhs = csr::from_dense(lhs_data, lhs_shape);
  auto rhs = csr::from_dense(rhs_data);

  auto [scr_result, iterations] = gcm::solve(lhs, rhs, 1e-6, 10'000);

  auto result = scr_result.to_dense();

  std::vector<size_t> result_shape{result.size()};

  npy::SaveArrayAsNumpy(args.output_path.string(), false, result_shape.size(),
                        result_shape.data(), result);

  std::cout << fmt::format("{}\n", iterations);
  std::cout << fmt::format("[{}]\n", fmt::join(result.begin(), result.end(), ","));

  return iterations > 10'000;
}
