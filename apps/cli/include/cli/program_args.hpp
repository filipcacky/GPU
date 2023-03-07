#pragma once

#include <cli/sanity.hpp>

namespace cli::program_args {

struct arguments {
  fs::path rhs_path;
  fs::path lhs_path;
  fs::path output_path;
  bool cuda;
};

bool parse(int argc, const char *argv[], arguments &args);

} // namespace cli::program_args
