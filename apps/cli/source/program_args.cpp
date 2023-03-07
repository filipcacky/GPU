#include <cli/program_args.hpp>

#include <boost/program_options.hpp>
#include <iostream>

namespace cli::program_args {

bool parse(int argc, const char *argv[], arguments &args) {
  namespace po = boost::program_options;

  po::options_description description{"Linear systems solver"};
  try {
    // clang-format off
    description.add_options()
      ("help,h", "Print options")
      ("lhs,l", po::value<std::filesystem::path>(&args.lhs_path)->required(), "LHS path")
      ("rhs,r", po::value<std::filesystem::path>(&args.rhs_path)->required(), "RHS path")
      ("output,o", po::value<std::filesystem::path>(&args.output_path)->required(), "Output path")
      ("cuda,c", po::value<bool>(&args.cuda)->required(), "Run in cuda.");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, description), vm);

    if (vm.count("help")) {
      std::cout << description << std::endl;
      return false;
    }

    po::notify(vm);

  } catch (const po::error &error) {
    LOG(error) << error.what();
    throw;
  } catch (const std::exception &ex) {
    LOG(error) << ex.what();
    throw;
  } catch (...) {
    LOG(error) << "Unknown exception.";
    throw;
  }

  return true;
}

} // namespace cli::program_args
