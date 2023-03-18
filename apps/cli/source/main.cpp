#include <cli/main.hpp>

int main(int argc, const char *argv[]) {
  cli::program_args::arguments args;

  try {
    if (not cli::program_args::parse(argc, argv, args))
      return 0;

    return cli::main(args);
  } catch (const std::exception &ex) {
    LOG(fatal) << ex.what();
    return 1;
  } catch (...) {
    LOG(fatal) << "Unknown exception";
    return 1;
  }

  return 0;
}
