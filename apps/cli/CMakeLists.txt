cmake_minimum_required(VERSION 3.13)
project(cli)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES source/program_args.cpp source/solver_main.cpp source/main.cpp)

find_package(Boost REQUIRED program_options log)
find_package(fmt REQUIRED)

add_executable(cli ${SOURCE_FILES})

target_include_directories(cli PUBLIC include)
target_link_libraries(cli gcm npy Boost::program_options Boost::log
                      fmt::fmt)

add_compile_options(
  "$<$<CONFIG:RELEASE>:-W;-Wall;-O2;-g;-pedantic;-fsanitize=undefined;fsanitize=address>"
)
add_compile_options("$<$<CONFIG:DEBUG>:-W;-Wall;-O0;-g;-pedantic>")
