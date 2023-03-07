cmake_minimum_required(VERSION 3.25)
project(cli)

set(CMAKE_CXX_STANDARD 20)

set(SOURCE_FILES source/program_args.cpp source/solver_main.cpp source/main.cpp source/loader.hpp)

find_package(Boost REQUIRED program_options log)
find_package(fmt REQUIRED)

add_executable(cli ${SOURCE_FILES})

target_include_directories(cli PUBLIC include)
target_link_libraries(cli cgm npy Boost::program_options Boost::log
                      fmt::fmt)
