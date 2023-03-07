cmake_minimum_required(VERSION 3.25)
project(libcsr_test)

set(CMAKE_CXX_STANDARD 20)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(cpu_vector_tests vector_tests.cpp)
add_executable(cpu_matrix_tests matrix_tests.cpp)

target_link_libraries(cpu_vector_tests PUBLIC csr Boost::unit_test_framework)
target_link_libraries(cpu_matrix_tests PUBLIC csr Boost::unit_test_framework)
