cmake_minimum_required(VERSION 3.25)
project(libcgm_test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(gpu_vector_tests vector_tests.cpp little_bit_of_indirection.cuh little_bit_of_indirection.cu)
add_executable(gpu_matrix_tests matrix_tests.cpp little_bit_of_indirection.cuh little_bit_of_indirection.cu)

target_link_libraries(gpu_vector_tests PUBLIC cgm Boost::unit_test_framework)
target_link_libraries(gpu_matrix_tests PUBLIC cgm Boost::unit_test_framework)

set_property(TARGET gpu_vector_tests PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET gpu_matrix_tests PROPERTY CUDA_ARCHITECTURES native)
