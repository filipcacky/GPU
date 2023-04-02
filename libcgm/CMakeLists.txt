cmake_minimum_required(VERSION 3.25)
project(libcgm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_library(cgm STATIC source/cuda_runner.cu source/cpu_runner.cpp source/runner.cpp)

target_include_directories(cgm PUBLIC include)
target_link_libraries(cgm PUBLIC csr Thrust::Thrust CUDA::toolkit CUDA::cudart)

set_property(TARGET cgm PROPERTY CUDA_ARCHITECTURES native)

target_link_libraries(cgm PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(cgm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

add_subdirectory(tests/)
