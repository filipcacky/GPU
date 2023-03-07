cmake_minimum_required(VERSION 3.25)
project(libcsr)

set(CMAKE_CXX_STANDARD 20)

add_library(csr INTERFACE)

target_include_directories(csr INTERFACE include/)

add_subdirectory(tests/)
