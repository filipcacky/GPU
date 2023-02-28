cmake_minimum_required(VERSION 3.13)
project(libnpy)

set(CMAKE_CXX_STANDARD 20)

add_library(npy INTERFACE)

target_include_directories(npy INTERFACE include/)
