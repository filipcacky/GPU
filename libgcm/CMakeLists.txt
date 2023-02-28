cmake_minimum_required(VERSION 3.13)
project(libgcm)

set(CMAKE_CXX_STANDARD 20)

add_library(gcm STATIC source/gcm.cpp)

target_include_directories(gcm PUBLIC include)
target_link_libraries(gcm PUBLIC csr)
