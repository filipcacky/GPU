cmake_minimum_required(VERSION 3.13)
project(gpu)

set(CMAKE_CXX_STANDARD 20)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

add_subdirectory(libcsr/)

add_subdirectory(libgcm/)

add_subdirectory(external/libnpy/)

add_subdirectory(apps/cli/)
