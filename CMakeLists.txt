cmake_minimum_required(VERSION 3.25)

project(gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

add_subdirectory(libcsr/)

add_subdirectory(libcgm/)

add_subdirectory(external/)

add_subdirectory(apps/cli/)

add_compile_options(
  "$<$<CONFIG:RELEASE>:-W;-Wall;-O2;-g;-pedantic;-fsanitize=undefined;fsanitize=address>"
  )
add_compile_options("$<$<CONFIG:DEBUG>:-W;-Wall;-O0;-g;-pedantic>")
