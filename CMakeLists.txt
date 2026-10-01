cmake_minimum_required(VERSION 3.28)
project(eva_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Keep -I flags inline; clangd cannot read nvcc --options-file response files.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Every .cu file under freeCodeCamp.org/ is a standalone program.
file(GLOB_RECURSE sources CONFIGURE_DEPENDS freeCodeCamp.org/*.cu)

foreach(src IN LISTS sources)
  # Target name from the relative path, e.g. "02 Kernels/00_vector_add.cu" -> 02_Kernels_00_vector_add
  file(RELATIVE_PATH rel "${PROJECT_SOURCE_DIR}/freeCodeCamp.org" "${src}")
  string(REGEX REPLACE "\\.cu$" "" name "${rel}")
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" name "${name}")
  add_executable(${name} "${src}")
  target_include_directories(${name} PRIVATE include)
endforeach()
